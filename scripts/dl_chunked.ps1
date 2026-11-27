param(
  [Parameter(Mandatory=$true)][string]$Url,
  [Parameter(Mandatory=$true)][string]$Out,
  [int]$Chunks = 16,
  [int]$MaxRetry = 3
)
$ErrorActionPreference = 'Continue'
$tmp = "$Out.parts"
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

# total size via HEAD
$len = $null
& curl.exe -sIL --max-time 30 $Url | ForEach-Object {
  if ($_ -match '^\s*content-length:\s*(\d+)') { $len = [int64]$matches[1] }
}
if (-not $len) { Write-Output "FAIL: no content-length"; exit 1 }
Write-Output ("total = {0:N0} bytes ({1:N1} MB)" -f $len, ($len/1MB))

$chunk = [int64][math]::Ceiling($len / $Chunks)
$ranges = @()
for ($i = 0; $i -lt $Chunks; $i++) {
  $s = [int64]($i * $chunk)
  $e = [int64][math]::Min($s + $chunk - 1, $len - 1)
  if ($s -le $e) { $ranges += ,@($i, $s, $e) }
}

for ($attempt = 1; $attempt -le $MaxRetry; $attempt++) {
  $todo = @()
  foreach ($r in $ranges) {
    $f = Join-Path $tmp ("part{0:D3}.bin" -f $r[0])
    $want = $r[2] - $r[1] + 1
    if ((Test-Path $f) -and ((Get-Item $f).Length -eq $want)) { continue }
    $todo += ,@($r[0], $r[1], $r[2])
  }
  if ($todo.Count -eq 0) { break }
  Write-Output "attempt ${attempt}: downloading $($todo.Count) chunks"
  $procs = @()
  foreach ($r in $todo) {
    $f = Join-Path $tmp ("part{0:D3}.bin" -f $r[0])
    $procs += Start-Process -FilePath 'curl.exe' -PassThru -WindowStyle Hidden -ArgumentList @(
      '-s','-L','--retry','5','--retry-delay','2','--max-time','1800',
      '-r', ("{0}-{1}" -f $r[1], $r[2]), '-o', $f, $Url)
  }
  $procs | Wait-Process -Timeout 1800 -ErrorAction SilentlyContinue
  $procs | Where-Object { -not $_.HasExited } | Stop-Process -Force -ErrorAction SilentlyContinue
}

# verify + concatenate
$bad = 0
foreach ($r in $ranges) {
  $f = Join-Path $tmp ("part{0:D3}.bin" -f $r[0])
  $want = $r[2] - $r[1] + 1
  $got = if (Test-Path $f) { (Get-Item $f).Length } else { 0 }
  if ($got -ne $want) { Write-Output ("BAD part {0}: {1}/{2}" -f $r[0], $got, $want); $bad++ }
}
if ($bad -gt 0) { Write-Output "FAIL: $bad bad chunks"; exit 1 }

if (Test-Path $Out) { Remove-Item $Out -Force }
$fs = [System.IO.File]::Create($Out)
try {
  foreach ($r in $ranges) {
    $f = Join-Path $tmp ("part{0:D3}.bin" -f $r[0])
    $src = [System.IO.File]::OpenRead($f)
    try { $src.CopyTo($fs) } finally { $src.Dispose() }
  }
} finally { $fs.Dispose() }

$size = (Get-Item $Out).Length
if ($size -ne $len) { Write-Output ("FAIL: merged size {0} != {1}" -f $size, $len); exit 1 }
Write-Output ("OK {0} ({1:N1} MB)" -f $Out, ($size/1MB))
Remove-Item $tmp -Recurse -Force
