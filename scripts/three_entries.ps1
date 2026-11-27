param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)][string[]]$Inputs
)

# 三个入口的统一实现（由同目录下的 .bat 用环境变量 ZERODCE_ENTRY 指定模式）：
#   base    → 原版 Zero-DCE（不做任何预处理）
#   denoise → 去噪版（--preprocess auto：场景自适应去噪/去块/近黑处理）
#   compare → 原版 vs 去噪版 对比（三列图 + 3× 放大 + 指标 csv）
$Entry = "$($env:ZERODCE_ENTRY)".Trim().ToLower()
if ($Entry -notin @('base', 'denoise', 'compare')) { $Entry = 'denoise' }

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms | Out-Null

$repro = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $repro 'enhance.py'))) {
    $repro = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}
$py  = Join-Path $repro '.venv\Scripts\python.exe'
$enh = Join-Path $repro 'enhance.py'
$cmp = Join-Path $repro 'scripts\compare_preprocess.py'

$title = switch ($Entry) {
    'base'    { '原版 Zero-DCE 增强（无预处理）' }
    'denoise' { '去噪版增强（场景自适应预处理）' }
    'compare' { '原版 vs 去噪版 对比' }
}

function Collect-Files($paths) {
    $out = @()
    foreach ($p in $paths) {
        if (Test-Path -LiteralPath $p -PathType Container) {
            $out += Get-ChildItem -LiteralPath $p -Recurse -File |
                    Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|bmp|tif|tiff|webp)$' } |
                    ForEach-Object { $_.FullName }
        } elseif (Test-Path -LiteralPath $p) {
            $out += (Resolve-Path -LiteralPath $p).Path
        }
    }
    return $out
}

if ($Inputs -and $Inputs.Count -gt 0) {
    $files = Collect-Files $Inputs
} else {
    $dlg = New-Object System.Windows.Forms.OpenFileDialog
    $dlg.Title = "选择照片：$title（按住 Ctrl 可多选）"
    $dlg.Filter = '图片文件|*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.webp|所有文件|*.*'
    $dlg.Multiselect = $true
    if ($dlg.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit }
    $files = @($dlg.FileNames)
}

$files = @($files | Where-Object { $_ })
if ($files.Count -eq 0) {
    [System.Windows.Forms.MessageBox]::Show('没有找到任何图片。', 'Zero-DCE') | Out-Null
    exit
}

$parent = Split-Path -Parent $files[0]

switch ($Entry) {
    'base' {
        $outDir = Join-Path $parent '原版增强结果'
        Write-Host "`n共 $($files.Count) 张 → 原版 Zero-DCE（不预处理，原汁原味）" -ForegroundColor Cyan
        Write-Host "输出目录：$outDir`n" -ForegroundColor DarkGray
        Push-Location $repro
        & $py $enh @files -m sice -c -o $outDir
        Pop-Location
    }
    'denoise' {
        $outDir = Join-Path $parent '去噪增强结果'
        Write-Host "`n共 $($files.Count) 张 → 去噪版（--preprocess auto）" -ForegroundColor Cyan
        Write-Host "输出目录：$outDir`n" -ForegroundColor DarkGray
        Push-Location $repro
        & $py $enh @files --preprocess auto -c -o $outDir
        Pop-Location
    }
    'compare' {
        $outDir = Join-Path $parent '对比_原版vs去噪'
        Write-Host "`n共 $($files.Count) 张 → 每张输出「原图 | 原版 | 去噪版」三列 + 3× 放大 + 指标" -ForegroundColor Cyan
        Write-Host "输出目录：$outDir`n" -ForegroundColor DarkGray
        & $py $cmp @files -o $outDir --mode auto
    }
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n完成！输出目录：" -ForegroundColor Green
    Write-Host "  $outDir"
    if ($Entry -eq 'compare') {
        Write-Host '  每张：xxx_对比.png；另有 metrics.csv（逐图指标）与 summary.md（汇总）'
    } else {
        Write-Host '  每张：xxx_enhanced.png（成片）+ xxx_compare.png（左右对比）'
    }
    try { Start-Process explorer.exe $outDir } catch { }
} else {
    Write-Host '处理过程中出错了，把上面的报错发我。' -ForegroundColor Red
}
Write-Host ''
Read-Host '按回车键关闭窗口'
