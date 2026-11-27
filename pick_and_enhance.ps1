param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)][string[]]$Inputs
)

# 拖拽/选图入口（默认一次出两个版本）：
#   环境变量 ZERODCE_MODE = both（默认）→ 原版增强 + edge 增强 + AB对比 + 指标
#                         = base       → 只出原版增强
#                         = edge       → 只出 edge 版增强
$Mode = "$($env:ZERODCE_MODE)".Trim().ToLower()
if ($Mode -notin @('both', 'base', 'edge')) { $Mode = 'both' }
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms | Out-Null

$repro = Split-Path -Parent $MyInvocation.MyCommand.Path
$py    = Join-Path $repro '.venv\Scripts\python.exe'
$ab    = Join-Path $repro 'scripts\ab_enhance_compare.py'
$solo  = Join-Path $repro 'enhance.py'

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
    $dlg.Title = '选择要增强的照片（可以按住 Ctrl 多选）'
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

if ($Mode -eq 'both') {
    $outDir = Join-Path $parent '增强结果_原版与edge'
    Write-Host ''
    Write-Host "共 $($files.Count) 张，一次生成两个版本（原版 Zero-DCE + edge 版）..." -ForegroundColor Cyan
    Write-Host "  两个模型权重同为 50 轮训练，只差边缘损失这一项，对比才公平。" -ForegroundColor DarkGray
    Write-Host "输出目录：$outDir" -ForegroundColor DarkGray
    Write-Host ''
    & $py $ab @files -o $outDir
} elseif ($Mode -eq 'base') {
    $outDir = Join-Path $parent '增强结果_原版'
    Write-Host ''
    Write-Host "共 $($files.Count) 张，只用原版 Zero-DCE..." -ForegroundColor Cyan
    Write-Host "输出目录：$outDir" -ForegroundColor DarkGray
    Write-Host ''
    & $py $solo @files -c -o $outDir -m sice
} else {
    $outDir = Join-Path $parent '增强结果_edge'
    Write-Host ''
    Write-Host "共 $($files.Count) 张，只用 edge 版（带边缘损失）..." -ForegroundColor Cyan
    Write-Host "输出目录：$outDir" -ForegroundColor DarkGray
    Write-Host ''
    & $py $solo @files -c -o $outDir -m edge
}

if ($LASTEXITCODE -eq 0) {
    Write-Host ''
    if ($Mode -eq 'both') {
        Write-Host '完成！每张图会得到 4 个文件：' -ForegroundColor Green
        Write-Host '  xxx_原版增强.png   ← 原版 Zero-DCE 的成片'
        Write-Host '  xxx_edge增强.png   ← 加边缘损失后的成片'
        Write-Host '  xxx_AB对比.png     ← 原图 | 原版 | edge，下排放大看细节'
        Write-Host '  xxx_指标.png       ← 亮度 / 梯度 / L_edge 等五项指标'
        Write-Host '另外还有 overview_AB.png（总览）、metrics.csv、summary.md'
    } else {
        Write-Host '完成！' -ForegroundColor Green
    }
    Write-Host "输出目录："
    Write-Host "  $outDir"
    try { Start-Process explorer.exe $outDir } catch { }
} else {
    Write-Host '处理过程中出错了，请把上面的报错发给我。' -ForegroundColor Red
}
Write-Host ''
Read-Host '按回车键关闭窗口'