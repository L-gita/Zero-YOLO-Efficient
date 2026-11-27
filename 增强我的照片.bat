@echo off
setlocal
set "HERE=%~dp0"
if "%~1"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%pick_and_enhance.ps1"
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%pick_and_enhance.ps1" %*
)