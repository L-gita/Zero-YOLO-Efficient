@echo off
REM Mode: edge.
set ZERODCE_MODE=edge
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pick_and_enhance.ps1" %*
