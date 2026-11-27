@echo off
REM Mode: base.
set ZERODCE_MODE=base
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pick_and_enhance.ps1" %*
