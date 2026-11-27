@echo off
set ZERODCE_PRE=mosaic
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\pick_preprocess_enhance.ps1" %*
