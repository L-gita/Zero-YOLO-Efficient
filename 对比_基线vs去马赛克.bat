@echo off
set ZERODCE_PRE=compare
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\pick_preprocess_enhance.ps1" %*
