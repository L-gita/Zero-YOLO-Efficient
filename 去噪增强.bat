@echo off
chcp 65001 >nul
set ZERODCE_ENTRY=denoise
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\three_entries.ps1" %*
