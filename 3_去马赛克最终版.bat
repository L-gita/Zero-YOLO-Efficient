@echo off
chcp 65001 >nul
set ZERODCE_ENTRY=mosaic
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\entry.ps1" %*
