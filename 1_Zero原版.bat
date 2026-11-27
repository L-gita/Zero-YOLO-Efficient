@echo off
chcp 65001 >nul
set ZERODCE_ENTRY=original
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\entry.ps1" %*
