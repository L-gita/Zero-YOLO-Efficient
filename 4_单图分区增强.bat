@echo off
set ZERODCE_ENTRY=region
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\entry.ps1" %*
