@echo off
rem ASCII-only wrapper: real logic lives in scripts\save_to_git.ps1 (UTF-8 with BOM)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\scripts\save_to_git.ps1"
