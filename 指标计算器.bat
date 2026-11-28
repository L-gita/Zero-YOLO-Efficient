@echo off
rem Metrics tool: pick images, compute metrics of input and final-version output
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\pick_metrics.ps1" %*
