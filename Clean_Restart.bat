@echo off
title SHI Vision System - Emergency Clean Restart
color 0E
cd /d "%~dp0"

echo ===================================================
echo   SHI VISION SYSTEM - EMERGENCY CLEAN RESTART
echo ===================================================
echo This will force-kill all background vision processes,
echo release the GigE camera locks, and launch a fresh session.
echo.

echo [1/3] Terminating any stale Python & OCR processes...
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name like '%%python%%' or Name like '%%py%%'\" | Where-Object { $_.CommandLine -match 'app\.py|paddleocr_server' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1

echo [2/3] Terminating any camera utility tools holding GigE locks...
taskkill /F /IM IKapCViewer.exe /IM IKapExpert.exe /IM IKTool.exe /IM GeneralConfigurator.exe /IM FeatureImportExportTool.exe /IM IPConfigurator.exe >nul 2>&1

echo [3/3] Freeing network ports 5000, 5001, 5050...
powershell -NoProfile -Command "5000,5001,5050 | ForEach-Object { $p = $_; Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } }" >nul 2>&1

echo.
echo Pausing 2 seconds for hardware driver & camera socket to clear...
timeout /t 2 /nobreak >nul

echo.
echo Starting fresh session...
call Start_Application.bat /force

