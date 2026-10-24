@echo off
setlocal enabledelayedexpansion
title SHI Shift Analytics Dashboard
color 0A

echo ============================================================
echo   SHI FUEL DOOR - SHIFT ANALYTICS DASHBOARD (PORT 5050)
echo ============================================================
echo.

:: Check if dashboard is already running on port 5050
netstat -ano | findstr :5050 | findstr LISTENING >nul 2>&1
if %errorLevel% equ 0 (
    echo [INFO] Dashboard server is already active on Port 5050.
    echo Opening Dashboard in your browser...
    start http://localhost:5050
    timeout /t 2 >nul
    exit /b 0
)

echo [INFO] Starting Shift Analytics Dashboard server on Port 5050...
py -3.12 --version >nul 2>&1
if %errorLevel% equ 0 (
    set PYTHON_CMD=py -3.12
) else (
    set PYTHON_CMD=python
)

%PYTHON_CMD% dashboard_app.py
