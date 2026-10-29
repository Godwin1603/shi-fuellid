@echo off
setlocal enabledelayedexpansion
title SHI Shift Analytics Dashboard
color 0A

:: Ensure working directory is set to this script's directory or the project folder
cd /d "%~dp0"
if not exist "app.py" (
    if exist "%~dp0shi_deployment_fuellid\app.py" (
        cd /d "%~dp0shi_deployment_fuellid"
    )
)

echo ============================================================
echo   SHI FUEL DOOR - SHIFT ANALYTICS DASHBOARD
echo ============================================================
echo.

:: Check if the main Vision System is already running on Port 5000
netstat -ano | findstr :5000 | findstr LISTENING >nul 2>&1
if %errorLevel% equ 0 (
    echo [INFO] Application is active on Port 5000.
    echo Opening Shift Analytics Dashboard in your browser...
    start http://localhost:5000/dashboard
    ping 127.0.0.1 -n 3 >nul
    exit /b 0
)

:: If not running on 5000, launch the main application which hosts both UI and Dashboard
echo [INFO] System is not running yet.
echo Launching Start_Application.bat...
start Start_Application.bat
exit /b 0
