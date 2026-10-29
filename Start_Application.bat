@echo off
setlocal enabledelayedexpansion
title SHI Vision System
color 0B

:: Ensure working directory is set to this script's directory or the project folder
cd /d "%~dp0"
if not exist "app.py" (
    if exist "%~dp0shi_deployment_fuellid\app.py" (
        cd /d "%~dp0shi_deployment_fuellid"
    )
)

if not exist "app.py" (
    color 4F
    echo.
    echo ===================================================
    echo                   SETUP ERROR
    echo ===================================================
    echo Cannot find 'app.py' in:
    echo !CD!
    echo.
    echo Please make sure this script is inside the 'shi_deployment_fuellid' folder
    echo or create a Desktop Shortcut pointing to Start_Application.bat.
    echo ===================================================
    echo.
    pause
    exit /b 1
)

echo ===================================================
echo   SHI VISION SYSTEM & SHIFT ANALYTICS DASHBOARD
echo ===================================================
echo [INFO] System Web Endpoints:
echo   * Live Inspection: http://localhost:5000
echo   * Shift Analytics: http://localhost:5000/dashboard
echo ===================================================
echo.

:: Check if Vision System is already running on port 5000
netstat -ano | findstr :5000 | findstr LISTENING >nul 2>&1
if %errorLevel% equ 0 (
    echo [INFO] SHI Vision System is already running on Port 5000.
    echo Opening application interface in your browser...
    start http://localhost:5000
    ping 127.0.0.1 -n 3 >nul
    exit /b 0
)
echo [1/4] Checking system environment...
py -3.12 --version >nul 2>&1
if %errorLevel% neq 0 (
    color 4F
    echo.
    echo [ERROR] Python 3.12 is not installed or not in PATH!
    echo Please install Python 3.12 and ensure "Add to PATH" is checked.
    echo.
    echo Please take a photo of this screen and send it to support.
    pause
    exit /b
)

echo [2/4] Verifying and installing missing dependencies...
py -3.12 -m pip install -r requirements.txt -q
echo.
echo [3/4] Preparing User Interface...
:: Start a background timer to open the browser after 5 seconds
start /b powershell -WindowStyle Hidden -Command "Start-Sleep -Seconds 5; Start-Process 'http://localhost:5000'"

echo [4/4] Launching AI Backend, OCR & Web Servers...
echo.
echo ---------------------------------------------------
echo DO NOT CLOSE THIS WINDOW. 
echo The system is running as long as this window is open.
echo ---------------------------------------------------
:: The OCR server is now automatically spawned invisibly by app.py in the background

:: Run the application in this exact window so errors are visible
py -3.12 app.py

:: If we reach this line, the application was closed or it crashed.
if %errorLevel% neq 0 (
    color 4F
    echo.
    echo ===================================================
    echo                   CRITICAL ERROR
    echo ===================================================
    echo The SHI Vision System encountered an error and stopped!
    echo.
    echo Please scroll up to read the specific error message above.
    echo Take a photo/screenshot of this entire window and send it to support.
    echo ===================================================
    echo.
    pause
) else (
    echo.
    echo System shut down normally.
    ping 127.0.0.1 -n 3 >nul
)
