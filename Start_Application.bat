@echo off
setlocal enabledelayedexpansion
title SHI Vision System - Production Controller
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
    echo Please make sure this script is inside the 'shi_deployment_fuellid' folder.
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

:: Check for force-restart parameter
if /i "%~1"=="/force" goto :FORCE_CLEAN
if /i "%~1"=="/restart" goto :FORCE_CLEAN

:: Check if Vision System is already listening on port 5000
netstat -ano | findstr :5000 | findstr LISTENING >nul 2>&1
if %errorLevel% equ 0 (
    echo [INFO] Port 5000 is in use. Performing health-check...
    powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:5000/status' -TimeoutSec 2 -UseBasicParsing; exit 0 } catch { exit 1 }" >nul 2>&1
    if !errorLevel! equ 0 (
        echo [INFO] SHI Vision System is already active and responsive.
        echo Opening application interface in your browser...
        start http://localhost:5000
        ping 127.0.0.1 -n 3 >nul
        exit /b 0
    ) else (
        echo [WARNING] Port 5000 is occupied by a frozen/unresponsive process.
        echo Automatically cleaning stale process to recover...
        goto :FORCE_CLEAN
    )
)

goto :START_SEQUENCE

:FORCE_CLEAN
echo [RECOVERY] Executing pre-flight process sanitization...
:: 1. Kill any competing Python & camera instances
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name like '%%python%%' or Name like '%%py%%'\" | Where-Object { $_.CommandLine -match 'app\.py|paddleocr_server' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1
taskkill /F /IM IKapCViewer.exe /IM IKapExpert.exe /IM IKTool.exe /IM GeneralConfigurator.exe /IM FeatureImportExportTool.exe /IM IPConfigurator.exe >nul 2>&1

:: 2. Free network ports
powershell -NoProfile -Command "5000,5001,5050 | ForEach-Object { $p = $_; Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } }" >nul 2>&1

:: 3. Brief hardware cool-down
timeout /t 1 /nobreak >nul

:START_SEQUENCE
echo [1/4] Checking Python 3.12 environment...
py -3.12 --version >nul 2>&1
if %errorLevel% neq 0 (
    color 4F
    echo.
    echo [ERROR] Python 3.12 is not installed or not in PATH!
    echo Please install Python 3.12 and ensure "Add to PATH" is checked.
    echo.
    pause
    exit /b
)

echo [2/4] Verifying dependencies...
py -3.12 -m pip install -r requirements.txt -q

echo [3/4] Preparing User Interface...
start /b powershell -WindowStyle Hidden -Command "Start-Sleep -Seconds 4; Start-Process 'http://localhost:5000'"

:RUN_LOOP
echo.
echo [4/4] Launching AI Backend, Camera Engine ^& Web Interface...
echo ---------------------------------------------------
echo DO NOT CLOSE THIS WINDOW.
echo The inspection system is running as long as this window is open.
echo ---------------------------------------------------
echo.

py -3.12 app.py

:: If execution reaches here, app.py exited
set EXIT_CODE=%errorLevel%
echo.
echo ===================================================
if %EXIT_CODE% equ 0 (
    echo System shut down normally.
    ping 127.0.0.1 -n 3 >nul
    exit /b 0
) else (
    color 4E
    echo [WARNING] Application stopped unexpectedly (Exit code: %EXIT_CODE%).
    echo [PRODUCTION AUTO-RECOVERY]
    echo Automatically restarting in 5 seconds to resume production...
    echo (Press Ctrl+C if you want to cancel auto-restart)
    echo ===================================================
    timeout /t 5
    color 0B
    goto :FORCE_CLEAN
)
