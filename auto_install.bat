@echo off
setlocal enabledelayedexpansion

echo ==============================================
echo SSP Vision System - 1-Click Installer
echo ==============================================
echo.

:: 1. Check for Python 3.12
echo Checking for Python 3.12...
set PYTHON_CMD=python
python --version 2>NUL | findstr "3.12" >NUL
if %errorLevel% == 0 (
    echo Python 3.12 detected directly!
) else (
    py -3.12 --version 2>NUL >NUL
    if !errorLevel! == 0 (
        echo Python 3.12 detected via py launcher!
        set PYTHON_CMD=py -3.12
    ) else (
        echo [ERROR] Python 3.12 is not installed or not in PATH!
        echo Please install Python 3.12 from python.org and ensure you check "Add Python to PATH".
        echo Note: PaddleOCR currently requires Python 3.12 maximum. Do not use Python 3.13+.
        pause
        exit /b 1
    )
)

:: 2. Install Python Dependencies
echo.
echo Installing Required Libraries (This may take 10-15 minutes on a fresh system)...
!PYTHON_CMD! -m pip install --upgrade pip
!PYTHON_CMD! -m pip install -r requirements.txt
echo Installing GPU PaddlePaddle Framework...
!PYTHON_CMD! -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple

:: 3. Run the Service Installer (NSSM)
echo.
echo Launching Background Service Setup...
:: Request admin privileges for NSSM
net session >nul 2>&1
if %errorLevel% == 0 (
    call install_service.bat
) else (
    echo [WARNING] Administrator privileges are required to install the background service.
    echo Please right-click "install_service.bat" and select "Run as Administrator" manually!
)

echo.
echo ==============================================
echo Installation Pipeline Completed!
echo ==============================================
pause
