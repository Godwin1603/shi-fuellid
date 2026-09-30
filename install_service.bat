@echo off
echo ==============================================
echo SHI Production - Windows Service Installer
echo ==============================================
echo.

:: Check for Administrator privileges
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Administrator privileges confirmed.
) else (
    echo [ERROR] Please right-click this script and select "Run as Administrator".
    pause
    exit /b 1
)

:: Ensure NSSM is in the folder (we will assume the client has downloaded nssm.exe or we bundle it)
if not exist "nssm.exe" (
    echo [ERROR] nssm.exe not found in the current directory!
    echo Please download NSSM and place nssm.exe here before running this installer.
    pause
    exit /b 1
)

set SERVICE_NAME=SHI_VisionSystem
set APP_PATH=%~dp0app.py

:: We will assume Python 3.12 is in the system PATH
echo Installing %SERVICE_NAME%...
nssm install %SERVICE_NAME% "py" "-3.12 \"%APP_PATH%\""
nssm set %SERVICE_NAME% AppDirectory "%~dp0"
nssm set %SERVICE_NAME% Description "SHI Fuel Lid AI Vision Inspection System"
nssm set %SERVICE_NAME% Start SERVICE_AUTO_START
nssm set %SERVICE_NAME% AppStdout "%~dp0logs\service_stdout.log"
nssm set %SERVICE_NAME% AppStderr "%~dp0logs\service_stderr.log"

echo Installing SHI_OCR_Service...
nssm install SHI_OCR_Service "py" "-3.12 \"%~dp0paddleocr_server.py\""
nssm set SHI_OCR_Service AppDirectory "%~dp0"
nssm set SHI_OCR_Service Description "SHI OCR Microservice"
nssm set SHI_OCR_Service Start SERVICE_AUTO_START
nssm set SHI_OCR_Service AppStdout "%~dp0logs\ocr_stdout.log"
nssm set SHI_OCR_Service AppStderr "%~dp0logs\ocr_stderr.log"

echo Starting the services...
nssm start SHI_OCR_Service
nssm start %SERVICE_NAME%

echo.
echo ==============================================
echo Installation Complete!
echo The application is now running in the background.
echo You can access the UI at: http://localhost:5000
echo ==============================================
pause
