# SSP Deployment Guide

This folder (`ssp_deployment`) contains the production-ready version of the SSP Fuel Lid Inspection system. It is designed to be highly reliable, easy to configure, and capable of running autonomously on a Windows Edge PC.

## Key Improvements Over Development Version
1. **`config.yaml`**: Hardcoded values (like thresholds, ports, timeouts) have been moved to `config.yaml` so the client can adjust settings without touching Python code.
2. **Auto-Cleanup**: A background thread runs daily to delete old `lid_data/` images and reports older than the `retention_days` specified in `config.yaml` (default 30 days). This prevents disk space crashes.
3. **Rotating Logs**: `print()` statements are now captured and written to `logs/ssp_app.log`. The logs are rotated automatically at 10MB (keeping the last 5 files) to prevent the log file from growing infinitely.
4. **Windows Service Installer**: Included `install_service.bat` to register the application as a background service using NSSM, ensuring it starts automatically on PC boot and restarts if it crashes.

## How to Package for the Client

To give this to the client with the highest chance of success, follow these steps:

### 1. Download NSSM
For the Windows Service installer to work, you need to bundle `nssm.exe`.
1. Download NSSM from [nssm.cc](http://nssm.cc/).
2. Extract the ZIP and copy `win64/nssm.exe` directly into this `ssp_deployment` folder.

### 2. Verify Weights
Ensure both YOLO models are present in this folder:
- `ssp_yolov8-seg.pt` (Your custom model)
- `yolov8n.pt` (Fallback)

### 3. Client Installation Steps
When you send this folder to the client, give them these instructions:

1. **Install Python 3.12**: Download and install Python 3.12 from python.org. **IMPORTANT**: Check the box "Add Python to PATH" during installation.
2. **Install Requirements**: Open Command Prompt in this folder and run:
   ```cmd
   pip install -r requirements.txt
   ```
3. **Configure**: Open `config.yaml` in Notepad and adjust any settings if needed.
4. **Install Service**: Right-click `install_service.bat` and select **"Run as Administrator"**.

The application will immediately start in the background and will continue to start automatically every time the PC is turned on. They can access the UI by navigating to `http://localhost:5000` in their browser.
