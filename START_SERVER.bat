@echo off
title Water Body Segmentation AI - Server
color 0B

echo.
echo  ============================================================
echo    Water Body Segmentation AI  ^|  Starting Server...
echo  ============================================================
echo.

:: Move to the project folder
cd /d "%~dp0"

:: Detect Python Executable
set "PY_CMD=python"
if exist "C:\Users\Abhis\.anaconda-desktop\micromamba\envs\cpu\python.exe" (
    set "PY_CMD=C:\Users\Abhis\.anaconda-desktop\micromamba\envs\cpu\python.exe"
)

:: Check Python
%PY_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not installed or not in PATH!
    echo  Please install Python from https://python.org
    pause
    exit /b 1
)

:: Check if model exists
if not exist "models\water_unet.h5" (
    echo  [WARNING] Trained model not found at models\water_unet.h5
    echo  The server will still start but predictions will fail.
    echo  Run train_all.py first to train the model.
    echo.
)

:: Install requirements silently if flask is missing
%PY_CMD% -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo  [INFO] Installing required packages...
    %PY_CMD% -m pip install -r requirements.txt
    echo.
)

echo  [OK] Starting Flask server on http://127.0.0.1:5050
echo  [OK] Opening browser...
echo.
echo  Press Ctrl+C to stop the server.
echo  ============================================================
echo.

:: Open browser after 2 seconds
start "" timeout /t 2 >nul && start "" "http://127.0.0.1:5050"

:: Start the Flask server
%PY_CMD% server.py

pause
