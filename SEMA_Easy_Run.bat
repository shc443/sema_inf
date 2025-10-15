@echo off
title SEMA VOC Analysis - Easy Run
echo 🎯 SEMA VOC Analysis - Easy Run
echo ===================================
echo.
echo Starting SEMA processing application...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python from https://python.org
    echo.
    pause
    exit /b 1
)

REM Run the Python GUI application
python sema_run_standalone.py

echo.
echo Press any key to exit...
pause >nul