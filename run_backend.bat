@echo off
title VoxGuard Backend
cd /d "%~dp0"

echo [VoxGuard] Checking Python...
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [VoxGuard] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
)

::echo [VoxGuard] Installing Python dependencies...
::.\.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
::.\.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt
@REM if errorlevel 1 (
@REM     echo [ERROR] pip install failed.
@REM     pause
@REM     exit /b 1
@REM )

echo [VoxGuard] Starting backend on http://127.0.0.1:8010 ...
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010
pause
