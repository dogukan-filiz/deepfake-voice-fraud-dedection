@echo off
title VoxGuard Full Stack
cd /d "%~dp0"

echo [VoxGuard] Checking Python...
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

echo [VoxGuard] Checking Node.js...
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install Node.js 18+ and add to PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [VoxGuard] Creating virtual environment...
    python -m venv .venv
)

echo [VoxGuard] Installing Python dependencies...
.\.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
.\.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt

if not exist "frontend\node_modules" (
    echo [VoxGuard] Installing frontend dependencies...
    cd frontend
    call npm ci || call npm install
    cd ..
)

echo.
echo [VoxGuard] Starting backend and frontend...
echo   Backend:  http://127.0.0.1:8010
echo   Frontend: http://127.0.0.1:5173
echo.

start "VoxGuard Backend" cmd /k "cd /d "%~dp0" && .\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010"
timeout /t 3 /nobreak >nul
start "VoxGuard Frontend" cmd /k "cd /d "%~dp0\frontend" && npm run dev"
