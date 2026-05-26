@echo off
title VoxGuard Frontend
cd /d "%~dp0\frontend"

echo [VoxGuard] Checking Node.js...
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install Node.js 18+ and add to PATH.
    pause
    exit /b 1
)

if not exist "node_modules" (
    echo [VoxGuard] Installing frontend dependencies...
    call npm ci
    if errorlevel 1 (
        echo [VoxGuard] npm ci failed, trying npm install...
        call npm install
        if errorlevel 1 (
            echo [ERROR] npm install failed.
            pause
            exit /b 1
        )
    )
)

echo [VoxGuard] Starting frontend on http://127.0.0.1:5173 ...
call npm run dev
pause
