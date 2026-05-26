@echo off
echo Starting VoxGuard Full Stack...
start "VoxGuard Backend" cmd /k ".\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010"
timeout /t 3 /nobreak >nul
start "VoxGuard Frontend" cmd /k "cd frontend && npm run dev"
echo Backend: http://127.0.0.1:8010
echo Frontend: http://127.0.0.1:5173
