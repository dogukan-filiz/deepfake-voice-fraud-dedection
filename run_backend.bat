@echo off
echo Starting VoxGuard Backend on port 8010...
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010
pause
