@echo off
echo Starting Deepfake Detection API with TensorFlow Model
echo =======================================================

REM Set environment for TensorFlow model
set USE_TENSORFLOW_MODEL=true
set AUTH_THRESHOLD=0.5
set VITE_API_TARGET=http://127.0.0.1:8010
set VITE_WS_TARGET=ws://127.0.0.1:8010

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.ps1" (
    echo Creating virtual environment...
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements-tf.txt
) else (
    .venv\Scripts\Activate.ps1
)

REM Check if TensorFlow model exists
if not exist "models\tf_conformer\model.h5" (
    echo TensorFlow model not found!
    echo Please ensure models\tf_conformer\model.h5 exists
    pause
    exit /b 1
)

echo.
echo Starting backend on port 8010...
echo Backend will use TensorFlow Conformer model
echo.
echo API will be available at: http://127.0.0.1:8010
echo Swagger docs at: http://127.0.0.1:8010/docs
echo.
echo Press Ctrl+C to stop
echo.

REM Start the backend
python -m uvicorn backend.main_tf:app --reload --host 127.0.0.1 --port 8010

pause