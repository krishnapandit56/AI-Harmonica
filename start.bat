@echo off
echo ========================================================
echo       SonusAI: Musical Instrument Classifier
echo       Powered by PANNs CNN14 & FastAPI & Vite React
echo ========================================================
echo.

echo Starting FastAPI Backend (Port 8000)...
start "FastAPI Backend" cmd /k "cd backend && python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload"

echo Waiting 2 seconds for backend initialization...
timeout /t 2 /nobreak >nul

echo Starting Vite React Frontend (Port 5173)...
start "Vite React Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================================
echo Frontend: http://localhost:5173
echo Backend API Docs: http://127.0.0.1:8000/docs
echo ========================================================
echo Both servers have been launched in separate windows.
pause
