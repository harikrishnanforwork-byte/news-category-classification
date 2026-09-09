@echo off
setlocal
cd /d "%~dp0"

set "NODE_DIR=C:\Program Files\nodejs"
set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Python environment not found: %PYTHON%
    echo Create the environment and install requirements first.
    pause
    exit /b 1
)

if not exist "%NODE_DIR%\npm.cmd" (
    echo Node.js was not found at %NODE_DIR%
    echo Install Node.js, then run this file again.
    pause
    exit /b 1
)

start "NewsLens API" cmd /k "cd /d ""%~dp0"" && ""%PYTHON%"" -m uvicorn api:app --host 127.0.0.1 --port 8000"
start "NewsLens React" cmd /k "cd /d ""%~dp0frontend"" && set ""PATH=%NODE_DIR%;%PATH%"" && ""%NODE_DIR%\npm.cmd"" run dev -- --host 127.0.0.1 --port 5173"

echo.
echo NewsLens services are starting...
echo React UI: http://127.0.0.1:5173/
echo API:      http://127.0.0.1:8000/
echo.
echo Opening the React interface in your browser...
timeout /t 4 /nobreak >nul
start "" "http://127.0.0.1:5173/"
pause
