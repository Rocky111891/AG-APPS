@echo off
setlocal enabledelayedexpansion

echo ================================================
echo      Market Voyager Terminal Launcher
echo ================================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found in PATH. Checking for 'py' launcher...
    py --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo.
        echo [ERROR] Python is not installed or not in your PATH.
        echo Please install Python from python.org or the Microsoft Store.
        echo.
        pause
        exit /b 1
    )
    set PYTHON_CMD=py -3
) else (
    set PYTHON_CMD=python
)

echo Using Python: %PYTHON_CMD%
echo.

:: Install Requirements
if exist requirements.txt (
    echo Installing/updating dependencies...
    %PYTHON_CMD% -m pip install -r requirements.txt
) else (
    echo [WARNING] requirements.txt not found. Skipping installation.
)

echo.
echo Starting Dashboard...
echo Open your browser to: http://127.0.0.1:8050/
echo.
echo ================================================
echo      Press Ctrl+C to stop the server
echo ================================================
echo.

%PYTHON_CMD% app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The application crashed or was stopped.
    pause
)
