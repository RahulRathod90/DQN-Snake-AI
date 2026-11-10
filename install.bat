@echo off
setlocal
echo ==============================================================================
echo           DQN Snake AI - Automated Environment Setup
echo ==============================================================================

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in your system PATH.
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Checking Python virtual environment...
if not exist ".venv" (
    echo Creating fresh virtual environment in .venv...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment .venv already exists.
)

echo [2/3] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [3/3] Upgrading pip and installing required dependencies...
python -m pip install --upgrade pip --quiet
if exist "requirements.txt" (
    pip install -r requirements.txt
) else if exist "DQN-Snake-AI\requirements.txt" (
    pip install -r DQN-Snake-AI\requirements.txt
)

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Dependency installation encountered an issue.
    pause
    exit /b 1
)

echo.
echo ==============================================================================
echo [SUCCESS] DQN Snake AI installation completed successfully!
echo To launch the application, run 'run.bat' or type: python app.py
echo ==============================================================================
pause
