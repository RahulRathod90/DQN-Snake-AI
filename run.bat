@echo off
setlocal
echo ==============================================================================
echo             Launching DQN Snake AI Professional Suite
echo ==============================================================================

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

set PYTHONPATH=%CD%\DQN-Snake-AI;%CD%;%PYTHONPATH%

if exist "app.py" (
    python app.py
) else if exist "DQN-Snake-AI\app.py" (
    cd DQN-Snake-AI
    python app.py
) else (
    echo [ERROR] app.py not found in current directory.
    pause
    exit /b 1
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo [NOTE] Application exited with code %ERRORLEVEL%.
    pause
)
