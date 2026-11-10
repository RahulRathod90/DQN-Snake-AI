@echo off
setlocal
echo ==============================================================================
echo           DQN Snake AI - Standalone Windows Executable Builder
echo ==============================================================================

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

set PYTHONPATH=%CD%\DQN-Snake-AI;%CD%;%PYTHONPATH%

python -c "import PyInstaller" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo PyInstaller not found in environment.
    echo Installing PyInstaller...
    pip install pyinstaller
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install PyInstaller. Please check internet connection.
        pause
        exit /b 1
    )
)

echo.
echo [1/2] Building standalone executable using snake_ai.spec...
if exist "snake_ai.spec" (
    pyinstaller snake_ai.spec --clean --noconfirm
) else if exist "DQN-Snake-AI\snake_ai.spec" (
    pyinstaller DQN-Snake-AI\snake_ai.spec --clean --noconfirm
) else (
    echo [ERROR] snake_ai.spec not found.
    pause
    exit /b 1
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] PyInstaller build failed. Check the output above.
    pause
    exit /b 1
)

echo.
echo ==============================================================================
echo [SUCCESS] Build Complete!
echo Standalone executable is available at: dist\SnakeAI\SnakeAI.exe
echo ==============================================================================
pause
