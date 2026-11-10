@echo off
setlocal
echo ==============================================================================
echo             DQN Snake AI - Complete Test and Quality Suite
echo ==============================================================================

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

set PYTHONPATH=%CD%\DQN-Snake-AI;%CD%;%PYTHONPATH%

echo.
echo [1/2] Running Unit and Integration Test Suite (17 Tests)...
if exist "test_suite.py" (
    python -m unittest test_suite.py
) else if exist "DQN-Snake-AI\test_suite.py" (
    python -m unittest DQN-Snake-AI/test_suite.py
) else (
    echo [ERROR] test_suite.py not found.
    pause
    exit /b 1
)

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Unit test suite failed!
    pause
    exit /b 1
)

echo.
echo [2/2] Running Stress and Reliability Test Suite (7 Tests)...
if exist "stress_test.py" (
    python stress_test.py
) else if exist "DQN-Snake-AI\stress_test.py" (
    python DQN-Snake-AI\stress_test.py
) else (
    echo [ERROR] stress_test.py not found.
    pause
    exit /b 1
)

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Stress testing suite failed!
    pause
    exit /b 1
)

echo.
echo ==============================================================================
echo [SUCCESS] All 24 Verification and Stress Tests Passed!
echo ==============================================================================
pause
