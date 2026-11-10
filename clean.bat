@echo off
setlocal
echo ==============================================================================
echo             DQN Snake AI - Clean Temporary Artifacts
echo ==============================================================================

echo Cleaning temporary Python cache and build directories...

if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist ".pytest_cache" rmdir /s /q ".pytest_cache"
if exist "DQN-Snake-AI\build" rmdir /s /q "DQN-Snake-AI\build"
if exist "DQN-Snake-AI\dist" rmdir /s /q "DQN-Snake-AI\dist"
if exist "DQN-Snake-AI\.pytest_cache" rmdir /s /q "DQN-Snake-AI\.pytest_cache"

for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
del /s /q *.pyc >nul 2>nul
del /s /q *.pyo >nul 2>nul

echo.
echo [SUCCESS] Temporary build artifacts and caches cleaned!
pause
