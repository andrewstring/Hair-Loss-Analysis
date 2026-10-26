@echo off
REM Scalp Analyzer - one-click Windows installer.
REM Double-click this file, or run from a terminal:
REM     install_windows.bat              (auto-detect GPU)
REM     install_windows.bat -Mode cpu    (force CPU)
REM     install_windows.bat -Recreate    (rebuild .venv)
setlocal
cd /d "%~dp0.."

if not exist "%~dp0setup_windows.ps1" (
  echo ERROR: setup_windows.ps1 must be in the same folder as this file.
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" %*
set RC=%ERRORLEVEL%

echo.
if not "%RC%"=="0" (
  echo Setup failed with exit code %RC%. See the messages above.
) else (
  echo Setup finished successfully.
)
pause
endlocal & exit /b %RC%
