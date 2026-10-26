@echo off
REM Scalp Analyzer - run helper.
REM   Drag an image onto this file, or:
REM     windows\run_analysis.bat C:\path\to\scalp.jpg [x20|x50|x100]
REM A JSON report is saved to outputs\<image name>.json
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1

if not exist ".venv\Scripts\python.exe" (
  echo The environment is not set up yet. Run windows\install_windows.bat first.
  pause
  exit /b 1
)

if "%~1"=="" (
  echo Usage: run_analysis.bat IMAGE [x20^|x50^|x100]
  echo Tip:   you can also drag an image file onto this script.
  pause
  exit /b 1
)

set "MAG=%~2"
if "%MAG%"=="" set "MAG=x20"

if not exist "outputs" mkdir "outputs"

".venv\Scripts\python.exe" main.py "%~1" --mag %MAG% --json "outputs\%~n1.json"
set RC=%ERRORLEVEL%

echo.
if "%RC%"=="0" (
  echo Report saved to outputs\%~n1.json
) else (
  echo Analysis failed with exit code %RC%.
)
pause
endlocal & exit /b %RC%
