@echo off
REM Scalp Analyzer installer (Windows, CPU path).
setlocal EnableExtensions
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: 'python' not found. Install Python 3.10+ and retry.
  exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
  echo ERROR: Python 3.10+ required.
  exit /b 1
)

if not exist ".venv" (
  echo Creating virtual environment in .venv ...
  python -m venv .venv
)
call ".venv\Scripts\activate.bat"

python -m pip install --upgrade pip --quiet
python -m pip install -r requirements-base.txt
python -m pip install onnxruntime
if errorlevel 1 goto :fail

if not exist "data"    mkdir data
if not exist "models"  mkdir models
if not exist "samples" mkdir samples
if not exist "outputs" mkdir outputs

if not exist "data\calibration_reference.json" (
  > "data\calibration_reference.json" echo {"x20": {"px_per_mm": 1180.0}, "x50": {"px_per_mm": 2950.0}, "x100": {"px_per_mm": 5900.0}}
)

python check_env.py
if errorlevel 1 goto :fail

echo.
echo ======================================================
echo  Installation complete.
echo ======================================================
echo  Activate:      .venv\Scripts\activate
echo  Run analysis:  python main.py C:\path\to\scalp.jpg
echo.
endlocal
exit /b 0

:fail
echo Installation failed. See errors above.
endlocal
exit /b 1
