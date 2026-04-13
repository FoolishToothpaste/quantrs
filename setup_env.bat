@echo off
echo ===============================
echo   quantrs environment setup
echo ===============================

where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: conda is not installed.
    echo Please install Miniconda from https://docs.conda.io/en/latest/miniconda.html
    echo Then re-run this script.
    pause
    exit /b 1
)

echo Creating conda environment 'quantrs' with Python 3.10...
conda create -n quantrs python=3.10 -y

echo Installing quantrs...
call conda activate quantrs
pip install -e ".[dev]"

echo.
echo ===============================
echo   Setup complete!
echo.
echo   To use quantrs, run:
echo     conda activate quantrs
echo     python
echo ===============================
pause