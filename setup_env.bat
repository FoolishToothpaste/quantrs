@echo off
setlocal

echo ===============================
echo   quantrs environment setup
echo ===============================

REM ── Step 1: Check/install conda ─────────────────────────────────────────────
where conda >nul 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo conda not found. Installing Miniconda...

    REM Download Miniconda installer
    curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o "%TEMP%\miniconda.exe"

    REM Run installer silently
    "%TEMP%\miniconda.exe" /InstallationType=JustMe /RegisterPython=0 /S /D=%USERPROFILE%\miniconda3

    REM Clean up installer
    del "%TEMP%\miniconda.exe"

    REM Add conda to PATH for this session
    set "PATH=%USERPROFILE%\miniconda3;%USERPROFILE%\miniconda3\Scripts;%PATH%"

    echo Miniconda installed.
) else (
    echo conda already installed. Skipping.
)

REM Make sure conda is available in this session
set "PATH=%USERPROFILE%\miniconda3;%USERPROFILE%\miniconda3\Scripts;%PATH%"

REM ── Step 2: Accept conda Terms of Service ───────────────────────────────────
echo Accepting conda Terms of Service...

call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

REM ── Step 3: Create the environment ──────────────────────────────────────────
echo Creating conda environment 'quantrs' with Python 3.10...

call conda create -n quantrs python=3.10 -y

REM ── Step 4: Activate environment ────────────────────────────────────────────
echo Activating quantrs environment...

call conda activate quantrs

REM ── Step 5: Install package ─────────────────────────────────────────────────
echo Installing quantrs...

python -m pip install -e ".[dev]"

echo.
echo ===============================
echo   Setup complete!
echo.
echo   To use quantrs, run:
echo     conda activate quantrs
echo     python
echo ===============================

endlocal
pause