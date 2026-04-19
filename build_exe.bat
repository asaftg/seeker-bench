@echo off
REM ──────────────────────────────────────────────────────────
REM  Build a folder-distribution .exe with PyInstaller.
REM  Output: dist\seeker_bench\seeker_bench.exe
REM
REM  To distribute: copy the ENTIRE dist\seeker_bench folder
REM  (not just the .exe) to another laptop and double-click
REM  seeker_bench.exe. No Python install required on target.
REM ──────────────────────────────────────────────────────────
cd /d "%~dp0"

echo.
echo  Seeker-01 build
echo  ================

REM ── Pre-flight: make sure PyInstaller is on PATH
python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo  PyInstaller not found. Installing from requirements.txt...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo  pip install failed. Aborting.
        pause
        exit /b 1
    )
)

echo.
echo  [1/3] Cleaning previous build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo  [2/3] Running PyInstaller (this takes a few minutes)...
python -m PyInstaller seeker_bench.spec --noconfirm --clean
if errorlevel 1 (
    echo.
    echo  BUILD FAILED.
    pause
    exit /b 1
)

echo  [3/3] Smoke-checking the output...
if not exist "dist\seeker_bench\seeker_bench.exe" (
    echo  ERROR: dist\seeker_bench\seeker_bench.exe was not produced.
    pause
    exit /b 1
)
if not exist "dist\seeker_bench\_internal\gui\static\index.html" (
    if not exist "dist\seeker_bench\gui\static\index.html" (
        echo  WARNING: GUI static files not found in dist. The .exe will
        echo           still launch but the browser page will be empty.
    )
)

echo.
echo  Build complete.
echo    Run it:     dist\seeker_bench\seeker_bench.exe
echo    Distribute: copy the whole dist\seeker_bench\ folder to another laptop.
echo.
pause
