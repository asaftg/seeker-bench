@echo off
REM ──────────────────────────────────────────────────────────
REM  SEEKER-01 Bench Test — Dev Launcher
REM
REM  Double-click this file to start the app on a Python dev
REM  machine. For a clean packaged launch, use
REM  dist\seeker_bench\seeker_bench.exe instead.
REM
REM  Dependency install is a one-shot. Run `install_deps.bat`
REM  (or pass --install) if requirements.txt changes.
REM ──────────────────────────────────────────────────────────
cd /d "%~dp0"

echo.
echo  ===========================================
echo   SEEKER-01 Bench Test
echo  ===========================================
echo.

REM ── Optional: pass --install as first arg to force a reinstall
if /I "%~1"=="--install" (
    echo  Forcing dependency reinstall...
    python -m pip install -r requirements.txt --disable-pip-version-check
    shift
)

REM ── Quick import check — if core packages are missing, install;
REM    otherwise skip pip entirely so startup is instant.
python -c "import fastapi, uvicorn, cv2, numpy, yaml" >nul 2>&1
if errorlevel 1 (
    echo  Core dependencies missing. Installing from requirements.txt...
    python -m pip install -r requirements.txt --disable-pip-version-check
    if errorlevel 1 (
        echo.
        echo  ERROR: pip install failed. Is Python installed and on PATH?
        pause
        exit /b 1
    )
)

REM ── Port preflight: a zombie Seeker holding 8080 is the #1 "my
REM    changes don't apply" gotcha. If we see one, tell the user
REM    exactly how to kill it.
netstat -ano | findstr /C:"127.0.0.1:8080" | findstr LISTENING >nul 2>&1
if not errorlevel 1 (
    echo.
    echo  WARNING: Port 8080 is already in use by another process.
    echo           This is probably a leftover Seeker from before.
    echo.
    echo  Find the PID:
    echo    netstat -ano ^| findstr :8080
    echo  Kill it:
    echo    taskkill /F /PID ^<pid^>
    echo.
    echo  Then run START_SEEKER.bat again.
    pause
    exit /b 1
)

echo  Launching Seeker...
echo.
python main.py %*
if errorlevel 1 (
    echo.
    echo  Seeker exited with an error. See logs\ for details.
    pause
)
