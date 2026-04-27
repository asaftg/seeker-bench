@echo off
REM ──────────────────────────────────────────────────────────
REM  SEEKER-01 — Replay the latest recorded session
REM
REM  Double-click to play back the newest recordings\seeker_*.jsonl
REM  through the existing dashboard at http://localhost:8081/.
REM
REM  Optional first arg = playback speed (e.g. 2.0, 0.5).
REM  Optional second arg = explicit JSONL path (skips --latest).
REM ──────────────────────────────────────────────────────────
cd /d "%~dp0"

set "SPEED=%~1"
if "%SPEED%"=="" set "SPEED=1.0"
set "FILE_ARG=%~2"

echo.
echo  ===========================================
echo   SEEKER-01 Replay
echo  ===========================================
echo.

REM ── Port preflight: 8081 is replay's default. If it's busy, point
REM    the user at the fix the same way START_SEEKER does.
netstat -ano | findstr /C:"127.0.0.1:8081" | findstr LISTENING >nul 2>&1
if not errorlevel 1 (
    echo.
    echo  WARNING: Port 8081 is already in use.
    echo           Probably a leftover replay server from before.
    echo.
    echo  Find the PID:  netstat -ano ^| findstr :8081
    echo  Kill it:       taskkill /F /PID ^<pid^>
    echo.
    pause
    exit /b 1
)

REM ── Auto-open the browser at 8081/?speed=N just after the server
REM    binds. start_seeker.bat doesn't need this because main.py
REM    handles its own browser open; replay_server.py doesn't.
start "" "http://localhost:8081/?speed=%SPEED%"

if "%FILE_ARG%"=="" (
    echo  Replaying NEWEST recording at %SPEED%x speed
    echo.
    python scripts\replay_server.py --latest --speed %SPEED%
) else (
    echo  Replaying %FILE_ARG% at %SPEED%x speed
    echo.
    python scripts\replay_server.py "%FILE_ARG%" --speed %SPEED%
)

if errorlevel 1 (
    echo.
    echo  Replay exited with an error.
    pause
)
