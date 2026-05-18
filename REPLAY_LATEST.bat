@echo off
REM ──────────────────────────────────────────────────────────
REM  SEEKER-01 — Replay a recorded session
REM
REM  Double-click and pick any recordings\*.jsonl from the
REM  file dialog. Plays through the dashboard at
REM  http://localhost:8081/ with pause + seek controls.
REM
REM  Optional first arg = playback speed (e.g. 2.0, 0.5).
REM ──────────────────────────────────────────────────────────
cd /d "%~dp0"

set "SPEED=%~1"
if "%SPEED%"=="" set "SPEED=1.0"

echo.
echo  ===========================================
echo   SEEKER-01 Replay
echo  ===========================================
echo.

REM ── Port preflight: 8081 is replay's default. If it's busy,
REM    bail BEFORE bothering the user with a file dialog.
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

REM ── File picker: native Win32 OpenFileDialog via PowerShell.
REM    -STA is REQUIRED — PS 5.1's default MTA host returns silently
REM    from ShowDialog() without it.
set "CHOSEN="
for /f "usebackq delims=" %%P in (`powershell -NoProfile -STA -Command "Add-Type -AssemblyName System.Windows.Forms | Out-Null; $d = New-Object System.Windows.Forms.OpenFileDialog; $d.Title = 'Pick a Seeker recording to replay'; $d.InitialDirectory = (Resolve-Path '.\recordings').Path; $d.Filter = 'Seeker recordings (*.jsonl)|*.jsonl'; $d.Multiselect = $false; if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $d.FileName }"`) do set "CHOSEN=%%P"

if "%CHOSEN%"=="" (
    echo  No file selected.
    exit /b 0
)

echo  Replaying %CHOSEN%
echo  Speed: %SPEED%x
echo.

REM ── Browser open (replay_server.py doesn't open it itself).
start "" "http://localhost:8081/?speed=%SPEED%"

python scripts\replay_server.py "%CHOSEN%" --speed %SPEED%

if errorlevel 1 (
    echo.
    echo  Replay exited with an error.
    pause
)
