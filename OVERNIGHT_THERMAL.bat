@echo off
REM ============================================================
REM Overnight thermal optimization launcher.
REM
REM Runs the parameter-sweep harness in a loop with auto-resume.
REM Survives transient camera/IO failures and Python process exits.
REM
REM Operator workflow:
REM   1. Position gimbal at the desired pose (in the GUI).
REM   2. Stop seeker so the harness can grab the camera handle.
REM   3. Double-click this .bat. Walk away.
REM   4. Results land in recordings\optim\<pose>\
REM
REM Stop the run cleanly by creating the sentinel file:
REM   echo. > recordings\optim\<pose>\STOP
REM ============================================================

cd /d "%~dp0"

set POSE=garage_overnight
if not "%~1"=="" set POSE=%~1

echo [overnight] pose: %POSE%
echo [overnight] press Ctrl-C in this window to abort
echo [overnight] or create recordings\optim\%POSE%\STOP to stop cleanly
echo.

python scripts\_thermal_optim_launcher.py --pose %POSE% --frames 30
set RC=%ERRORLEVEL%
echo.
echo [overnight] launcher exited with rc=%RC%
if %RC%==0 (
  echo [overnight] SUCCESS
) else (
  echo [overnight] FAILED — check the output above and recordings\optim\%POSE%\HEARTBEAT.txt
)
pause
