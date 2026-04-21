@echo off
REM ──────────────────────────────────────────────────────────
REM  SEEKER-01 — EO-only launcher
REM
REM  Skips the thermal pipeline entirely. Use this when the
REM  FLIR ADK isn't plugged in — otherwise the thermal probe
REM  cycles through DirectShow indices 0..3 and leaves the
REM  webcam bus in a flaky state, so EO fails to open.
REM ──────────────────────────────────────────────────────────
cd /d "%~dp0"
call "%~dp0START_SEEKER.bat" --no-thermal %*
