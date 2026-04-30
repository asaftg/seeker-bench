@echo off
REM Double-click to launch the synchronized EO+thermal capture GUI
REM (used for the stereo extrinsic step).
REM Saves shots under recordings\calib_paired_<timestamp>\shot_NNN\
cd /d "%~dp0"
python -m scripts.calibration_capture --mode paired
if errorlevel 1 (
  echo.
  echo Capture exited with an error. Press any key to close.
  pause >nul
)
