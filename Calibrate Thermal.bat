@echo off
REM Double-click to launch the thermal-only calibration capture GUI.
REM Saves shots under recordings\calib_thermal_<timestamp>\
cd /d "%~dp0"
python -m scripts.calibration_capture --mode thermal
if errorlevel 1 (
  echo.
  echo Capture exited with an error. Press any key to close.
  pause >nul
)
