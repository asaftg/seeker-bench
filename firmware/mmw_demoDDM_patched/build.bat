@echo off
REM build.bat - build the patched mmw_demoDDM fork.
REM
REM Five patches applied vs the SDK source:
REM   1. mss_main.c:2606  -- disableFrameStopAsyncEvent = true
REM   2. mss_main.c:2147  -- suppress dataPathStop() in FRAME_END handler
REM   3. mss_main.c:2264  -- remove PeriodicCycle call site (cycle is no-op anyway)
REM   4. mss_main.c:3294  -- PAD_BYPASS write after Board_driversOpen
REM   5. mmw_lvds_stream.c:618 -- delete per-frame [SEEKER] HW frame printf
REM
REM TLV path on the chip (DSS + DPC + on-chip CFAR) is intentionally
REM untouched -- it's what produces the human/vehicle point cloud
REM consumed by RadarManager via UART. The lvdsStreamCfg in
REM awr2944P_unified.cfg keeps raw ADC flowing to DCA1000 for host-side
REM PMM drone detection.

setlocal

set MMWAVE_SDK_DEVICE=awr2x44P
set MMWAVE_SDK_INSTALL_PATH=C:/ti/mmwave_mcuplus_sdk_04_07_02_01
set MCU_PLUS_INSTALL_PATH=C:/ti/mcu_plus_sdk_awr2x44p_10_02_00_04
set MCU_PLUS_AWR2X44P_INSTALL_PATH=C:/ti/mcu_plus_sdk_awr2x44p_10_02_00_04
set MMWAVE_AWR294X_DFP_INSTALL_PATH=C:/ti/mmwave_dfp_02_04_18_01
set R5F_CLANG_INSTALL_PATH=C:/ti/ti-cgt-armllvm_4.0.2.LTS
set C66_CGT_INSTALL_PATH=C:/ti/ti-cgt-c6000_8.3.13
set M4_CGT_INSTALL_PATH=C:/ti/ti-cgt-armllvm_4.0.2.LTS
set SYSCONFIG_INSTALL_PATH=C:/ti/sysconfig_1.23.0
set CCS_INSTALL_PATH=C:/ti/ccs2010
set CCS_CYGWIN_PATH=C:/ti/ccs2010/ccs/utils/cygwin
set AWR2X44P_RADARSS_IMAGE_BIN=C:/ti/mmwave_dfp_02_04_18_01/firmware/radarss/xwr2x4xp_radarss_metarprc.bin
set DOWNLOAD_FROM_CCS=yes

set PATH=C:\ti\make-portable\bin;C:\ti\ccs2010\ccs\utils\cygwin;C:\Users\asaf.ruf.BLUERIVERTECH\AppData\Local\Programs\Python\Python311;%PATH%

pushd "%~dp0"
gmake -s %1 %2 %3 %4 %5 MMWAVE_SDK_DEVICE=awr2x44P
set RC=%ERRORLEVEL%
popd

endlocal & exit /b %RC%
