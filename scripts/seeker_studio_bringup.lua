-- Seeker production bring-up via mmWave Studio's radar-only firmware path.
--
-- IMPORTANT: Studio runs exactly ONE startup script (the Path var under
-- /Settings/Scripter/Startup Script). When we point Studio at us, the
-- original Startup.lua does NOT run -- so we have to invoke it
-- ourselves first, then layer our chip bring-up on top.
dofile([[C:\ti\mmwave_studio_03_01_04_04\mmWaveStudio\Scripts\Startup.lua]])

-- Defensive log helper: every step prints what it's doing AND the
-- return value, so when something hangs we know exactly which API call
-- went into the void.
local function step(name, fn)
    WriteToLog("[seeker] >> " .. name .. "\n", "blue")
    local ok, rv = pcall(fn)
    if not ok then
        WriteToLog("[seeker] !! " .. name .. " threw: " .. tostring(rv) .. "\n", "red")
        return nil
    end
    WriteToLog("[seeker] << " .. name .. " = " .. tostring(rv) .. "\n", "green")
    return rv
end

WriteToLog("[seeker] Studio Startup.lua finished. Beginning chip bring-up.\n", "blue")

local COM_PORT = 9       -- Last channel of AWR J10 FTDI (AR-DevPack-EVM-012).
local BSS_FW   = [[C:\ti\mmwave_studio_03_01_04_04\rf_eval_firmware\radarss\xwr2x4xp_radarss_rprc.bin]]
local MSS_FW   = [[C:\ti\mmwave_studio_03_01_04_04\rf_eval_firmware\masterss\awr2xxx_mmwave_full_mss_rprc.bin]]

-- Disable interactive dialogs that block scripted runs. Without this,
-- Studio's "Downloading Firmware to device..." modal can stall the
-- background worker even though we don't need user clicks.
RSTD.SetVar("/Settings/Automation/Automation Mode", "true")
WriteToLog("[seeker] Automation Mode set TRUE (suppress modal dialogs)\n", "blue")
RSTD.Sleep(500)

------------------------------------------------------------------------
-- 1. Reset, set SOP, connect, download firmware.
--
-- Order matters and must be exactly:
--   FullReset -> SOPControl(2) -> Connect(9, 921600) -> DownloadBSSFw
-- Confirmed working in OpenRadar / mmWave-user-recognition reference
-- scripts. Two specific points:
--   * 921600 baud, NOT 115200. Studio handshakes the chip up to 921600
--     during Connect(); the BSS download channel runs at 921600.
--   * FullReset must happen IN OUR CODE, not just in Studio's auto-init.
--     Studio's init opens GPIO ports, calls FullReset, then CLOSES
--     the GPIO ports -- so a SOPControl issued after Studio init has
--     no GPIO context. Doing FullReset ourselves re-opens GPIO.

step("FullReset",                          function() return ar1.FullReset() end)
RSTD.Sleep(500)
step("SOPControl(2)",                      function() return ar1.SOPControl(2) end)
RSTD.Sleep(500)
step("frequencyBandSelection(77G)",        function() return ar1.frequencyBandSelection("77G") end)
step("deviceVariantSelection(XWR2944P)",   function() return ar1.deviceVariantSelection("XWR2944P") end)
step("Connect("..COM_PORT..", 921600)",    function() return ar1.Connect(COM_PORT, 921600, 1000) end)
RSTD.Sleep(2000)

step("DownloadBSSFw",                      function() return ar1.DownloadBSSFw(BSS_FW) end)
RSTD.Sleep(500)
step("GetBSSFwVersion",                    function() return ar1.GetBSSFwVersion() end)
step("DownloadMSSFw",                      function() return ar1.DownloadMSSFw(MSS_FW) end)
RSTD.Sleep(500)
step("GetMSSFwVersion",                    function() return ar1.GetMSSFwVersion() end)
step("PowerOn(0,1000,0,0)",                function() return ar1.PowerOn(0, 1000, 0, 0) end)
RSTD.Sleep(1000)
step("RfEnable",                           function() return ar1.RfEnable() end)
RSTD.Sleep(500)

------------------------------------------------------------------------
-- 2. Static config: 4 TX + 4 RX, 16-bit Real ADC.
--    matches  channelCfg 15 15 0 / adcCfg 2 0  in awr2944P_unified.cfg.

step("ChanNAdcConfig", function() return
    ar1.ChanNAdcConfig(
        1, 1, 1, 1,    -- TX0..TX3 enabled
        1, 1, 1, 1,    -- RX0..RX3 enabled
        2,             -- 16-bit
        0,             -- Real format (matches our cfg adcCfg 2 0)
        0)             -- IQ swap = I first
end)
step("LPModConfig(0,0)",  function() return ar1.LPModConfig(0, 0) end)
step("RfInit",            function() return ar1.RfInit() end)
RSTD.Sleep(1500)         -- RF init runs calibration; give it time

------------------------------------------------------------------------
-- 3. Data path: LVDS, 2 lanes (max for AWR2x4xP), DDR clock.

step("DataPathConfig",  function() return ar1.DataPathConfig(513, 1216644097, 0) end)
step("LVDSLaneConfig",  function() return ar1.LVDSLaneConfig(0, 1, 1, 0, 0, 1, 0, 0) end)

------------------------------------------------------------------------
-- 4. Profile / Chirp / Frame -- matches awr2944P_unified.cfg exactly.

step("ProfileConfig", function() return ar1.ProfileConfig(
    0, 77, 12, 7, 20.81,
    0, 0, 0, 0, 0, 0, 0, 0,
    8.883, 0, 384, 30000, 2216755200,
    0, 30, 0, 0, 0) end)

step("ChirpConfig",       function() return ar1.ChirpConfig(0, 5, 0, 0, 0, 0, 0, 1, 1, 1, 1) end)
step("DisableTestSource", function() return ar1.DisableTestSource(0) end)
step("FrameConfig",       function() return ar1.FrameConfig(0, 5, 0, 128, 50, 0, 1) end)

------------------------------------------------------------------------
-- 5. DCA1000 setup -- UDP forwarding only, NO disk record.

step("GetCaptureCardDllVersion", function() return ar1.GetCaptureCardDllVersion() end)
step("SelectCaptureDevice",      function() return ar1.SelectCaptureDevice("DCA1000") end)
step("CaptureCardConfig_EthInit", function() return ar1.CaptureCardConfig_EthInit(
    "192.168.33.30", "192.168.33.180",
    "12:34:56:78:90:12", 4096, 4098) end)
step("CaptureCardConfig_Mode",        function() return ar1.CaptureCardConfig_Mode(1, 2, 1, 2, 3, 30) end)
step("CaptureCardConfig_PacketDelay", function() return ar1.CaptureCardConfig_PacketDelay(25) end)
step("GetCaptureCardFPGAVersion",     function() return ar1.GetCaptureCardFPGAVersion() end)

------------------------------------------------------------------------
-- 6. Start streaming. Order matters: chip frames FIRST so LVDS is
--    flowing, THEN DCA arms (no "No LVDS data" timeout).

step("StartFrame", function() return ar1.StartFrame() end)

WriteToLog("[seeker] Studio bring-up complete. Chip is streaming LVDS;\n", "green")
WriteToLog("[seeker] DCA1000 forwarding UDP to host:4098. No disk record.\n", "green")
