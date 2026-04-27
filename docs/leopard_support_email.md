# Email to Leopard Imaging support

**To:** support@leopardimaging.com
**Subject:** LI-IMX568-GMSL2 - I2C register map for programmatic exposure/gain control

---

Hi,

I'm developing a real-time machine-vision application that uses an
LI-IMX568-GMSL2 module over your FX3 USB3 EVM. I need to apply manual
exposure and gain settings programmatically from my own software,
without going through CameraTool's GUI.

**What I have working:**
- `LeopardCamera.dll` loaded via pythonnet (32-bit Python). `LPCamera.Open(dev, 0, 0)` succeeds, `LPCamera.Run()` brings the stream up.
- `LPCamera.I2CRegRW(...)` is functional. Reading `subAddr=0x34, reg=0x0000, n=2` returns `[255, 15]`, so the I2C link to the sensor is alive.
- `LPCamera.ExposureExt = N` is honored — setting it stops the bridge's auto-exposure breathing on a static scene (mean span dropped from "wandering" to 1.7/255 over 30s). This alone is a big win.

**What I need:**
1. The **I2C register map** the FX3 firmware uses on the LI-IMX568-GMSL2 for:
   - Coarse integration time (true exposure)
   - Analog gain
   - Digital gain (if separate)
   - Black-level / pedestal offset
2. **Confirmation of the I2C subaddress** — is `0x34` correct for the IMX568 sensor on this module, or does the FX3 expose a different proxy address?
3. Whether your firmware uses **standard Sony SMIA register addresses** (e.g. `0x0202` COARSE_INTEG_TIME, `0x0204` ANALOG_GAIN) or has a custom mapping. Reading `0x34:0x0202:2` returns `[0, 0]`, which doesn't match the sensor's actual exposure (output mean ~100/255), so something is mapped differently.
4. **What modes `LPCamera.SetSensorMode(0..3)` selects** — linear vs HDR vs binning, etc. The LPCamera property setters for `Bits`, `Width`, `Height`, `SensorMode` all "succeed" (read back the written value) but I see no change in the captured stream over USB, so I assume these need a stream restart or are simply not supported on the GMSL2 variant.

**Why I can't just use CameraTool:**
The application is closed-loop (drone-classification metric → adjust exposure → re-acquire) and must run unattended. CameraTool is the right tool for a human operator; we need the same control via API. The standard UVC Camera Control properties (`CAP_PROP_EXPOSURE`, `CAP_PROP_AUTO_EXPOSURE`, `CAP_PROP_GAIN`) are silently ignored by the FX3 firmware on this rig — every write is dropped — which is why I'm going through `LPCamera.I2CRegRW` directly.

**Module info:**
- LI-IMX568-GMSL2 with FX3 USB3 EVM
- Windows 10/11, OpenCV via PyAV/ffmpeg-dshow
- LeopardCamera.dll v1.0 from your CameraTool Release folder

If you have an SDK doc / register cheat sheet / NDA-gated map for this module, please share. Even a list of "the registers our firmware writes when the user moves the Exposure slider in CameraTool" would unblock me completely.

Thanks,
Asaf
