# Jetson AGX Xavier deployment

This directory holds Linux-specific deployment artifacts for the
Jetson AGX Xavier port of seeker-bench.

## Files

- **seeker.service** — user-level systemd unit. Install at
  `~/.config/systemd/user/seeker.service`, then:
  ```
  systemctl --user daemon-reload
  systemctl --user enable seeker.service
  systemctl --user start seeker.service
  loginctl enable-linger $USER  # so it survives logout
  ```
- The unit runs `scripts/leopard_xu_init.py` as ExecStartPre to disable
  the FX3 trigger mode (XU 0x0b) so the camera streams free-running.

## EO RAW12 path (Linux)

The IMX568 over Leopard FX3 USB3 bridge advertises a YUYV UVC stream
on Linux but actually ships **RAW12 RGGB Bayer packed as little-endian
uint16**. This was confirmed by:

1. Decompiling LeopardCamera.dll. The Windows `LPCamera.SetParam(...,
   SENSOR_DATA_MODE.RAW12)` discards the data_type argument; the DLL
   always asks DirectShow for YUYV/16bpp. CameraTool reinterprets the
   bytes as u16 client-side.
2. Analyzing a saved CameraTool .raw frame: every byte pair is a u16
   value 0..4095 with classic RGGB Bayer phase pattern (greens ~628,
   red ~503, blue ~364).
3. Verifying the same pattern on the live Jetson stream.

Code path on Linux:

```
RawV4L2Backend.read()    -> raw YUYV bytes (H, 2W) uint8
RawV4L2Backend.grab()    -> u16 reinterpretation + p1/p99 AGC stretch
                            -> mono BGR uint8 (H, W, 3)
                            populates last_raw_stats for seeker AE loop
IMX568Capture.start()    -> Linux branch picks RawV4L2Backend, sets
                            _sdk_stream_mode=True so seekers downstream
                            takes the SDK-stream code path
```

## XU selectors (Leopard FX3 firmware on this rig)

Discovered via decompilation of LeopardCamera.dll IL:

| Selector | Size | Meaning |
|----------|------|---------|
| 0x01 | 2 | SetSensorMode |
| 0x06 | 2 | SetExposureExt |
| 0x09 | 2 | SoftTrigger (fires one shot) |
| **0x0b** | **2** | **EnableTriggerMode (write [0,0] = free-running)** |
| 0x0a | 4 | TriggerDelayTime |
| 0x0c | 256 | SensorRegisterConfiguration (bulk) |
| 0x0e | 5 | SetRegRW |
| 0x10 | 262 | I2CRegRW |

The FX3 firmware on this module boots with trigger mode armed
(XU 0x0b nonzero). Without `XU 0x0b = [0,0]` written AFTER STREAMON,
the sensor only captures on soft-trigger pulses and the V4L2 stream
returns black-level frames forever.
