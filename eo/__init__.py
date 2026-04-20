"""EO (electro-optical / RGB) pipeline.

Phase B Ticket 3 — emulates the Leopard LI-USB30-IMX568-GMSL2 + Commonlands
CIL350 (35mm EFL, F/2.4, M12 mount, NIR-compatible) telephoto lens using
any USB webcam. Mirrors the `thermal/` package structure: capture + fake
+ classifier + manager + __main__.

The real optics are narrow: ~11° H × 9.2° V on the IMX568's 6.77×5.65mm
active area (2472×2064 px @ 2.74µm). A webcam will be much wider — we
don't try to match the FOV, we match the pipeline shape. Config carries
the real FOV numbers so the gimbal math (Ticket 4) uses IMX568 values
regardless of which camera is currently plugged in.
"""
