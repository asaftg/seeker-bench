"""Check which Python libs are available for bypassing OpenCV's broken
YUY2->BGR auto-decode on this rig."""
mods = [
    "comtypes",
    "win32api",
    "win32com",
    "pythoncom",
    "winsdk",
    "imageio_ffmpeg",
    "av",
    "ffmpeg",
    "mfcap",
    "pyMediaFoundation",
    "pygrabber",
    "uvc",
]
for m in mods:
    try:
        mod = __import__(m)
        ver = getattr(mod, "__version__", "(no __version__)")
        print(f"OK  {m:<24s} {ver}")
    except Exception as e:
        print(f"--  {m:<24s} {type(e).__name__}: {e}")
