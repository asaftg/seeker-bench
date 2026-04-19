# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Seeker-01 Bench Test.

Folder distribution — output is dist/seeker_bench/seeker_bench.exe plus
DLLs, static assets, config, and models. Copy the whole folder to
another laptop and double-click the exe; no Python install required.

Why folder (not onefile):
  * Faster startup (no LZMA extract on every launch).
  * You can drop a newly trained model into models/seeker_thermal.pt
    without rebuilding.
  * PyInstaller onefile + ultralytics/torch is a minefield on Windows.

Hidden imports and collect_all calls below handle the usual PyInstaller
blind spots for FastAPI/uvicorn (dynamic protocol loading), ultralytics
(dynamic model ops), and OpenCV (native DLLs).
"""
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

block_cipher = None

# ── Third-party packages that do a lot of runtime introspection.
#    collect_all pulls binaries, data files, and hidden imports in one go.
_dynamic_pkgs = [
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.protocols",
    "uvicorn.lifespan",
    "fastapi",
    "starlette",
    "anyio",
    "h11",
    "websockets",
    "wsproto",
    "cv2",
]

datas = []
binaries = []
hiddenimports = []

for pkg in _dynamic_pkgs:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# Ultralytics + torch are optional at runtime (classifier falls back to
# shape heuristic). Only bundle them if they're installed.
for pkg in ("ultralytics", "torch", "torchvision"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# ── Our own project data: static GUI assets, yaml config, models.
datas += [
    ("gui/static", "gui/static"),
    ("config",     "config"),
]

import os
if os.path.isdir("models"):
    datas += [("models", "models")]

# Explicit hidden imports that PyInstaller misses via static analysis.
hiddenimports += [
    "numpy",
    "scipy",
    "yaml",
    "PIL",
    "PIL.Image",
    # our own package so uvicorn can import "gui.app:app" if ever needed
    "gui.app",
    "thermal.thermal_manager",
    "thermal.boson_capture",
    "thermal.fake_thermal_source",
    "thermal.heat_detector",
    "thermal.drone_classifier",
    "thermal.digital_zoom",
    "thermal.thermal_processor",
    "common.frames",
    "common.frame_bus",
    "common.config",
    "common.logging_setup",
]

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Heavy / unused — shrink the dist folder.
        "matplotlib",
        "tkinter",
        "IPython",
        "jupyter",
        "notebook",
        "pandas",
        "tests",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="seeker_bench",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # keep the console so users see crashes; flip to False for prod
    disable_windowed_traceback=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="seeker_bench",
)
