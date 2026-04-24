"""
Seeker-01 entry point.

Launches the enabled sensor managers and the FastAPI GUI in a
single process. Flags:

    --fake-thermal     Use synthetic thermal source (no camera needed)
    --no-classifier    Skip the YOLO/shape classifier
    --host, --port     Override GUI bind address
    --no-browser       Don't auto-open Chrome
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
import webbrowser
from typing import Optional

import uvicorn

from common.config import load_config
from common.logging_setup import configure, get_logger
from eo.eo_manager import EOManager
from fusion.fusion_manager import FusionManager
from gimbal.gimbal_manager import GimbalManager
from gui.app import create_app
from radar.clustering import ClusterParams
from radar.radar_manager import RadarManager
from thermal.thermal_manager import ThermalManager


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seeker-01 Bench Test")
    p.add_argument("--fake-thermal", action="store_true", help="Synthetic thermal source")
    p.add_argument("--fake-eo", action="store_true", help="Synthetic EO (webcam) source")
    p.add_argument("--no-eo", action="store_true", help="Disable EO pipeline entirely")
    p.add_argument("--no-thermal", action="store_true",
                   help="Disable thermal pipeline entirely (EO-only mode). "
                        "Also skips the 4-index DirectShow probe that can "
                        "leave webcams in a flaky state on Windows.")
    p.add_argument("--no-classifier", action="store_true", help="Skip YOLO/shape classifier")
    p.add_argument("--no-gimbal", action="store_true", help="Disable gimbal (Maestro servo controller)")
    p.add_argument("--no-radar", action="store_true", help="Disable radar (AWR2944P)")
    p.add_argument("--host", default=None, help="GUI bind host")
    p.add_argument("--port", type=int, default=None, help="GUI bind port")
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser")
    p.add_argument("--device", default="auto",
                   help="Thermal camera device index (auto|0|1|...)")
    p.add_argument("--eo-device", default=None,
                   help="EO camera device index (auto|0|1|...). "
                        "Defaults to config eo.device_index.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cfg = load_config()
    log_cfg = cfg.get("logging", {})
    configure(
        level=str(log_cfg.get("level", "INFO")),
        log_dir=log_cfg.get("log_dir"),
        max_bytes=int(log_cfg.get("max_bytes", 5_000_000)),
        backup_count=int(log_cfg.get("backup_count", 3)),
    )
    log = get_logger(__name__)
    log.info("=" * 50)
    log.info("Seeker-01 starting (fake_thermal=%s fake_eo=%s no_eo=%s)",
             args.fake_thermal, args.fake_eo, args.no_eo)
    log.info("=" * 50)

    # Start thermal manager (unless --no-thermal). Skipping it avoids the
    # DirectShow 4-index probe that can leave the webcam in a flaky state
    # on Windows, so EO-only runs start cleanly.
    thermal: ThermalManager | None = None
    if not args.no_thermal:
        thermal = ThermalManager(
            use_fake=args.fake_thermal,
            device_index=args.device,
            enable_classifier=not args.no_classifier,
        )
        thermal.start()
    else:
        log.info("Thermal disabled (--no-thermal): skipping ThermalManager")

    # Start EO manager (optional). Disabling leaves the GUI's EO panel in
    # DISCONNECTED state; rest of the app is unaffected.
    eo: EOManager | None = None
    eo_cfg = (cfg.get("eo") or {})
    eo_enabled_in_cfg = bool(eo_cfg.get("enabled", True))
    if not args.no_eo and eo_enabled_in_cfg:
        eo_device = args.eo_device if args.eo_device is not None else eo_cfg.get("device_index", "auto")
        # Wait briefly for thermal to finish opening its camera so we can
        # exclude that index from EO's auto-probe. cv2/DirectShow does NOT
        # reliably lock devices on Windows — without this, both managers
        # race for index 0 and one ends up with a broken handle whose
        # grabs return None, leaving both panels in DISCONNECTED.
        thermal_idx: Optional[int] = None
        if thermal is not None and not args.fake_thermal:
            deadline = time.time() + 8.0
            while time.time() < deadline:
                src = getattr(thermal, "_source", None)
                if src is not None:
                    idx = getattr(src, "device_index", None)
                    if isinstance(idx, int):
                        thermal_idx = idx
                        break
                time.sleep(0.1)
            if thermal_idx is not None:
                log.info("Thermal opened on index %d — excluding from EO probe", thermal_idx)
            else:
                log.warning("Thermal not opened within 8s — EO probe may collide")
        try:
            excludes = [thermal_idx] if thermal_idx is not None else []
            eo = EOManager(
                use_fake=args.fake_eo,
                device_index=eo_device,
                enable_classifier=not args.no_classifier,
                exclude_indices=excludes,
            )
            eo.start()
        except Exception as e:
            log.warning("EO manager failed to start: %s — continuing without EO", e)
            eo = None

    # Start fusion manager — reads from the bus only, no hardware.
    # Safe to run even if only one sensor is connected.
    fusion: FusionManager | None = None
    fusion_cfg = (cfg.get("fusion") or {})
    if bool(fusion_cfg.get("enabled", True)):
        fusion = FusionManager()
        fusion.start()

    # Start radar manager — optional. Degrades gracefully if the EVM
    # is absent (publishes connected=false sentinel; GUI shows DISCONNECTED).
    radar: RadarManager | None = None
    radar_cfg = (cfg.get("radar") or {})
    if not args.no_radar and bool(radar_cfg.get("enabled", False)):
        try:
            # Build cluster params from YAML, falling back to ClusterParams
            # defaults for anything not set — so a minimal radar: block still
            # works.
            _defaults = ClusterParams()
            _trk = (radar_cfg.get("tracker") or {})
            cluster_params = ClusterParams(
                eps_pos_m=float(radar_cfg.get("cluster_eps_pos_m", _defaults.eps_pos_m)),
                eps_dop_mps=float(radar_cfg.get("cluster_eps_dop_mps", _defaults.eps_dop_mps)),
                min_samples=int(radar_cfg.get("cluster_min_samples", _defaults.min_samples)),
                min_size_m=float(radar_cfg.get("cluster_min_size_m", _defaults.min_size_m)),
                max_size_m=float(radar_cfg.get("cluster_max_size_m", _defaults.max_size_m)),
                assoc_gate_m=float(_trk.get("assoc_gate_m", _defaults.assoc_gate_m)),
                gate_growth_m_per_s=float(_trk.get("gate_growth_m_per_s", _defaults.gate_growth_m_per_s)),
                merge_overlap_m=float(_trk.get("merge_overlap_m", _defaults.merge_overlap_m)),
                coast_max_frames=int(_trk.get("coast_max_frames", _defaults.coast_max_frames)),
                coast_vel_halflife_s=float(_trk.get("coast_vel_halflife_s", _defaults.coast_vel_halflife_s)),
                confirm_min_hits=int(_trk.get("confirm_min_hits", _defaults.confirm_min_hits)),
                confirm_window=int(_trk.get("confirm_window", _defaults.confirm_window)),
                q_accel_mps2=float(_trk.get("q_accel_mps2", _defaults.q_accel_mps2)),
                r_pos_m=float(_trk.get("r_pos_m", _defaults.r_pos_m)),
                graveyard_ttl_s=float(_trk.get("graveyard_ttl_s", _defaults.graveyard_ttl_s)),
                resurrect_radius_m=float(_trk.get("resurrect_radius_m", _defaults.resurrect_radius_m)),
            )
            _ext = (radar_cfg.get("extrinsic") or {})
            radar = RadarManager(
                cli_port=radar_cfg["cli_port"],
                data_port=radar_cfg["data_port"],
                cfg_path=radar_cfg["cfg_path"],
                cli_baud=int(radar_cfg.get("cli_baud", 115200)),
                data_baud=int(radar_cfg.get("data_baud", 921600)),
                snr_min_db=float(radar_cfg.get("snr_min_db", 12.0)),
                max_range_m=float(radar_cfg.get("max_range_m", 250.0)),
                az_half_deg=float(radar_cfg.get("az_half_deg", 60.0)),
                speed_min_mps=float(radar_cfg.get("speed_min_mps", 0.0)),
                range_min_m=float(radar_cfg.get("range_min_m", 0.0)),
                profile_name=str(radar_cfg.get("profile_name", "awr2944p_ddm")),
                cluster_params=cluster_params,
                az_bias_deg=float(_ext.get("az_bias_deg", 0.0)),
                el_bias_deg=float(_ext.get("el_bias_deg", 0.0)),
            )
            radar.start()
        except KeyError as e:
            log.warning("Radar config missing key %s — disabling radar", e)
            radar = None
        except Exception as e:
            log.warning("Radar manager failed to start: %s — continuing without radar", e)
            radar = None

    # Start gimbal manager — optional. Degrades gracefully if the
    # Maestro isn't plugged in (publishes connected=false state).
    gimbal: GimbalManager | None = None
    gimbal_cfg = (cfg.get("gimbal") or {})
    if not args.no_gimbal and bool(gimbal_cfg.get("enabled", True)):
        try:
            gimbal = GimbalManager()
            gimbal.start()
        except Exception as e:
            log.warning("Gimbal manager failed to start: %s — continuing without gimbal", e)
            gimbal = None

    # Build FastAPI app. The managers are passed in so the runtime
    # config endpoints can mutate detector parameters live from the GUI.
    app = create_app(
        thermal_manager=thermal,
        eo_manager=eo,
        gimbal_manager=gimbal,
        radar_manager=radar,
        fusion_manager=fusion,
    )

    host = args.host or str(cfg.get("gui", {}).get("host", "127.0.0.1"))
    port = args.port or int(cfg.get("gui", {}).get("port", 8080))
    open_browser = not args.no_browser and bool(cfg.get("gui", {}).get("open_browser", True))

    if open_browser:
        def _delayed_open():
            time.sleep(1.0)  # let uvicorn bind first
            try:
                webbrowser.open(f"http://{host}:{port}/")
            except Exception:
                pass
        threading.Thread(target=_delayed_open, daemon=True).start()

    def _shutdown(*_):
        log.info("Shutdown signal received, stopping sensors")
        if gimbal is not None:
            gimbal.stop()
        if fusion is not None:
            fusion.stop()
        if radar is not None:
            radar.stop()
        if eo is not None:
            eo.stop()
        if thermal is not None:
            thermal.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    try:
        signal.signal(signal.SIGTERM, _shutdown)
    except (AttributeError, ValueError):
        pass  # Windows / non-main thread

    log.info("GUI -> http://%s:%d/", host, port)
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        if gimbal is not None:
            gimbal.stop()
        if fusion is not None:
            fusion.stop()
        if radar is not None:
            radar.stop()
        if eo is not None:
            eo.stop()
        if thermal is not None:
            thermal.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
