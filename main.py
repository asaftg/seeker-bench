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
from common.frame_bus import BUS
from common.frames import Topic
from common.logging_setup import configure, get_logger
from eo.eo_manager import EOManager
from fusion.fusion_manager import FusionManager
from gimbal.gimbal_manager import GimbalManager
from gui.app import create_app
from radar.clustering import ClusterParams
from radar.radar_manager import RadarManager
from radar.composite_manager import CompositeRadarBackend
from radar_dca.data_port import DataPortListener
from radar_dca.dca_control import DCAControl
from radar_dca.dca_pipeline import DCAPipeline, dims_from_cfg, dims_from_cfg_file
from recording.jsonl_recorder import JSONLRecorder
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
    p.add_argument(
        "--radar-firmware",
        choices=["demoDDM"],
        default="demoDDM",
        help="Radar firmware backend. Single supported mode: chip runs "
             "TI's mmw_demoDDM (or our patched fork). Chip emits TLV "
             "(humans/vehicles via on-chip CFAR) AND raw ADC over LVDS "
             "(host PMM for drones) simultaneously. Other modes (studio, "
             "studio-py, external) deprecated -- they bypassed the TLV "
             "consumer and broke human/vehicle detection.",
    )
    p.add_argument("--host", default=None, help="GUI bind host")
    p.add_argument("--port", type=int, default=None, help="GUI bind port")
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser")
    p.add_argument("--device", default="auto",
                   help="Thermal camera device index (auto|0|1|...)")
    p.add_argument("--eo-device", default=None,
                   help="EO camera device index (auto|0|1|...). "
                        "Defaults to config eo.device_index.")
    p.add_argument("--auto-record", action="store_true",
                   help="Begin recording immediately at app launch "
                        "(overrides config recording.auto_start). Useful "
                        "for unattended capture and remote/CI runs.")
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

    # Startup order depends on the EO backend:
    #   sensor=imx568 — start EO first, then exclude EO's index from thermal's
    #     probe. The IMX568 opens reliably at a known native resolution, and
    #     once it's streaming we must NOT let thermal's DirectShow probe
    #     open+configure the same index or the IMX568's handle gets kicked.
    #   sensor=webcam (or fake-eo) — keep the legacy order (thermal first,
    #     then EO waits up to 8s for thermal's index). This order was the
    #     only way that worked before the IMX568 was online.
    eo_cfg = (cfg.get("eo") or {})
    eo_enabled_in_cfg = bool(eo_cfg.get("enabled", True))
    eo_is_imx568 = (str(eo_cfg.get("sensor", "webcam")).lower() == "imx568"
                    and not args.fake_eo and not args.no_eo and eo_enabled_in_cfg)

    thermal: ThermalManager | None = None
    eo: EOManager | None = None

    def _start_thermal(exclude_indices: list[int]) -> None:
        nonlocal thermal
        if args.no_thermal:
            log.info("Thermal disabled (--no-thermal): skipping ThermalManager")
            return
        thermal = ThermalManager(
            use_fake=args.fake_thermal,
            device_index=args.device,
            enable_classifier=not args.no_classifier,
            exclude_indices=exclude_indices,
        )
        thermal.start()

    def _start_eo(exclude_indices: list[int]) -> None:
        nonlocal eo
        if args.no_eo or not eo_enabled_in_cfg:
            return
        eo_device = args.eo_device if args.eo_device is not None else eo_cfg.get("device_index", "auto")
        try:
            eo = EOManager(
                use_fake=args.fake_eo,
                device_index=eo_device,
                enable_classifier=not args.no_classifier,
                exclude_indices=exclude_indices,
            )
            eo.start()
        except Exception as e:
            log.warning("EO manager failed to start: %s — continuing without EO", e)
            eo = None

    def _wait_for_device_index(mgr, deadline_s: float) -> Optional[int]:
        deadline = time.time() + deadline_s
        while time.time() < deadline:
            src = getattr(mgr, "_source", None)
            if src is not None:
                idx = getattr(src, "device_index", None)
                if isinstance(idx, int):
                    return idx
            time.sleep(0.1)
        return None

    if eo_is_imx568:
        # EO-first order: IMX568 claims its index, then thermal is told to
        # skip it.
        _start_eo(exclude_indices=[])
        eo_idx: Optional[int] = None
        if eo is not None:
            # 25s — covers SDK stream cold-start (~5s nominal, ~15s if
            # the FX3 bridge is stuck in a prior CameraTool resolution)
            # AND the PyAV fallback path (~10s prelude + open). Without
            # this, thermal index-probe starts in parallel and the
            # DSHOW VideoCapture::open warnings on the EO USB device
            # disturb the SDK helper handshake → STREAM_HDR timeout →
            # whole EO path fails over to PyAV which then can't open
            # because the helper is still alive holding the camera.
            eo_idx = _wait_for_device_index(eo, 25.0)
            if eo_idx is not None:
                log.info("EO(IMX568) opened on index %d — excluding from thermal probe", eo_idx)
            else:
                log.warning("EO(IMX568) not opened within 25s — thermal probe may collide")
        _start_thermal(exclude_indices=[eo_idx] if eo_idx is not None else [])
    else:
        # Legacy order: thermal first, EO excludes thermal's index.
        _start_thermal(exclude_indices=[])
        thermal_idx: Optional[int] = None
        if thermal is not None and not args.fake_thermal:
            thermal_idx = _wait_for_device_index(thermal, 8.0)
            if thermal_idx is not None:
                log.info("Thermal opened on index %d — excluding from EO probe", thermal_idx)
            else:
                log.warning("Thermal not opened within 8s — EO probe may collide")
        _start_eo(exclude_indices=[thermal_idx] if thermal_idx is not None else [])

    # Start fusion manager — reads from the bus only, no hardware.
    # Safe to run even if only one sensor is connected.
    fusion: FusionManager | None = None
    fusion_cfg = (cfg.get("fusion") or {})
    if bool(fusion_cfg.get("enabled", True)):
        fusion = FusionManager()
        fusion.start()

    # Start gimbal manager early — independent of every other sensor and
    # we don't want it blocked by long downstream startups (the radar
    # path's mmW Studio bring-up waits up to 90 s for the first DCA1000
    # packet on UDP:4098, which previously starved the gimbal of any
    # initialization until that wait completed or timed out — visible
    # in seeker.log as "Studio launched ... Waiting for first DCA1000
    # packet" with NO subsequent "GimbalManager started" line).
    gimbal: GimbalManager | None = None
    gimbal_cfg = (cfg.get("gimbal") or {})
    if not args.no_gimbal and bool(gimbal_cfg.get("enabled", True)):
        try:
            gimbal = GimbalManager()
            gimbal.start()
        except Exception as e:
            log.warning("Gimbal manager failed to start: %s — continuing without gimbal", e)
            gimbal = None

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
                range_aware_clamp=bool(radar_cfg.get(
                    "range_aware_clamp", _defaults.range_aware_clamp)),
                range_aware_min_size_m=float(radar_cfg.get(
                    "range_aware_min_size_m", _defaults.range_aware_min_size_m)),
                range_aware_size_per_meter_m=float(radar_cfg.get(
                    "range_aware_size_per_meter_m",
                    _defaults.range_aware_size_per_meter_m)),
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
                # 0 disables the legacy "no TLV → reconnect" path; the
                # unified cfg's lvdsStreamCfg suppresses UART TLV on
                # this firmware so the timeout would fire forever and
                # the reconnect would stomp LVDS.
                stream_timeout_s=float(radar_cfg.get("stream_timeout_s", 3.0)),
                cluster_params=cluster_params,
                az_bias_deg=float(_ext.get("az_bias_deg", 0.0)),
                el_bias_deg=float(_ext.get("el_bias_deg", 0.0)),
            )
            # demoDDM mode: RadarManager owns the chip via UART CLI
            # (cfg push, sensorStart, TLV consumer producing the
            # humans/vehicles target list on Topic.RADAR). Always start
            # it -- skipping this is what broke detection in the prior
            # external/studio modes.
            radar.start()
            # Phase 3: build the composite that wraps RadarManager + the
            # DCA1000 raw-ADC pipeline. The composite owns mode dispatch
            # (Stock / A/G / A/A) and the raw-ADC consumer. RadarManager
            # is already started; composite.start() is idempotent and
            # adds the DCA control plane + UDP listener + PMM pipeline
            # alongside.
            try:
                _dca = (radar_cfg.get("dca") or {})
                _dca_control = None
                _dca_listener = None
                _dca_pipeline = None
                if bool(_dca.get("enabled", True)):
                    _host_ip = str(_dca.get("host_ip", "192.168.33.30"))
                    _dca_ip = str(_dca.get("dca_ip", "192.168.33.180"))
                    _cfg_port = int(_dca.get("config_port", 4096))
                    _udp_port = int(_dca.get("data_udp_port", 4098))
                    _dca_control = DCAControl(
                        host_ip=_host_ip, dca_ip=_dca_ip,
                        config_port=_cfg_port, data_port=_udp_port,
                    )
                    _dca_listener = DataPortListener(
                        host_ip=_host_ip, data_port=_udp_port,
                    )
                    # Parse the actual chip cfg so the wire-byte layout
                    # (n_chirps, n_samples, n_rx, chirp_period) matches
                    # what mmw_demoDDM is emitting. Falls back to
                    # hard-coded defaults only if the cfg can't be
                    # parsed — in which case A/A will likely refuse to
                    # reshape frames and we'll see frames_dropped > 0.
                    try:
                        _dims = dims_from_cfg_file(str(radar_cfg["cfg_path"]))
                        log.info(
                            "DCA dims from cfg: %d chirps × %d RX × %d samples, "
                            "PRF %.0f Hz, range res %.2f m → %.1f MB/frame",
                            _dims.n_chirps, _dims.n_rx, _dims.n_samples,
                            _dims.prf_hz, _dims.range_resolution_m,
                            _dims.bytes_per_frame / 1e6,
                        )
                    except Exception as e:
                        log.warning(
                            "Could not parse cfg for DCA dims (%s); "
                            "falling back to hard-coded defaults", e,
                        )
                        _dims = dims_from_cfg()
                    # New DCAPipeline owns its own per-mode defaults; the
                    # composite calls set_mode/set_params after start() to
                    # activate the operator's saved knobs from radar_modes.json.
                    #
                    # pmm_only=True: in hybrid demoDDM mode the chip's
                    # on-chip CFAR provides humans/vehicles via TLV
                    # (RadarManager + Topic.RADAR). DCAPipeline only
                    # needs to do PMM for drones on raw ADC -- skipping
                    # Stage 3 (Doppler FFT) + Stage 4 (CFAR/AoA) +
                    # tracker frees CPU and avoids the host-side
                    # AoA-without-DDMA-unfold bug.
                    _dca_pipeline = DCAPipeline(
                        listener=_dca_listener,
                        dims=_dims,
                        profile_name="awr2944p_unified",
                        max_range_m=float(radar_cfg.get("max_range_m", 250.0)),
                        az_half_deg=float(radar_cfg.get("az_half_deg", 60.0)),
                        pmm_only=True,
                    )
                # Build composite. start() is idempotent; the inner
                # RadarManager.start() above is also called inside, but
                # is a no-op since the threads already exist.
                composite = CompositeRadarBackend(
                    radar=radar,
                    dca_control=_dca_control,
                    dca_listener=_dca_listener,
                    dca_pipeline=_dca_pipeline,
                    initial_mode="stock",
                    radar_firmware=args.radar_firmware,
                )
                # Run composite.start() in a daemon thread. The mmW
                # Studio bring-up inside it waits up to 90 s for the
                # first DCA1000 packet on UDP:4098 — historically
                # blocked main.py from reaching GUI startup until the
                # wait completed or timed out. Composite construction
                # above is non-blocking, so `radar = composite` is
                # safe to assign IMMEDIATELY; create_app downstream
                # gets a valid reference even if the chip bring-up is
                # still in progress when the GUI binds. The composite
                # publishes connected=False sentinels until the DCA
                # path is alive — same UX as a missing EVM.
                _comp_starter = threading.Thread(
                    target=composite.start,
                    name="composite_start",
                    daemon=True,
                )
                _comp_starter.start()
                # Use composite as THE radar reference everywhere — it
                # exposes the same set_tuning/set_extrinsic/diagnostics
                # surface as RadarManager and additionally drives mode.
                radar = composite
                # Restore any per-mode saved sliders so the operator's
                # last save survives the restart. Each mode is an
                # independent snapshot in config/radar_modes.json.
                try:
                    import json as _json
                    from pathlib import Path as _Path
                    _saved_path = _Path("config/radar_modes.json")
                    if _saved_path.exists():
                        _saved = _json.loads(_saved_path.read_text("utf-8"))
                        # Apply Stock first as the baseline if present.
                        if "stock" in _saved:
                            stock_p = _saved["stock"]
                            radar.set_tuning(
                                snr_min_db=stock_p.get("snr_min_db"),
                                az_half_deg=stock_p.get("az_half_deg"),
                                speed_min_mps=stock_p.get("speed_min_mps"),
                                range_min_m=stock_p.get("range_min_m"),
                                cluster_eps_pos_m=stock_p.get("cluster_eps_pos_m"),
                                cluster_min_samples=stock_p.get("cluster_min_samples"),
                            )
                        # A/G mode currently has no host-side DSP knobs
                        # to restore — it's a filter preset on Stock TLV
                        # (see radar/composite_manager.py:_MODE_FILTERS).
                        # The integrate_chirps/cfar_*/capon_bf fields
                        # were removed 2026-05-05 (host CFAR pipeline
                        # is skipped, chip-side CFAR can't be retuned
                        # at runtime on this firmware).
                        if "aa" in _saved:
                            radar.update_aa_params(**{
                                k: _saved["aa"][k] for k in
                                ("pmm_band_low_hz","pmm_band_high_hz","pmm_threshold_db",
                                 "pmm_slow_time_win")
                                if k in _saved["aa"]
                            })
                        log.info("Restored saved per-mode radar config from %s", _saved_path)
                except Exception as e:
                    log.warning("Could not restore radar_modes.json: %s — using YAML defaults", e)
                log.info("Composite radar backend up (TLV + raw-ADC, mode=stock)")
            except Exception as e:
                log.warning("Composite/DCA wiring failed (%s) — falling back to RadarManager only; A/A unavailable", e)
        except KeyError as e:
            log.warning("Radar config missing key %s — disabling radar", e)
            radar = None
        except Exception as e:
            log.warning("Radar manager failed to start: %s — continuing without radar", e)
            radar = None

    # (Gimbal manager started earlier — moved ahead of radar so the
    # radar's mmW Studio bring-up doesn't gate manual control.)

    # Load persisted extrinsic calibration (if any) and apply it to the
    # live managers BEFORE the GUI starts pushing frames. This is what
    # makes the SAVE button on the EXTRINSIC CALIBRATION card useful:
    # values survive restarts. Missing file = use YAML defaults; corrupt
    # file = warning logged, fall back to YAML. Never blocks startup.
    try:
        from common import calibration_store
        applied = calibration_store.apply_to_managers(
            radar_manager=radar, fusion_manager=fusion,
        )
        if applied:
            log.info("Loaded persisted extrinsic calibration: %s", applied)
    except Exception as e:
        log.warning("Calibration load skipped: %s", e)

    # Construct the JSONL recorder. Always exists so the WS "record"
    # button is wired even when recording.enabled=false (in which case
    # start() is a no-op). Lifecycle is driven by gui.app's WS handler
    # OR --auto-record below.
    rec_cfg = (cfg.get("recording") or {})
    # demoDDM mode: humans/vehicles arrive on Topic.RADAR (TLV via
    # RadarManager); drone PMM hits arrive on Topic.RADAR_AA (raw-ADC
    # via DCAPipeline). Recorder writes BOTH as separate JSONL channels
    # ("radar/frame" + "radar/aa_frame") -- see jsonl_recorder._channels_table.
    recorder = JSONLRecorder(
        BUS,
        output_dir=str(rec_cfg.get("output_dir", "./recordings")),
        jpeg_quality=int(rec_cfg.get("jpeg_quality", 92)),
        channel_enable=dict(rec_cfg.get("channels") or {}),
    )

    # Build FastAPI app. The managers are passed in so the runtime
    # config endpoints can mutate detector parameters live from the GUI.
    app = create_app(
        thermal_manager=thermal,
        eo_manager=eo,
        gimbal_manager=gimbal,
        radar_manager=radar,
        fusion_manager=fusion,
        recorder=recorder,
        config_snapshot=cfg,
    )

    # Auto-start recording if either the YAML or the CLI says so. The
    # CLI flag is sticky — it overrides config=false. We DO respect
    # recording.enabled=false as a hard kill switch (logged + skipped).
    auto = bool(args.auto_record) or bool(rec_cfg.get("auto_start", False))
    if auto and bool(rec_cfg.get("enabled", True)):
        try:
            path = recorder.start(config_snapshot=cfg)
            log.info("--auto-record on: writing %s", path)
        except Exception as e:
            log.warning("--auto-record failed: %s", e)
    elif auto:
        log.info("--auto-record requested but recording.enabled=false; skipping")

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
        try:
            if recorder.is_recording:
                recorder.stop()
        except Exception:
            log.exception("recorder stop failed during shutdown")
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
        try:
            if recorder.is_recording:
                recorder.stop()
        except Exception:
            log.exception("recorder stop failed at shutdown")
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
