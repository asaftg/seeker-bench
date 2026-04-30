"""Phase-3 ``meta.yaml`` writer — the canonical "what was this?" file.

Spec: ``docs/PHASE_3_RECORDER_FIELDS.md`` §"meta.yaml — the canonical
'what was this?' file". Written once at recording start, alongside
the JSONL bus stream. Failure here is non-fatal — recording proceeds
without meta if anything raises (the GUI wraps the call in try/except
and logs).

Why YAML and not JSON: the spec's example uses YAML, comments aid the
post-flight reader, and pyyaml is already a dep
(``requirements.txt: pyyaml>=6.0,<7.0``). ``yaml.safe_dump`` keyword
ordering is preserved by ``sort_keys=False`` so the output reads
top-to-bottom in the same order as the spec.

Sourcing convention:

  - **Mandatory keys** (always populated): schema_version, recorded_at,
    label, mode, backend, profile_name, awr_cfg_path, awr_cfg_sha256,
    awr_cli_port, awr_cli_baud, dca.{control_endpoint,data_endpoint},
    frame_dims.*, pipeline.pmm_*, capture.*. Anything missing here
    means the manager is in an unexpected state — we still emit the
    file, with ``null`` placeholders, because half a meta is better
    than none for offline triage.

  - **Best-effort** (silently omitted on failure): awr_firmware.*,
    dca.fpga_version. These require a subprocess call into the TI
    CLI which we do NOT want to do on REC-press (popups, latency).
    Left unset by V1.0; can be filled in by post-flight tooling.

  - **Operator-fillable**: ``operator``, ``scene``. Empty placeholders
    so the analyst knows where to write.

The writer never raises — it logs and returns ``None`` on any error.
The caller treats meta failure as a soft warning.
"""
from __future__ import annotations

import hashlib
import logging
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

log = logging.getLogger(__name__)


_SCHEMA_VERSION = 1


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────
def _sha256_of_file(path: Path) -> Optional[str]:
    """Return hex sha256 of ``path``, or None on any I/O error."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _isoformat_local() -> str:
    """ISO-8601 with the local UTC offset (e.g. ``2026-04-29T14:33:21+03:00``).

    Matches the spec's example. Uses the system's local zone via the
    naive→aware conversion that Python provides since 3.6."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    """Like getattr, but never raises. Used because radar managers
    expose different attribute sets across stock / composite, and we
    want the meta file to fail soft rather than blow up REC."""
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _frame_dims_from_manager(rm: Any, cfg_path: Optional[Path]) -> Dict[str, Any]:
    """Best-effort frame-dim extraction.

    Order of preference:
      1. The DCA pipeline already parsed the cfg at start — read its
         FrameDims directly (most authoritative; it's what the
         live system is using).
      2. Re-parse the .cfg via ``radar_dca.dca_pipeline.dims_from_cfg_file``.
      3. Studio .mmwave.json next to the cfg (if present).
      4. Empty dict — analyst will have to derive from the cfg.
    """
    out: Dict[str, Any] = {}
    pipeline = _safe_get(rm, "_dca_pipeline", None)
    fd = _safe_get(pipeline, "_dims", None)
    if fd is None:
        # Try parsing the cfg directly. Import lazily so meta_writer
        # is importable in environments without numpy / radar deps.
        if cfg_path is not None and Path(cfg_path).exists():
            try:
                from radar_dca.dca_pipeline import dims_from_cfg_file
                fd = dims_from_cfg_file(str(cfg_path))
            except Exception as e:
                log.debug("meta: dims_from_cfg_file(%s) failed: %s", cfg_path, e)
                fd = None
    if fd is not None:
        out["n_chirps"] = int(_safe_get(fd, "n_chirps", 0)) or None
        out["n_rx"] = int(_safe_get(fd, "n_rx", 0)) or None
        out["n_samples"] = int(_safe_get(fd, "n_samples", 0)) or None
        out["bytes_per_sample"] = int(_safe_get(fd, "bytes_per_sample", 4))
        out["bytes_per_frame"] = int(_safe_get(fd, "bytes_per_frame", 0)) or None
        try:
            out["prf_hz"] = float(_safe_get(fd, "prf_hz", 0.0)) or None
        except Exception:
            out["prf_hz"] = None
        try:
            out["chirp_period_s"] = float(_safe_get(fd, "chirp_period_s", 0.0)) or None
        except Exception:
            out["chirp_period_s"] = None
        try:
            out["range_resolution_m"] = float(
                _safe_get(fd, "range_resolution_m", 0.0)) or None
        except Exception:
            out["range_resolution_m"] = None

    # max_range_m is a tuning slider on the radar manager, not part
    # of FrameDims — pull it separately. Falls back to None.
    try:
        mr = _safe_get(rm, "max_range_m", None)
        if mr is not None:
            out["max_range_m"] = float(mr)
    except Exception:
        pass
    return out


def _pipeline_block(rm: Any) -> Dict[str, Any]:
    """Pull the live PMM tuning off the DCA pipeline, if any."""
    out: Dict[str, Any] = {}
    pipeline = _safe_get(rm, "_dca_pipeline", None)
    if pipeline is None:
        return out
    out["pmm_band_low_hz"] = _safe_get(pipeline, "_pmm_band_low", None)
    out["pmm_band_high_hz"] = _safe_get(pipeline, "_pmm_band_high", None)
    out["pmm_threshold_db"] = _safe_get(pipeline, "_pmm_threshold", None)
    # Slow-time window is a separate knob; include it as it changes
    # the detector responsiveness materially.
    out["pmm_slow_time_win"] = _safe_get(pipeline, "_pmm_slow_time_win", None)
    return out


def _capture_block(config_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Network capture sanity — host_ip + (best-effort) hostname.

    No new deps for V1.0. ``psutil`` would let us read NIC name +
    MTU, but it isn't in requirements.txt; rather than add it just
    for the meta file, we skip those keys and the analyst can grep
    `ipconfig` if they need them.
    """
    out: Dict[str, Any] = {}
    radar_cfg = (config_snapshot or {}).get("radar", {}) or {}
    dca_cfg = radar_cfg.get("dca", {}) or {}
    host_ip = dca_cfg.get("host_ip")
    if host_ip:
        out["host_ip"] = str(host_ip)
    # Hostname is cheap and useful for cross-machine recordings.
    try:
        out["host_name"] = socket.gethostname()
    except Exception:
        pass
    # NIC + MTU intentionally omitted — would need psutil. Leave a
    # marker so the analyst knows it was deliberate, not forgotten.
    out["host_nic"] = None
    out["mtu"] = None
    # Socket recv buffer — DataPortListener requests 8 MB (see
    # data_port.py::start). Hard-code rather than introspect because
    # the actual granted size depends on OS limits.
    out["socket_recv_buffer_bytes"] = 8 * 1024 * 1024
    return out


# ──────────────────────────────────────────────────────────────────────
# public entry point
# ──────────────────────────────────────────────────────────────────────
def write_meta(
    recording_dir: Path,
    *,
    jsonl_path: Path,
    dca_bin_path: Optional[Path],
    radar_manager: Any,
    config_snapshot: Dict[str, Any],
) -> Optional[Path]:
    """Write ``<recording_dir>/<stem>.meta.yaml`` for the active session.

    Parameters
    ----------
    recording_dir
        Directory where the JSONL was just opened. We write the meta
        as a sibling of the JSONL.
    jsonl_path
        Path returned by ``JSONLRecorder.start()``. Its stem (filename
        without ``.jsonl``) becomes the meta filename stem so a
        post-rename ``<stem>.meta.yaml`` pairs cleanly with
        ``<stem>.jsonl``.
    dca_bin_path
        Sibling .bin path if DCA recording is also active; ``None``
        when radar isn't in ``aa`` mode. Recorded so post-flight
        scripts know whether to expect a binary stream.
    radar_manager
        The composite or stock RadarManager instance. We pull mode,
        cfg path, profile name, CLI port, etc. via getattr — every
        attribute is guarded with a default so a barebones manager
        still produces a valid (if sparse) meta file.
    config_snapshot
        ``common.config.load_config()`` dict already snapshotted by
        the GUI. Used for paths the manager doesn't carry on itself
        (DCA endpoints, host IP).

    Returns
    -------
    Path of the written file on success, ``None`` on any failure.
    Never raises — failure is logged.
    """
    try:
        recording_dir = Path(recording_dir)
        jsonl_path = Path(jsonl_path)
        recording_dir.mkdir(parents=True, exist_ok=True)
        stem = jsonl_path.stem
        meta_path = recording_dir / f"{stem}.meta.yaml"

        rm = radar_manager
        cfg_snap = config_snapshot or {}
        radar_cfg = cfg_snap.get("radar", {}) or {}
        dca_cfg = radar_cfg.get("dca", {}) or {}

        # ── identifiers / timestamps ────────────────────────────────
        recorded_at = _isoformat_local()
        # Label defaults to the stem (a timestamped ID); operator can
        # rename via the GUI's rename_to which also renames this file.
        label = stem

        mode = str(_safe_get(rm, "mode", "stock") or "stock").lower()
        # Backend is a string label so the analyst can grep without
        # importing Python classes. We infer from the manager type.
        backend = type(rm).__name__ if rm is not None else "unknown"
        profile_name = _safe_get(rm, "profile_name", None)

        # ── awr cfg / chip ──────────────────────────────────────────
        cfg_path_val = _safe_get(rm, "cfg_path", None) or radar_cfg.get("cfg_path")
        cfg_path = Path(cfg_path_val) if cfg_path_val else None
        awr_cfg_sha256 = _sha256_of_file(cfg_path) if cfg_path else None

        cli_port = (
            _safe_get(rm, "cli_port", None)
            or radar_cfg.get("cli_port")
        )
        cli_baud = (
            _safe_get(rm, "cli_baud", None)
            or radar_cfg.get("cli_baud", 115200)
        )

        # awr_firmware.*: best-effort, NOT obtainable without an extra
        # CLI subprocess call we don't want at REC-start. Skipped per
        # spec ("skip silently if not obtainable").
        awr_firmware: Optional[Dict[str, Any]] = None

        # ── DCA endpoints ───────────────────────────────────────────
        host_ip = dca_cfg.get("host_ip", "192.168.33.30")
        dca_ip = dca_cfg.get("dca_ip", "192.168.33.180")
        config_port = dca_cfg.get("config_port", 4096)
        data_udp_port = dca_cfg.get("data_udp_port", 4098)
        dca_block: Dict[str, Any] = {
            # Spec format: ``ip:port``. control_endpoint is the DCA's
            # CLI port (we send to it); data_endpoint is the host's
            # UDP socket (DCA sends to us).
            "control_endpoint": f"{dca_ip}:{config_port}",
            "data_endpoint": f"{host_ip}:{data_udp_port}",
        }
        # fpga_version — best-effort, skipped silently. Reading it
        # would require a DCA1000EVM_CLI_Control subprocess at the
        # moment of REC-press, which can pop a console window and
        # block for ~1 s. Not worth it for V1.0.

        # ── frame dims + pipeline ────────────────────────────────────
        frame_dims = _frame_dims_from_manager(rm, cfg_path)
        pipeline = _pipeline_block(rm)
        capture = _capture_block(cfg_snap)

        # ── assemble the meta dict (order matches the spec doc) ─────
        meta: Dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "recorded_at": recorded_at,
            "operator": "",        # operator-fillable placeholder
            "label": label,
            "mode": mode,
            "backend": backend,
            "profile_name": profile_name,
            "awr_cfg_path": str(cfg_path) if cfg_path else None,
            "awr_cfg_sha256": awr_cfg_sha256,
            "awr_cli_port": cli_port,
            "awr_cli_baud": int(cli_baud) if cli_baud else None,
            "dca": dca_block,
            "frame_dims": frame_dims,
            "pipeline": pipeline,
            "capture": capture,
            # Operator-fillable scene block. Empty placeholders so
            # the analyst knows the slots exist.
            "scene": {
                "background": "",
                "weather": "",
                "targets": [],
            },
            # Pointer to companion artifacts, useful for replay tools
            # that get only the meta path.
            "artifacts": {
                "jsonl": jsonl_path.name,
                "dca_bin": Path(dca_bin_path).name if dca_bin_path else None,
                "dca_index_csv": (
                    Path(dca_bin_path).with_suffix(".csv").name
                    if dca_bin_path else None
                ),
            },
        }
        # Only include awr_firmware if we actually have something.
        if awr_firmware:
            meta["awr_firmware"] = awr_firmware

        # ── write ────────────────────────────────────────────────────
        # safe_dump: no Python tags, no anchors; sort_keys=False so the
        # file reads top-to-bottom in the same order as our dict.
        with open(meta_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(
                meta, fh,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )
        log.info("meta.yaml written → %s", meta_path)
        return meta_path
    except Exception:
        log.exception("write_meta failed (non-fatal)")
        return None


def rename_meta_alongside(jsonl_old_path: Path, jsonl_new_path: Path) -> Optional[Path]:
    """Rename the sibling ``<old_stem>.meta.yaml`` to match a renamed
    JSONL, with the same dedup-on-collision logic the JSONL rename
    uses. No-op if the source meta file isn't there.

    Returns the new meta path on success, ``None`` if there was
    nothing to rename or the rename failed (logged, not raised).
    """
    try:
        old = Path(jsonl_old_path)
        new = Path(jsonl_new_path)
        old_meta = old.with_name(old.stem + ".meta.yaml")
        if not old_meta.exists():
            return None
        new_meta = new.with_name(new.stem + ".meta.yaml")
        # Dedup: append _1, _2, ... until we find a free slot. Mirrors
        # gui/app.py's JSONL rename.
        if new_meta.exists():
            base = new_meta.with_suffix("")  # strips .yaml
            # Strip the .meta as well so the suffix stack is .meta.yaml
            base_stem = base.stem if base.stem.endswith(".meta") else base.stem
            parent = new_meta.parent
            n = 1
            while True:
                candidate = parent / f"{base_stem}_{n}.meta.yaml"
                if not candidate.exists():
                    new_meta = candidate
                    break
                n += 1
        old_meta.rename(new_meta)
        log.info("meta.yaml renamed → %s", new_meta)
        return new_meta
    except Exception:
        log.exception("rename_meta_alongside failed (non-fatal)")
        return None
