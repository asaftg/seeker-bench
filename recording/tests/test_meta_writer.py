"""Unit tests for ``recording.meta_writer``.

We test the writer in isolation — no real RadarManager, no real DCA.
A small fake stands in for the manager so we can verify that:

  - mandatory keys are present
  - getattr-guarded fields fall back rather than raising
  - the cfg sha256 matches the actual file content
  - the sibling rename helper preserves the file and dedups on
    collision.

The writer is documented to NEVER raise — these tests pin that
contract by feeding it deliberately broken inputs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from recording.meta_writer import (
    rename_meta_alongside,
    write_meta,
)


# ──────────────────────────────────────────────────────────────────────
# fixtures
# ──────────────────────────────────────────────────────────────────────
def _fake_pipeline(**kwargs):
    """Stand-in for DCAPipeline carrying just the attributes the
    meta writer reads off it."""
    defaults = dict(
        _pmm_band_low=50.0,
        _pmm_band_high=500.0,
        _pmm_threshold=6.0,
        _pmm_slow_time_win=256,
        _dims=SimpleNamespace(
            n_chirps=2304, n_rx=4, n_samples=384,
            bytes_per_sample=4, bytes_per_frame=2304 * 4 * 384 * 4,
            prf_hz=10000.0, chirp_period_s=100e-6,
            range_resolution_m=0.04,
        ),
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _fake_radar_manager(*, cfg_path: Path, with_pipeline: bool = True):
    pipeline = _fake_pipeline() if with_pipeline else None
    return SimpleNamespace(
        mode="aa",
        profile_name="awr2944p_aa",
        cfg_path=str(cfg_path),
        cli_port="COM10",
        cli_baud=115200,
        max_range_m=250.0,
        _dca_pipeline=pipeline,
        _dca_listener=object(),
    )


def _make_cfg(tmp_path: Path) -> Path:
    """Drop a tiny placeholder cfg file so sha256 is deterministic."""
    cfg = tmp_path / "fake.cfg"
    cfg.write_bytes(b"% smoke cfg\nsensorStop\n")
    return cfg


def _cfg_snapshot() -> dict:
    return {
        "radar": {
            "cli_port": "COM10",
            "cli_baud": 115200,
            "cfg_path": "radar/cfg/fake.cfg",
            "dca": {
                "host_ip": "192.168.33.30",
                "dca_ip": "192.168.33.180",
                "config_port": 4096,
                "data_udp_port": 4098,
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────
# happy-path coverage
# ──────────────────────────────────────────────────────────────────────
def test_write_meta_produces_expected_keys(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    rm = _fake_radar_manager(cfg_path=cfg)
    jsonl = tmp_path / "seeker_2026-04-29_12-00-00.jsonl"
    jsonl.write_text("{}\n")  # touch so the writer sees a real file
    bin_path = tmp_path / "seeker_2026-04-29_12-00-00_radar.bin"

    out = write_meta(
        recording_dir=tmp_path,
        jsonl_path=jsonl,
        dca_bin_path=bin_path,
        radar_manager=rm,
        config_snapshot=_cfg_snapshot(),
    )
    assert out is not None and out.exists()

    data = yaml.safe_load(out.read_text(encoding="utf-8"))

    # Mandatory top-level keys per spec.
    for k in ("schema_version", "recorded_at", "label", "mode", "backend",
              "profile_name", "awr_cfg_path", "awr_cfg_sha256",
              "awr_cli_port", "awr_cli_baud", "dca", "frame_dims",
              "pipeline", "capture", "scene"):
        assert k in data, f"missing key {k!r}"

    assert data["schema_version"] == 1
    assert data["mode"] == "aa"
    assert data["profile_name"] == "awr2944p_aa"
    assert data["awr_cli_port"] == "COM10"
    assert data["awr_cli_baud"] == 115200

    # DCA endpoints in ip:port form.
    assert data["dca"]["control_endpoint"] == "192.168.33.180:4096"
    assert data["dca"]["data_endpoint"] == "192.168.33.30:4098"

    # Frame dims propagated from the fake pipeline.
    fd = data["frame_dims"]
    assert fd["n_chirps"] == 2304
    assert fd["n_rx"] == 4
    assert fd["n_samples"] == 384
    assert fd["bytes_per_sample"] == 4
    assert fd["max_range_m"] == 250.0

    # Pipeline PMM tuning round-tripped.
    assert data["pipeline"]["pmm_band_low_hz"] == 50.0
    assert data["pipeline"]["pmm_band_high_hz"] == 500.0
    assert data["pipeline"]["pmm_threshold_db"] == 6.0
    assert data["pipeline"]["pmm_slow_time_win"] == 256

    # sha256 matches the on-disk cfg.
    expected = hashlib.sha256(cfg.read_bytes()).hexdigest()
    assert data["awr_cfg_sha256"] == expected

    # Operator-fillable placeholders are present and empty.
    assert data["operator"] == ""
    assert data["scene"]["targets"] == []
    assert data["scene"]["background"] == ""

    # Companion artifacts pointer.
    assert data["artifacts"]["jsonl"] == jsonl.name
    assert data["artifacts"]["dca_bin"] == bin_path.name
    assert data["artifacts"]["dca_index_csv"] == bin_path.with_suffix(".csv").name


def test_write_meta_no_dca_listener(tmp_path: Path):
    """Stock-mode recording: no DCA pipeline, no .bin path. Meta should
    still write, with null artifacts where applicable."""
    cfg = _make_cfg(tmp_path)
    rm = _fake_radar_manager(cfg_path=cfg, with_pipeline=False)
    rm.mode = "stock"
    jsonl = tmp_path / "stock_run.jsonl"
    jsonl.write_text("")

    out = write_meta(
        recording_dir=tmp_path,
        jsonl_path=jsonl,
        dca_bin_path=None,
        radar_manager=rm,
        config_snapshot=_cfg_snapshot(),
    )
    assert out is not None and out.exists()
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["mode"] == "stock"
    # Pipeline block can be empty when no DCA pipeline exists.
    assert isinstance(data["pipeline"], dict)
    assert data["artifacts"]["dca_bin"] is None
    assert data["artifacts"]["dca_index_csv"] is None


def test_write_meta_never_raises_on_broken_inputs(tmp_path: Path):
    """The writer is documented to swallow all exceptions and return
    None — feed it junk to confirm."""
    out = write_meta(
        recording_dir=tmp_path,
        jsonl_path=tmp_path / "x.jsonl",
        dca_bin_path=None,
        radar_manager=None,        # no manager at all
        config_snapshot=None,      # type: ignore[arg-type]
    )
    # Even with no manager, we expect a written file (sparse but valid).
    assert out is None or out.exists()


def test_write_meta_handles_missing_cfg_file(tmp_path: Path):
    """If the cfg path doesn't exist, sha256 is None and dims fall
    back gracefully — must NOT raise."""
    rm = SimpleNamespace(
        mode="ag",
        profile_name="awr2944p_ag",
        cfg_path=str(tmp_path / "does_not_exist.cfg"),
        cli_port="COM10",
        cli_baud=115200,
    )
    jsonl = tmp_path / "ag_run.jsonl"
    jsonl.write_text("")

    out = write_meta(
        recording_dir=tmp_path,
        jsonl_path=jsonl,
        dca_bin_path=None,
        radar_manager=rm,
        config_snapshot=_cfg_snapshot(),
    )
    assert out is not None
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["awr_cfg_sha256"] is None


# ──────────────────────────────────────────────────────────────────────
# rename helper
# ──────────────────────────────────────────────────────────────────────
def test_rename_meta_alongside_basic(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    rm = _fake_radar_manager(cfg_path=cfg)
    jsonl = tmp_path / "before.jsonl"
    jsonl.write_text("")
    write_meta(
        recording_dir=tmp_path, jsonl_path=jsonl,
        dca_bin_path=None, radar_manager=rm,
        config_snapshot=_cfg_snapshot(),
    )
    new_jsonl = tmp_path / "after.jsonl"
    new_meta = rename_meta_alongside(jsonl, new_jsonl)
    assert new_meta is not None
    assert new_meta.name == "after.meta.yaml"
    assert new_meta.exists()
    assert not (tmp_path / "before.meta.yaml").exists()


def test_rename_meta_alongside_dedups(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    rm = _fake_radar_manager(cfg_path=cfg)
    jsonl = tmp_path / "src.jsonl"
    jsonl.write_text("")
    write_meta(
        recording_dir=tmp_path, jsonl_path=jsonl,
        dca_bin_path=None, radar_manager=rm,
        config_snapshot=_cfg_snapshot(),
    )
    # Pre-existing collision — the rename should pick a numeric suffix.
    (tmp_path / "dst.meta.yaml").write_text("decoy")
    new_meta = rename_meta_alongside(jsonl, tmp_path / "dst.jsonl")
    assert new_meta is not None
    assert new_meta.name == "dst_1.meta.yaml"
    assert new_meta.exists()
    # The decoy must still be there.
    assert (tmp_path / "dst.meta.yaml").read_text() == "decoy"


def test_rename_meta_alongside_missing_source(tmp_path: Path):
    """No source meta → return None, do nothing."""
    out = rename_meta_alongside(
        tmp_path / "nope.jsonl", tmp_path / "still_nope.jsonl",
    )
    assert out is None
