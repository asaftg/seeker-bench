"""
Best-effort autonomous download of publicly-hosted thermal datasets.

Direct-URL datasets attempted:
    - CVC-14 thermal pedestrian  (UAB direct link)
    - Anti-UAV (DUT)             (GitHub releases mirror, if available)

Datasets NOT attempted (they sit behind forms/cloud-drives):
    - FLIR ADAS v2               (user downloaded manually → Downloads/)
    - M3FD                        (OneDrive/Baidu only)
    - KAIST Multispectral         (Google Form gate)
    - Anti-UAV410 / 600           (Baidu/OneDrive)

Failures are non-fatal: the orchestrator still runs training with whatever
arrived. Re-run this any time to try again.

Usage:
    python scripts/download_public_datasets.py
"""
from __future__ import annotations
import urllib.request
import ssl
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"

# (name, target_dir, url, expected_min_bytes)
TARGETS = [
    # CVC-14 — thermal pedestrian from UAB. FIR (far-IR) split only.
    ("cvc14_day",
     DATASETS / "cvc14" / "CVC-14-Day.zip",
     "http://adas.cvc.uab.es/elektra/wp-content/uploads/CVC-14-Day.zip",
     50_000_000),
    ("cvc14_night",
     DATASETS / "cvc14" / "CVC-14-Night.zip",
     "http://adas.cvc.uab.es/elektra/wp-content/uploads/CVC-14-Night.zip",
     50_000_000),
    # Anti-UAV — try the GitHub release mirror. Fails cleanly if not present.
    ("antiuav_github",
     DATASETS / "antiuav" / "antiuav_release.zip",
     "https://github.com/ZhaoJ9014/Anti-UAV/releases/download/v1.0/Anti-UAV-train.zip",
     10_000_000),
]


def try_download(name: str, dst: Path, url: str, min_bytes: int) -> bool:
    if dst.exists() and dst.stat().st_size >= min_bytes:
        print(f"[{name}] already present: {dst} ({dst.stat().st_size/1e6:.0f} MB)")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] trying {url}")
    try:
        # Some academic servers have broken SSL chains — fall back to unverified.
        ctx = ssl.create_default_context()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            urllib.request.urlretrieve(url, dst)
        except Exception:
            ctx = ssl._create_unverified_context()
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, dst)
        size = dst.stat().st_size
        if size < min_bytes:
            print(f"[{name}] download too small ({size} bytes), probably a 404 page — discarding")
            dst.unlink()
            return False
        print(f"[{name}] OK: {size/1e6:.0f} MB -> {dst}")
        return True
    except Exception as e:
        print(f"[{name}] FAILED: {e}")
        if dst.exists():
            try:
                dst.unlink()
            except Exception:
                pass
        return False


def main():
    results = {}
    for name, dst, url, min_bytes in TARGETS:
        results[name] = try_download(name, dst, url, min_bytes)
    print()
    print("[download] summary:")
    for name, ok in results.items():
        print(f"  {name:20s} {'OK' if ok else 'FAIL'}")
    return any(results.values())


if __name__ == "__main__":
    main()
