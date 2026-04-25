"""LI-IMX568-GMSL2 USB3 capture driver.

Native 2472x2064 global-shutter mono sensor via Leopard Imaging's
FX3-based EVA bridge. The bridge advertises a YUY2-labeled UVC stream
on DirectShow; Windows auto-decodes the payload to BGR uint8, which is
what OpenCV hands us. We treat the camera as mono and use the Y
channel (via cvtColor) for AGC / detection input.

API mirrors ``eo.webcam_capture.WebcamCapture`` so ``EOManager`` can
consume either source interchangeably:

    start() / stop() / grab() -> BGR uint8 (H, W, 3)

Extensions beyond WebcamCapture:
  * set_exposure_ms(x) / set_gain(x) — runtime profile control
  * get_scene_mean() — cheap luminance probe the profile-selector uses

Standalone::

    python -m eo.imx568_capture             # auto-detect index, 5s probe
    python -m eo.imx568_capture --device 0  # force index
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from common.logging_setup import get_logger

log = get_logger(__name__)


# Native resolution confirmed via eo_imx568_probe.py — anything lower
# and the FX3 bridge falls back to a cropped/binned mode we don't want.
NATIVE_W = 2472
NATIVE_H = 2064

# Pixel formats to try, in order of preference. YUY2 first because
# ``scripts/eo_list_formats.py`` showed it's the only carrier the FX3
# bridge actually delivers real pixels on — every other FOURCC we request
# gets silently remapped to YUY2 by DirectShow anyway, but asking for MJPG
# explicitly has been observed to hand back one good frame and then a
# stream of black frames (seen 2026-04-24: "scene mean=147.9" on frame 1,
# "scene mean=0.0" four seconds later). ``None`` = "don't force a
# format, let DirectShow negotiate whatever the driver's default is" —
# last-ditch fallback for a future bridge firmware.
# Keep this list in sync with scripts/eo_list_formats.py.
_FOURCC_CANDIDATES: list[Optional[str]] = ["YUY2", None]

# Capture backends to try, in order. DSHOW is the only backend that
# actually opens the FX3 bridge on this laptop — MSMF returns
# isOpened()==False every time (tested 2026-04-24 across five device
# indices), and CAP_ANY just delegates back to DSHOW anyway. Keeping
# DSHOW alone avoids a per-reconnect 100 ms × N-indices stall we'd
# otherwise eat probing dead backends every few seconds.
_BACKEND_CANDIDATES: list[tuple[int, str]] = [
    (cv2.CAP_DSHOW, "DSHOW"),
]


def _fourcc_to_str(fourcc_int: int) -> str:
    """Decode a 4-byte FOURCC integer (as returned by CAP_PROP_FOURCC)."""
    try:
        return "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
    except Exception:
        return "?"


def _clamp_exposure_for_g_saturation(cap: "cv2.VideoCapture") -> None:
    """No-op stub. The 2026-04-24 attempt to manually clamp exposure by
    flipping to AE=manual and walking CAP_PROP_EXPOSURE / CAP_PROP_GAIN
    made the live image WORSE — every test value drove G further into
    saturation, ending in "exposure clamp ran out of room" + a fully
    blown-out white frame on the GUI.

    Either this bridge interprets CAP_PROP_EXPOSURE in reverse vs. the
    UVC log₂-seconds convention, or it ignores the writes entirely while
    the gain write (set first) bumped to MAX gain on a 0..10 scale.
    Without hardware introspection I can't tell which. Until I have a
    safe way to probe direction (e.g. test +/- one stop and only commit
    if mean brightness moved the expected direction), do nothing here
    so the bridge's internal AE stays in charge and the image is at
    least as good as before this function existed.

    Kept as a named no-op so the call site in start() and any future
    re-introduction path is obvious.
    """
    return


def _frame_has_spatial_content(frame: np.ndarray) -> bool:
    """True iff the frame carries real scene data (not a flat fake buffer).

    The FX3-over-YUY2 failure mode produces a frame where each of the B,
    G, R channels is a single constant value, so the *per-channel* std is
    zero even though the whole-frame std can look non-zero (because the
    three channels are different constants). We test per-channel std on a
    downsampled slice: ~1 ms, unambiguous.
    """
    if frame is None or frame.size == 0:
        return False
    if frame.ndim == 2:
        return float(frame[::8, ::8].std()) > 1.0
    slab = frame[::8, ::8]
    return max(float(slab[:, :, c].std()) for c in range(slab.shape[2])) > 1.0


class IMX568Capture:
    """LI-IMX568-GMSL2 USB3 wrapper, BGR uint8 output.

    The sensor is *mono*. BGR comes out of DirectShow because its YUY2
    decode path always produces 3-channel output. All three channels
    carry the same luma; we keep them as BGR so downstream YOLO
    (trained on color imagery) doesn't have to special-case.
    """

    def __init__(
        self,
        device_index: int | str = "auto",
        exclude_indices: Optional[list[int]] = None,
    ) -> None:
        self.requested_index = device_index
        self.exclude_indices = list(exclude_indices or [])

        self.device_index: Optional[int] = None
        self.actual_width: int = 0
        self.actual_height: int = 0
        # API-compat shim for EOManager's _CaptureLike protocol — IMX568
        # does not expose a 16-bit raw path through UVC, so raw16 is
        # unavailable here. Classifier/detector read the BGR field.
        self.raw16_available: bool = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._fourcc: str = "?"  # which pixel format won negotiation
        # True iff CAP_PROP_CONVERT_RGB=0 took effect AND the resulting
        # buffer is the YUY2 byte stream we expect (HxWx2 or HxW*2).
        # When True, grab() slices the raw bytes for a clean mono Y plane;
        # when False, grab() falls back to BGR-with-broken-color-decode and
        # eo_processor's _to_luma has to recover from the green tint.
        self._raw_yuy2_mode: bool = False

    # ───────────────────────── lifecycle ─────────────────────────

    def start(self) -> None:
        """Open the IMX568 at native resolution and commit to the first
        combo that delivers a native-res frame — flat or not.

        Why NOT a full matrix search for "spatial content":
          * Windows UVC is exclusive-open. Once we hold a cap on idx 0,
            every subsequent `cv2.VideoCapture(0, …)` on MSMF or DSHOW
            returns `isOpened()==False`. The search looks like
            "nothing else works" but really it's self-blocked.
          * Repeated open/close cycles against the FX3 bridge push it
            into a bad state (seen 2026-04-24: after a 10s probe, the
            very same cap that produced a flat-but-valid test frame
            returned None from its first real grab).
          * If the first combo hands back a flat buffer, the sensor is
            stuck at the USB layer and NO OpenCV combo will fix it —
            the fix is a USB replug, not another backend try.

        Strategy: walk (backend × FOURCC × index), pick the first combo
        that opens at NATIVE resolution, log a warning if the test frame
        is flat, and return. The user either sees real content or sees
        the flat-frame warning plus "replug the cable". No more silent
        black panel, no more 10-second stall.
        """
        if self._cap is not None:
            return

        candidates = self._candidate_indices()
        last_err: Optional[str] = None

        for idx in candidates:
            if idx in self.exclude_indices:
                continue

            wrong_device_on_index = False

            for backend_id, backend_name in _BACKEND_CANDIDATES:
                if wrong_device_on_index:
                    break
                for fourcc_tag in _FOURCC_CANDIDATES:
                    label = f"idx {idx} {backend_name}/{fourcc_tag or 'DEFAULT'}"

                    try:
                        cap = cv2.VideoCapture(idx, backend_id)
                    except Exception as e:
                        last_err = f"{label}: ctor threw {e!r}"
                        log.info("%s", last_err)
                        continue
                    if not cap.isOpened():
                        cap.release()
                        last_err = f"{label}: isOpened()==False"
                        log.info("%s", last_err)
                        continue

                    try:
                        if fourcc_tag is not None:
                            cap.set(cv2.CAP_PROP_FOURCC,
                                    cv2.VideoWriter_fourcc(*fourcc_tag))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, NATIVE_W)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, NATIVE_H)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        # Force AE ON. Without this, a previous run that
                        # set MANUAL mode (CAP_PROP_AUTO_EXPOSURE=0.25)
                        # can leave the bridge stuck at a fixed exposure
                        # across reboots — image looks dark or blown
                        # depending on what value it last latched. The
                        # 0.75 = AUTO convention is DSHOW-specific.
                        # Bridge may ignore the write; harmless either way.
                        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                    except Exception as e:
                        cap.release()
                        last_err = f"{label}: set-props threw {e!r}"
                        continue

                    ok, test = cap.read()
                    if not ok or test is None:
                        cap.release()
                        last_err = f"{label}: opened but no frames"
                        continue

                    h, w = test.shape[:2]
                    if (w, h) != (NATIVE_W, NATIVE_H):
                        cap.release()
                        last_err = (f"{label}: returned {w}x{h}, not "
                                    f"native {NATIVE_W}x{NATIVE_H}")
                        log.info("%s — skipping, not IMX568", last_err)
                        # Not IMX568 on this index — don't bother
                        # trying other backends / formats here, move on.
                        wrong_device_on_index = True
                        break

                    negotiated = _fourcc_to_str(
                        int(cap.get(cv2.CAP_PROP_FOURCC))
                    )
                    has_content = _frame_has_spatial_content(test)

                    # ── Saturation clamp ─────────────────────────────
                    #
                    # The bridge's internal AE regularly drives Y so
                    # high that DirectShow's YUY2→BGR decode saturates
                    # the G channel across most of the frame (G pegged
                    # at 255). Everything past that point is lost to
                    # the [120,179] dead zone — no amount of software
                    # recovery brings it back.
                    #
                    # Fix: flip to manual exposure, walk the exposure
                    # value down until G's 95th percentile drops below
                    # 245 on a 16× subsample. We start at UVC log₂-s
                    # = -6 (≈ 15 ms, a typical indoor target) and drop
                    # 1 stop at a time; cap the search at 6 steps so a
                    # bridge that ignores the writes doesn't stall us.
                    # If the bridge rejects manual mode entirely, log
                    # and fall through — recovery still works on the
                    # saturated frame, it just looks noisier.
                    _clamp_exposure_for_g_saturation(cap)

                    # ── Raw YUY2 attempt ───────────────────────────────
                    #
                    # The FX3 bridge ships a MONO sensor through YUY2 with
                    # U=V=0. DirectShow's auto-decode of that into BGR
                    # produces a frame where:
                    #     B = Y - 227 (clipped to 0)
                    #     G = Y + 135 (clipped to 255 above Y=120)
                    #     R = Y - 179 (clipped to 0 below Y=179)
                    # which permanently destroys ~60 codes of dynamic range
                    # in the Y∈[120,179] band — the visible "oil-painting
                    # mid-tone patches" everyone has been blaming on the
                    # enhancement chain.
                    #
                    # CAP_PROP_CONVERT_RGB=0 disables that decode and hands
                    # us the raw YUY2 byte stream. Slicing every-other
                    # byte gives us the FULL 0..255 mono Y plane with no
                    # dead zone. This is the only configuration that
                    # produces an image that looks like a real day sensor.
                    #
                    # If the bridge or this opencv build refuses raw mode,
                    # we leave it on the BGR path and let eo_processor
                    # recover what it can from the broken color decode.
                    raw_ok = False
                    try:
                        if cap.set(cv2.CAP_PROP_CONVERT_RGB, 0):
                            ok2, raw_test = cap.read()
                            if ok2 and raw_test is not None:
                                rh, rw = raw_test.shape[:2]
                                # YUY2 is 2 bytes per pixel. Acceptable
                                # shapes: (H, W, 2) packed-channel, or
                                # (H, 2W) flat-byte. Anything else means
                                # the bridge ignored the flag.
                                if (raw_test.ndim == 3 and rw == NATIVE_W
                                        and rh == NATIVE_H
                                        and raw_test.shape[2] == 2):
                                    raw_ok = True
                                elif (raw_test.ndim == 2
                                      and rh == NATIVE_H
                                      and rw == 2 * NATIVE_W):
                                    raw_ok = True
                                else:
                                    # Bridge accepted the flag but format
                                    # is something we don't expect — turn
                                    # the auto-decode back on.
                                    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                            else:
                                cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                    except Exception as e:
                        log.debug("raw YUY2 attempt threw %r — staying BGR", e)
                        try:
                            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                        except Exception:
                            pass

                    # COMMIT — regardless of flatness. See docstring.
                    self._cap = cap
                    self.device_index = idx
                    self.actual_width = w
                    self.actual_height = h
                    self._fourcc = negotiated
                    self._raw_yuy2_mode = raw_ok
                    log.info(
                        "IMX568Capture mode: %s (raw YUY2 %s)",
                        "RAW_YUY2_MONO" if raw_ok else "BGR_FALLBACK",
                        "enabled — clean mono Y plane"
                            if raw_ok else
                            "rejected — falling back to broken-color BGR + "
                            "_to_luma green-tint recovery",
                    )
                    if has_content:
                        log.info(
                            "IMX568Capture opened on %s at %dx%d "
                            "(requested=%s, negotiated=%s) — "
                            "frame has real content",
                            label, w, h,
                            fourcc_tag or "DEFAULT", negotiated,
                        )
                    else:
                        log.warning(
                            "IMX568Capture opened on %s at %dx%d "
                            "(requested=%s, negotiated=%s) but test "
                            "frame is FLAT (per-ch std ~0). Sensor is "
                            "streaming a degenerate buffer at the USB "
                            "layer — unplug/replug the USB3 cable to "
                            "reset the FX3 bridge. Proceeding anyway so "
                            "the pipeline stays alive.",
                            label, w, h,
                            fourcc_tag or "DEFAULT", negotiated,
                        )
                    return

        raise RuntimeError(
            f"Could not open IMX568 (tried {candidates}, "
            f"excluded {self.exclude_indices}). Last error: {last_err}. "
            f"Make sure CameraTool / other viewers are closed."
        )

    def stop(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def __enter__(self) -> "IMX568Capture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ───────────────────────── capture ───────────────────────────

    def grab(self) -> Optional[np.ndarray]:
        """Return the next BGR uint8 frame (H, W, 3), or None on failure.

        In RAW_YUY2_MONO mode (preferred): pulls the raw YUY2 byte stream,
        slices the Y plane (every other byte), and re-expands to BGR for
        downstream consumers. Skips DirectShow's broken color decode
        entirely so the full 0..255 sensor range survives to the GUI.

        In BGR_FALLBACK mode: returns whatever DirectShow's YUY2->BGR
        gave us, green tint and all. eo_processor._to_luma is responsible
        for recovering luma in that case.
        """
        if self._cap is None:
            return None
        try:
            ok, frame = self._cap.read()
        except Exception:
            return None
        if not ok or frame is None:
            return None

        if not self._raw_yuy2_mode:
            return frame

        # Raw YUY2 → mono Y plane → BGR (replicated luma).
        # YUY2 byte order is [Y0 U Y1 V] per 2 pixels: bytes 0,2,4,... = Y.
        if frame.ndim == 3 and frame.shape[2] == 2:
            # OpenCV packed (H, W, 2): channel 0 holds Y for both pixels.
            y = frame[:, :, 0]
        elif frame.ndim == 2:
            # Flat byte stream (H, 2W): take every other byte.
            y = frame[:, 0::2]
        else:
            # Shouldn't happen if start()'s shape check passed, but be safe.
            return frame
        # Ensure contiguous uint8 — downstream uses np operations that
        # don't tolerate strided views silently.
        y = np.ascontiguousarray(y, dtype=np.uint8)
        return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ───────────────────────── controls ──────────────────────────
    #
    # UVC exposure is in log-2 seconds: CAP_PROP_EXPOSURE=-6 means
    # 2^-6 = 1/64 s ≈ 15.6 ms. Some bridges respect it, some don't —
    # we set it and don't error if the write is ignored (the scene's
    # measured brightness tells the profile selector whether it took).

    def set_exposure_ms(self, exposure_ms: float) -> bool:
        """Request a manual exposure in milliseconds. Returns True on success."""
        if self._cap is None or exposure_ms <= 0:
            return False
        try:
            # Flip to manual exposure mode (0.25 = manual on DSHOW).
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            # UVC: exposure in log-2 seconds.
            log2_s = np.log2(exposure_ms / 1000.0)
            return bool(self._cap.set(cv2.CAP_PROP_EXPOSURE, float(log2_s)))
        except Exception as e:
            log.debug("set_exposure_ms(%.2f) threw %r", exposure_ms, e)
            return False

    def set_auto_exposure(self, enabled: bool) -> bool:
        if self._cap is None:
            return False
        try:
            # 0.75 = auto, 0.25 = manual (DSHOW convention).
            return bool(self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,
                                      0.75 if enabled else 0.25))
        except Exception:
            return False

    def set_gain(self, gain: float) -> bool:
        """Request analog gain. Range depends on bridge; typically 0..100."""
        if self._cap is None or gain < 0:
            return False
        try:
            return bool(self._cap.set(cv2.CAP_PROP_GAIN, float(gain)))
        except Exception:
            return False

    # ───────────────────────── helpers ───────────────────────────

    def _candidate_indices(self) -> list[int]:
        if isinstance(self.requested_index, int):
            return [self.requested_index]
        if self.requested_index == "auto" or self.requested_index is None:
            # IMX568 showed up on index 0 on this rig after the USB-port
            # swap, but don't hardcode — probe 0..4 for other hosts.
            return [0, 1, 2, 3, 4]
        try:
            return [int(self.requested_index)]
        except (TypeError, ValueError):
            return [0, 1, 2, 3, 4]


# ───────────────────────────────────────────────────────────────
# Self-test
# ───────────────────────────────────────────────────────────────
def probe(duration_s: float = 5.0,
          device_index: int | str = "auto") -> Tuple[int, float, float]:
    """Open, stream for N seconds, return (frames, fps, mean_brightness)."""
    cap = IMX568Capture(device_index=device_index)
    cap.start()
    frames = 0
    mean_acc = 0.0
    t0 = time.time()
    try:
        while time.time() - t0 < duration_s:
            f = cap.grab()
            if f is not None:
                frames += 1
                # Sample the mean on every 5th frame — cheap enough to
                # log without flooding stdout for a 5s probe.
                if frames % 5 == 0:
                    mean_acc += float(f.mean())
    finally:
        cap.stop()
    elapsed = time.time() - t0
    fps = frames / elapsed if elapsed > 0 else 0.0
    mean = mean_acc / max(1, frames // 5)
    return frames, fps, mean


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--duration", type=float, default=5.0)
    args = ap.parse_args()

    f, fps, mean = probe(args.duration, device_index=args.device)
    print(f"IMX568 probe: {f} frames in {args.duration:.1f}s = {fps:.1f} fps, "
          f"scene mean brightness {mean:.1f}/255")
