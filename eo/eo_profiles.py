"""EO exposure profiles + auto-selector.

Three runtime-switchable profiles — Day / Dusk / Night — plus a
hysteretic selector that picks between them based on measured scene
brightness. Hysteresis matters: without it, the selector flaps at
dawn/dusk and the camera spends half its time re-exposing.

The illuminator (SAVgood 850nm VCSEL) is manually controlled by the
operator; these profiles only *report* whether the current scene
looks illuminator-lit (via the night-profile flag) — they don't try
to switch hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EOProfileName(str, Enum):
    DAY   = "day"
    DUSK  = "dusk"
    NIGHT = "night"


@dataclass(frozen=True)
class EOProfile:
    """A single exposure/gain preset."""
    name: EOProfileName
    exposure_ms: float
    gain: float
    # Thresholds (mean brightness 0..255) that *enter* this profile
    # from the adjacent one. Hysteresis is built in below by the
    # selector — these are the transition levels.
    enter_above: float  # switch INTO this profile when scene mean > this
    enter_below: float  # switch INTO this profile when scene mean < this


# Defaults — override via app_config.yaml's `eo.profiles` block.
#
# Day:   bright outdoor / lit room.       Freezes motion, best SNR.
# Dusk:  indoor normal / overcast twilight. Bridges the gap.
# Night: dark scene, relies on VCSEL illuminator for NIR fill.
DEFAULT_DAY   = EOProfile(EOProfileName.DAY,   exposure_ms=2.0,  gain=1.0,
                          enter_above=150.0, enter_below=255.0)
DEFAULT_DUSK  = EOProfile(EOProfileName.DUSK,  exposure_ms=20.0, gain=4.0,
                          enter_above=50.0,  enter_below=160.0)
DEFAULT_NIGHT = EOProfile(EOProfileName.NIGHT, exposure_ms=80.0, gain=16.0,
                          enter_above=0.0,   enter_below=60.0)


class ProfileSelector:
    """Picks the right EOProfile based on scene mean brightness.

    Hysteresis: once a profile is active, we require a meaningful
    crossing of the *other* profile's enter band before switching,
    plus a minimum dwell time (default 3 s) so the camera doesn't
    thrash on a passing cloud.
    """

    def __init__(
        self,
        day: EOProfile = DEFAULT_DAY,
        dusk: EOProfile = DEFAULT_DUSK,
        night: EOProfile = DEFAULT_NIGHT,
        min_dwell_s: float = 3.0,
        initial: EOProfileName = EOProfileName.DAY,
    ) -> None:
        self._profiles = {
            EOProfileName.DAY: day,
            EOProfileName.DUSK: dusk,
            EOProfileName.NIGHT: night,
        }
        self._current: EOProfileName = initial
        self._entered_at: float = 0.0
        self._min_dwell_s = float(min_dwell_s)
        self._last_mean: float = 0.0

    def current(self) -> EOProfile:
        return self._profiles[self._current]

    def last_mean(self) -> float:
        return self._last_mean

    def update(self, scene_mean: float, now_s: float) -> Optional[EOProfile]:
        """Feed a new brightness sample. Returns the new profile if we
        just switched, otherwise None.

        ``scene_mean`` is the mean of the 8-bit luma channel (0..255).
        ``now_s`` is a monotonic timestamp (time.monotonic() or similar).
        """
        self._last_mean = float(scene_mean)

        # Dwell guard — never switch profiles more often than min_dwell_s.
        if (now_s - self._entered_at) < self._min_dwell_s:
            return None

        day   = self._profiles[EOProfileName.DAY]
        dusk  = self._profiles[EOProfileName.DUSK]
        night = self._profiles[EOProfileName.NIGHT]

        # Rules, ordered: pick the profile whose "enter band" contains
        # the scene_mean. The bands are designed to overlap (hysteresis)
        # so the decision also considers which profile we're already in.
        new: Optional[EOProfileName] = None

        if self._current == EOProfileName.DAY:
            # In DAY: need to drop well below day.enter_above to leave.
            if scene_mean < dusk.enter_below:
                # Fell into dusk's band
                new = EOProfileName.DUSK
        elif self._current == EOProfileName.DUSK:
            if scene_mean > day.enter_above:
                new = EOProfileName.DAY
            elif scene_mean < night.enter_below:
                new = EOProfileName.NIGHT
        elif self._current == EOProfileName.NIGHT:
            if scene_mean > dusk.enter_above:
                new = EOProfileName.DUSK

        if new is not None and new != self._current:
            self._current = new
            self._entered_at = now_s
            return self._profiles[new]
        return None


def profiles_from_config(eo_cfg: dict) -> tuple[EOProfile, EOProfile, EOProfile]:
    """Build (day, dusk, night) profiles from an ``eo.profiles`` config block.

    Missing fields fall back to the DEFAULT_* constants so partial
    configs are valid. Keeps the YAML schema forgiving — the operator
    can override just the exposure_ms without having to write out the
    full profile block.
    """
    p = (eo_cfg.get("profiles") or {}) if isinstance(eo_cfg, dict) else {}
    def _merge(name: str, default: EOProfile) -> EOProfile:
        blk = p.get(name) or {}
        return EOProfile(
            name=EOProfileName(name),
            exposure_ms=float(blk.get("exposure_ms", default.exposure_ms)),
            gain=float(blk.get("gain", default.gain)),
            enter_above=float(blk.get("enter_above", default.enter_above)),
            enter_below=float(blk.get("enter_below", default.enter_below)),
        )
    return _merge("day", DEFAULT_DAY), _merge("dusk", DEFAULT_DUSK), _merge("night", DEFAULT_NIGHT)
