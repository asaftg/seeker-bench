"""Phase 3 — DCA1000 raw-ADC radar pipeline (host-side).

Lives alongside ``radar/`` (the on-chip TLV pipeline). The two are
interchangeable from the rest of the system's perspective: both publish
``RadarFrame`` on ``Topic.RADAR``. ``main.py`` spawns ONE of them at
startup based on the ``radar.backend`` config knob, and the GUI can
hot-swap them at runtime.

Currently this package contains only a scaffold ``DCAManager`` that
publishes ``connected=False`` sentinel frames — no actual DCA capture
yet. The full pipeline (UART firmware upload, mmWaveLink chirp config,
UDP raw-ADC ingest, range-FFT → Doppler-FFT → CFAR → tracker) is
implemented incrementally over Phase 1+ milestones, behind this same
``DCAManager`` interface so the rest of the app sees a stable contract.
"""

from radar_dca.dca_manager import DCAManager

__all__ = ["DCAManager"]
