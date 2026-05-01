"""DCA1000 control plane.

Thin wrapper around TI's ``DCA1000EVM_CLI_Control.exe`` (ships with
mmWave Studio at ``mmWaveStudio/PostProc/``). Re-implementing the
binary UDP control protocol from scratch is brittle — TI ships
firmware updates that can change the wire format — so we shell out
to the official tool for the rare commands we send (version query,
record start/stop). For high-rate paths (data UDP receive) we use
plain Python sockets.

The CLI tool needs a ``cf.json`` config file pointing at the right
host/DCA IPs and ports. We generate one dynamically from the
``radar.dca`` block in app_config.yaml so this stays machine-portable.

Methods raise ``DCAControlError`` on any failure that isn't a clean
"system is disconnected" sentinel (which is non-fatal — it just means
the AWR isn't streaming yet, which is the normal idle state).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from common.logging_setup import get_logger

log = get_logger(__name__)


class DCAControlError(RuntimeError):
    """Raised when DCA1000EVM_CLI_Control.exe fails in a way we
    couldn't recover from (binary missing, malformed output, etc)."""


@dataclass
class FpgaVersion:
    """Result of the ``fpga_version`` CLI command.

    The CLI prints lines like ``FPGA Version : 2.9 [Record]`` — we
    parse the version number and the build flavour ("Record" vs
    "Playback") so callers don't have to."""
    raw: str               # the full stdout line, for logging
    version: str           # e.g. "2.9"
    flavor: str            # "Record" or "Playback"


@dataclass
class SystemStatus:
    """Result of the ``query_sys_status`` CLI command."""
    raw: str
    connected: bool        # True iff CLI reported "System is connected."


class DCAControl:
    """Owns the lifecycle around DCA1000EVM_CLI_Control.exe.

    Parameters
    ----------
    cli_path
        Absolute path to ``DCA1000EVM_CLI_Control.exe``. Defaults to
        the standard mmWave Studio install location.
    host_ip, dca_ip
        Static IPs to put in the generated cf.json.
    config_port, data_port
        UDP ports for the FPGA control plane and data plane.
    timeout_s
        Per-CLI-invocation timeout. The CLI is fast (UDP round-trip);
        anything above 2 s indicates a wedged FPGA or a missing IP route.
    """

    DEFAULT_CLI_PATH = (
        r"C:\ti\mmwave_studio_03_01_04_04\mmWaveStudio\PostProc\DCA1000EVM_CLI_Control.exe"
    )

    def __init__(
        self,
        *,
        cli_path: Optional[str] = None,
        host_ip: str = "192.168.33.30",
        dca_ip: str = "192.168.33.180",
        config_port: int = 4096,
        data_port: int = 4098,
        timeout_s: float = 1.5,
    ) -> None:
        self.cli_path = cli_path or self.DEFAULT_CLI_PATH
        self.host_ip = host_ip
        self.dca_ip = dca_ip
        self.config_port = int(config_port)
        self.data_port = int(data_port)
        self.timeout_s = float(timeout_s)
        self._cf_path: Optional[Path] = None  # lazy-written cf.json

    # ───────────────────────── cf.json materialization ─────────────────────
    def _ensure_cf(self) -> Path:
        """Write our cf.json next to the CLI binary in a temp dir.

        Done lazily on first use; cached for the rest of the process
        lifetime. We have to put it next to the binary because the
        CLI's PostProc folder has a working-directory dependency
        (libgcc, libstdc++, Qt5* DLLs all live there) — we run the
        CLI with cwd=PostProc to satisfy that, but the cf.json itself
        can live anywhere as long as we pass an absolute path."""
        if self._cf_path is not None and self._cf_path.exists():
            return self._cf_path
        tmp_dir = Path(tempfile.gettempdir()) / "seeker_dca"
        tmp_dir.mkdir(exist_ok=True)
        path = tmp_dir / "cf.json"
        # ─── DCA1000 anti-buffer-overflow tuning ───────────────────────
        # Default packetDelay_us=25 + LVDS at ~50 MB/s on AWR2944P DDM
        # overruns the FPGA's internal LVDS buffer after a 5-30 s burst,
        # which TI logs internally as MMWSDK-2560 + LVDS_PATH_ERR_LED.
        # Customer-confirmed fix on TI E2E forum (threads 1163474 +
        # 1195414 + 1207190): bump packetDelay_us to 75 (gives Ethernet
        # more time per packet, FPGA buffer drains between bursts) so
        # streaming stays continuous instead of locking up.
        cf = {
            "DCA1000Config": {
                "dataLoggingMode": "raw",
                "dataTransferMode": "LVDSCapture",
                "dataCaptureMode": "ethernetStream",
                "lvdsMode": 1,
                "dataFormatMode": 3,
                "packetDelay_us": 200,  # 25 → 75 → 200 escalation
                "ethernetConfig": {
                    "DCA1000IPAddress": self.dca_ip,
                    "DCA1000ConfigPort": self.config_port,
                    "DCA1000DataPort": self.data_port,
                },
                "ethernetConfigUpdate": {
                    "systemIPAddress": self.host_ip,
                    "DCA1000IPAddress": self.dca_ip,
                    # MAC is read from EEPROM at update-time; placeholder
                    # is fine for query-only commands.
                    "DCA1000MACAddress": "12.34.56.78.90.12",
                    "DCA1000ConfigPort": self.config_port,
                    "DCA1000DataPort": self.data_port,
                },
                "captureConfig": {
                    "fileBasePath": str(tmp_dir),
                    "filePrefix": "seeker_dca",
                    "maxRecFileSize_MB": 1024,
                    "sequenceNumberEnable": 1,
                    "captureStopMode": "infinite",
                    "bytesToCapture": 4000,
                    "durationToCapture_ms": 4000,
                    "framesToCapture": 0,
                },
                "dataFormatConfig": {
                    "MSBToggle": 0,
                    "laneFmtMap": 0,
                    "reorderEnable": 0,
                    "dataPortConfig": [
                        {"portIdx": 0, "dataType": "real"},
                        {"portIdx": 1, "dataType": "real"},
                        {"portIdx": 2, "dataType": "real"},
                        {"portIdx": 3, "dataType": "real"},
                    ],
                },
            }
        }
        path.write_text(json.dumps(cf, indent=2))
        self._cf_path = path
        return path

    # ───────────────────────── CLI exec ────────────────────────────────────
    def _run(self, command: str, *, fire_and_forget: bool = False) -> subprocess.CompletedProcess:
        """Invoke ``DCA1000EVM_CLI_Control.exe <command> <cf.json>``
        without leaving a visible console window OR a hung process
        on Windows 11.

        Two pathologies we have to defend against:

        1. **Visible popup** — the TI CLI calls ``AllocConsole`` and
           writes its status text directly to the console handle, so
           our ``stdout=PIPE`` redirection doesn't catch it. On
           Windows 11 with Windows Terminal as the default console
           handler, AllocConsole spawns a visible WT tab even with
           ``CREATE_NO_WINDOW`` set on our subprocess. The
           workaround is to launch the CLI via ``cmd /c`` — cmd is
           a well-behaved console subsystem app whose own no-window
           state propagates to its child, and whose AllocConsole
           call is intercepted by ``CREATE_NO_WINDOW`` reliably.
           ``DETACHED_PROCESS`` is intentionally NOT set: per
           CreateProcess docs it is mutually exclusive with
           ``CREATE_NO_WINDOW``, and combining them silently
           re-enables the popup.

        2. **Hung process blocking Seeker startup** — the user
           reported that Seeker could not finish starting until
           they manually closed the popup. That's the CLI
           hanging on a "press any key" / UDP-ack wait that never
           comes. Mitigations:
             - Tight per-call timeout (default 1.5s) — the CLI
               sends a single UDP packet and prints status; if
               that takes more than a second the FPGA is wedged
               and waiting longer won't help.
             - On timeout, force-kill the entire process tree
               (the cmd wrapper plus its CLI child) so no
               orphaned WT tab survives.
             - ``fire_and_forget=True`` — for ``start_record``
               specifically, the UDP "start streaming" command
               is one-shot; we do not need to wait for the CLI
               to print its success line. Returning immediately
               with a synthesized CompletedProcess unblocks
               Seeker startup even if WT eats the popup.
        """
        cli = Path(self.cli_path)
        if not cli.exists():
            raise DCAControlError(f"DCA1000EVM_CLI_Control.exe not found at {cli}")
        cf = self._ensure_cf()
        # Run with cwd=PostProc so the CLI finds its sibling DLLs.
        cwd = cli.parent

        # STARTUPINFO + SW_HIDE is the belt; CREATE_NO_WINDOW is the
        # suspenders. We need both because TI's CLI mixes console
        # output with explicit CreateWindow calls (the cf.json view
        # dialog from older builds), and SW_HIDE catches the latter.
        startupinfo = None
        creation_flags = 0
        if hasattr(subprocess, "STARTUPINFO"):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
            startupinfo.wShowWindow = 0  # SW_HIDE
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        # Wrap with cmd /c. Some Windows 11 setups have Windows
        # Terminal registered as the default ConPTY host; running
        # the TI CLI directly causes WT to pop a tab even with
        # CREATE_NO_WINDOW. Going through cmd ensures the legacy
        # conhost path is taken and the no-window flag actually
        # suppresses the window. The "" after /c is an empty title
        # so the WT tab title (if it leaks anyway) is blank instead
        # of leaking the cwd path.
        argv = ["cmd.exe", "/c", "", str(cli), command, str(cf)]

        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                startupinfo=startupinfo,
                creationflags=creation_flags,
            )
        except FileNotFoundError as e:
            raise DCAControlError(f"DCA CLI failed to launch: {e}") from e

        if fire_and_forget:
            # Don't wait. The FPGA already received the UDP packet by
            # the time the CLI gets to spawn its console output; we
            # don't need the printable confirmation. Schedule a
            # background reaper so the OS handle is reclaimed.
            try:
                proc.poll()
            except Exception:
                pass
            try:
                # Drain any stdout/stderr already in the pipe (so we
                # have something to log) but DON'T block.
                stdout, stderr = proc.communicate(timeout=0.05)
            except subprocess.TimeoutExpired:
                stdout = stderr = ""
            return subprocess.CompletedProcess(
                args=proc.args, returncode=proc.returncode if proc.returncode is not None else 0,
                stdout=stdout or "", stderr=stderr or "",
            )

        try:
            stdout, stderr = proc.communicate(timeout=self.timeout_s)
            return subprocess.CompletedProcess(
                args=proc.args, returncode=proc.returncode,
                stdout=stdout, stderr=stderr,
            )
        except subprocess.TimeoutExpired as e:
            # Kill the cmd wrapper + the CLI grandchild. terminate()
            # alone leaves the CLI orphaned (it's cmd's child, not
            # ours), so we use taskkill /T /F to nuke the whole
            # subtree. Without this, every timed-out invocation
            # leaks a popup that the user has to close manually —
            # exactly the symptom that motivated this rewrite.
            try:
                proc.kill()
            except Exception:
                pass
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True, timeout=1.0,
                    creationflags=creation_flags,
                    startupinfo=startupinfo,
                )
            except Exception:
                pass
            raise DCAControlError(
                f"DCA CLI '{command}' timed out after {self.timeout_s}s"
            ) from e

    # ───────────────────────── high-level commands ────────────────────────
    _VERSION_RE = re.compile(
        r"FPGA Version\s*:\s*([\d.]+)\s*\[(\w+)\]", re.IGNORECASE,
    )

    def fpga_version(self) -> FpgaVersion:
        """Read the FPGA bitstream version. UDP round-trip; fast.

        Raises ``DCAControlError`` if the CLI didn't print a parseable
        version line — that means the DCA wasn't reachable on UDP."""
        cp = self._run("fpga_version")
        text = (cp.stdout or "") + (cp.stderr or "")
        m = self._VERSION_RE.search(text)
        if not m:
            raise DCAControlError(
                f"fpga_version: no version line in output (rc={cp.returncode}): {text!r}"
            )
        return FpgaVersion(raw=m.group(0).strip(),
                           version=m.group(1),
                           flavor=m.group(2))

    def query_sys_status(self) -> SystemStatus:
        """Ask the FPGA whether the AWR is streaming.

        "System is connected" = AWR is actively pushing samples on
        LVDS. "System is disconnected" = LVDS idle / no clock — this
        is the normal state when mmw_demoDDM is running (it doesn't
        emit LVDS in our default cfg)."""
        cp = self._run("query_sys_status")
        text = (cp.stdout or "") + (cp.stderr or "")
        connected = "System is connected" in text
        # "System is disconnected" or any other status are both not-connected.
        return SystemStatus(raw=text.strip().splitlines()[-1] if text.strip() else "",
                            connected=connected)

    # ───────────────────────── capture lifecycle ──────────────────────────
    def reset_fpga(self) -> None:
        """Soft-reset the DCA FPGA. Clears any prior recording state.
        Fire-and-forget: failures are non-fatal (the next
        setup_capture would fail with a clearer message if
        anything is wrong) and we never want this to block
        Seeker startup behind a stuck CLI popup."""
        try:
            self._run("reset_fpga", fire_and_forget=True)
        except DCAControlError:
            pass

    def setup_capture(self) -> None:
        """Configure the FPGA for raw-ADC capture, in the order TI
        expects:

        1. ``fpga``    — CONFIG_FPGA_GEN — applies dataFormatConfig
           from cf.json (lvdsMode, dataFormatMode, lane map).
        2. ``record``  — CONFIG_PACKET_DATA — applies packetDelay_us
           from cf.json (gap between consecutive UDP packets, prevents
           overrunning the host NIC at high data rates).

        After this, ``start_record()`` puts the FPGA into forwarding
        mode and UDP packets begin flowing to the host data port."""
        # Order matters per SPRUIJ4A §5.
        cp = self._run("fpga")
        if cp.returncode != 0:
            raise DCAControlError(
                f"DCA fpga config failed (rc={cp.returncode}): "
                f"{(cp.stdout or '') + (cp.stderr or '')}"
            )
        cp = self._run("record")
        if cp.returncode != 0:
            raise DCAControlError(
                f"DCA record config failed (rc={cp.returncode}): "
                f"{(cp.stdout or '') + (cp.stderr or '')}"
            )

    def start_record(self) -> None:
        """Tell the FPGA to start forwarding LVDS samples to UDP.

        FIRE-AND-FORGET. The CLI sends a single UDP packet to the
        FPGA and prints "Start Record command : Success" — but on
        Windows 11 with WT default-handler, the print can block
        the CLI process indefinitely (the popup the user has to
        close manually). We don't need the print: by the time we
        get here, the UDP packet has already been delivered, so we
        return immediately and let the CLI process die in the
        background. If UDP didn't get through to the FPGA, the
        symptom shows up downstream as zero data on the UDP data
        port — DataPortListener.connected goes False and the GUI
        shows the radar disconnected pill, which is the right
        diagnostic anyway."""
        self._run("start_record", fire_and_forget=True)

    def stop_record(self) -> None:
        """Take the FPGA out of forwarding mode. Safe to call from
        DCAManager.stop() even if start_record was never issued.
        Also fire-and-forget so a Seeker shutdown never hangs."""
        try:
            self._run("stop_record", fire_and_forget=True)
        except DCAControlError:
            # Swallow — stop should never raise.
            pass
