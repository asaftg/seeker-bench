#!/usr/bin/env bash
# Hard-reset Seeker hardware before launching main.py.
# Clears wedged states from prior kill-restart cycles:
#   - IMX568 (EO)  USB power-cycle
#   - Boson (Thermal) USB power-cycle (cheap, idempotent)
#   - TI XDS110 (Radar)  USB power-cycle  forces sensorStart to ack again
#   - port 8080 stale socket killed via fuser
# All operations require NOPASSWD sudo (already configured).
set -e

EO_USB="/sys/bus/usb/devices/2-4.4.3"
THERMAL_USB="/sys/bus/usb/devices/1-4.4.1"
RADAR_USB="/sys/bus/usb/devices/1-4.1"

echo "[reset] kill any seeker process + free port 8080"
pkill -9 -f "python.*seeker-bench/main.py" 2>/dev/null || true
sudo fuser -k 8080/tcp 2>/dev/null || true
sleep 1

reset_usb() {
    local label="$1"; local path="$2"
    if [ -e "$path/authorized" ]; then
        echo "[reset] USB-cycle $label ($path)"
        sudo sh -c "echo 0 > $path/authorized"
        sleep 1
        sudo sh -c "echo 1 > $path/authorized"
    else
        echo "[reset] $label not at $path  skipping"
    fi
}

reset_usb "IMX568 (EO)"      "$EO_USB"
reset_usb "Boson (Thermal)"  "$THERMAL_USB"
reset_usb "TI XDS110 (Radar)" "$RADAR_USB"

echo "[reset] waiting 5s for udev to re-create symlinks..."
sleep 5

echo "[reset] device tree:"
ls -l /dev/seeker_eo_v /dev/seeker_thermal_ctrl /dev/seeker_radar_cli /dev/seeker_radar_data 2>/dev/null || echo "  (one or more symlinks missing)"

echo "[reset] running leopard XU init (disable trigger mode)..."
~/seeker-bench/.venv/bin/python ~/seeker-bench/scripts/leopard_xu_init.py 2>&1 | tail -3 || true

echo "[reset] done"
