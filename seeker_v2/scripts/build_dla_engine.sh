#!/usr/bin/env bash
# Build a DLA-targeted TensorRT engine from an ultralytics-exported ONNX.
#
# Phase 2.4 of the seeker_v2 rewrite: the thermal H/V YOLO classifier is
# moved off the iGPU and onto DLA0 so it doesn't fight the EO classifier
# for SM time. Most YOLO ops are DLA-supported in FP16; the few that
# aren't (typically a final reshape or NMS path) gracefully fall back to
# the GPU via `--allowGPUFallback`.
#
# Usage:
#   ./build_dla_engine.sh <input.onnx> <output.engine> [dla_core] [imgsz]
#
# Example:
#   # First export from ultralytics (run once on Jetson):
#   yolo export model=models/seeker_thermal_hv_v2.pt format=onnx \
#       imgsz=640 simplify=True opset=13 device=0
#
#   # Then build an FP16/DLA0 engine:
#   ./build_dla_engine.sh \
#       models/seeker_thermal_hv_v2.onnx \
#       models/seeker_thermal_hv_v2_dla0.engine \
#       0 640
#
# Verification:
#   trtexec --loadEngine=models/seeker_thermal_hv_v2_dla0.engine \
#           --useDLACore=0 --fp16 --warmUp=200 --iterations=500
#
# DLA constraints to remember:
#   - Max 1 active engine per core. Don't load the same engine on
#     DLA0 *and* DLA1 unless you really need it.
#   - Input shapes are static. Re-export ONNX if you change imgsz.
#   - Quantization (INT8) on DLA needs a calibration cache; FP16 is the
#     practical choice for now.
set -euo pipefail

ONNX="${1:?onnx path required}"
ENGINE="${2:?engine output path required}"
DLA_CORE="${3:-0}"
IMGSZ="${4:-640}"

if ! command -v trtexec >/dev/null 2>&1; then
    echo "trtexec not on PATH. On JetPack 5 it lives at /usr/src/tensorrt/bin/trtexec." >&2
    exit 1
fi

if [[ ! -f "$ONNX" ]]; then
    echo "ONNX not found: $ONNX" >&2
    exit 1
fi

mkdir -p "$(dirname "$ENGINE")"

echo "[build_dla_engine] $ONNX -> $ENGINE (DLA${DLA_CORE}, imgsz=${IMGSZ})"

trtexec \
    --onnx="$ONNX" \
    --saveEngine="$ENGINE" \
    --useDLACore="$DLA_CORE" \
    --allowGPUFallback \
    --fp16 \
    --workspace=2048 \
    --shapes=images:1x3x${IMGSZ}x${IMGSZ} \
    --buildOnly

echo "[build_dla_engine] done. Run a quick perf check with:"
echo "  trtexec --loadEngine=$ENGINE --useDLACore=$DLA_CORE --fp16 \\"
echo "          --warmUp=200 --iterations=500 --avgRuns=100"
