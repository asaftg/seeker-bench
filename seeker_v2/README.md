# seeker_v2 — Multi-Process Architecture Rewrite

Branched off `JETSON_BASELINE` (commit c539c28). Sibling to v1 seeker.

## Status: SCAFFOLDING (early Phase 2.1)

The v1 baseline at `~/seeker-bench/{eo,thermal,radar,fusion,gui}/` is
unchanged. This `seeker_v2/` directory is the start of the
multi-process rewrite. Run v1 as today; v2 is parallel and not yet
end-to-end runnable.

## Architecture (target)

```
EO_CAP ──┐                                  ┌── Inference (TRT)
THM_CAP ─┼── shared memory ── orchestrator ──┤
RDR_CAP ─┘                                   └── Fusion
                                              └── GUI/WS
```

Each capture is a separate OS process (own GIL). Frames cross
process boundaries via `multiprocessing.shared_memory.SharedMemory`
with single-producer-single-consumer ring of (frame_id, mtime, idx)
descriptors over `multiprocessing.Queue`.

## Phase progression

- **2.1** (current): Multi-process Python scaffolding. EXPECTED
  EO 18-22 Hz, thermal 22-25 Hz.
- **2.2**: pybind11 C++ V4L2 backend. EXPECTED EO 23-27 Hz.
- **2.3**: nvjpeg hardware encoder.
- **2.4**: Optional DLA inference offload.
- **2.5**: JS frontend RAF render queue.
- **2.6**: Pure C++ orchestrator (only if 2.1-2.4 insufficient).

See `deploy/jetson/PERF_REWRITE_PLAN.md` for the full plan.

## Build/run

Phase 2.1 (Python multi-process):
```
cd ~/seeker-bench
python3 -m seeker_v2.main --config config/app_config.yaml
```

Phase 2.2+ (with C++ extension):
```
cd ~/seeker-bench/seeker_v2/native
mkdir build && cd build
cmake .. && make -j4
# .so will be in seeker_v2/native/seeker_native.cpython-38-aarch64-linux-gnu.so
cd ~/seeker-bench
python3 -m seeker_v2.main --config config/app_config.yaml
```

## Reverting

The v1 baseline is at jetson branch HEAD. To go back:
```
git reset --hard JETSON_BASELINE     # safe rollback to baseline
# OR
git reset --hard JETSON_SMALL_OPTIMIZATION   # rollback to Phase 1
```

The `jetson-v2-rewrite` branch will never be merged into `jetson` or
`main` until the user explicitly authorizes.

