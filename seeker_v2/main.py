"""seeker_v2.main — multi-process orchestrator.

Lifecycle:
  1. Read config
  2. Allocate shared memory for each sensors frame ring
  3. Spawn capture processes (EO, thermal, radar)
  4. Spawn inference + fusion processes
  5. Run GUI/WS server in main process (asyncio)
  6. Coordinate shutdown on SIGTERM

This is a Phase 2.1 stub. Filling in incrementally.
"""
import sys
import logging

def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log = logging.getLogger("seeker_v2")
    log.info("seeker_v2 main: stub — not yet runnable. See deploy/jetson/PERF_REWRITE_PLAN.md.")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
