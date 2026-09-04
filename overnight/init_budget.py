"""Time the exemplar's import on one core, the way the competition will run it.

The engine JIT-compiles itself at import (`_warmup`), and that has only ever been timed
on this machine, which has 32 cores. The competition gives one core and 60 seconds of
initialization. LLVM does use threads, so a 20 second compile here is no evidence at all
about a 1 core box - and if compilation overruns the budget the entry does not start a
single game.

Affinity is set before `agent` is imported, so every part of the compile is charged to
one core, and NUMBA_NUM_THREADS is pinned to match what the entry would see there.

    python overnight/init_budget.py            # one core, the competition's case
    python overnight/init_budget.py --all      # unrestricted, for comparison
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXEMPLAR = HERE / "exemplar"
BUDGET_S = 60.0

one_core = "--all" not in sys.argv

if one_core:
    # Pin to CPU 0 before numba is imported, so compilation cannot quietly spread
    # across the other 31 cores and report a time the competition will never see.
    # The argtypes are not optional: a HANDLE is 64-bit here, and letting ctypes
    # guess truncates it to an int, so the call fails with "invalid handle".
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int
    if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), 1):
        raise SystemExit(f"could not set affinity: {ctypes.get_last_error()}")
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

sys.path.insert(0, str(EXEMPLAR))
os.chdir(EXEMPLAR)

started = time.perf_counter()
import agent  # noqa: E402  (the import is the thing being measured)
elapsed = time.perf_counter() - started

# A move on a cold table, because the first real call pays for anything `_warmup`
# missed and that cost lands on the game clock rather than the init budget.
move_started = time.perf_counter()
first = agent.get_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 120_000)
move_elapsed = time.perf_counter() - move_started

where = "one core" if one_core else "all cores"
print(f"import + warmup on {where}: {elapsed:.1f}s of the {BUDGET_S:.0f}s budget")
print(f"first move: {first} in {move_elapsed:.2f}s")

if one_core:
    headroom = BUDGET_S - elapsed
    print(f"headroom: {headroom:+.1f}s")
    if headroom < 0:
        print("FAIL: the entry would not finish initializing")
        sys.exit(1)
    if headroom < 15:
        print("WARNING: thin. A slower competition box could overrun this.")
        sys.exit(2)
    print("PASS")
