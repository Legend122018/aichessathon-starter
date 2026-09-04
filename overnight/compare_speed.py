"""Where does the gap come from - searching faster, or searching better?

A 300 Elo gap is either node rate or knowledge, and the answer decides whether it can be
closed by tuning or only by rewriting. This gives both engines the same positions and the
same wall clock and reports what each got through, and how deep it got.

    python overnight/compare_speed.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import chess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

POSITIONS = [
    ("opening", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("middlegame", "r2q1rk1/1b1nbppp/p2ppn2/1p6/3NPP2/1BN1B3/PPPQ2PP/2KR3R w - - 0 12"),
    ("endgame", "8/5pk1/6p1/3K3p/2P4P/6P1/5P2/8 w - - 0 40"),
]
SECONDS = 4.0


def load(name: str, path: Path, add_to_path: Path | None = None):
    if add_to_path is not None:
        sys.path.insert(0, str(add_to_path))
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    started = time.monotonic()
    spec.loader.exec_module(module)
    print(f"  {name} loaded in {time.monotonic() - started:.1f}s", flush=True)
    return module


def ours_nps(agent, fen: str, seconds: float) -> tuple[int, float, int]:
    """Search a fixed number of nodes and time it, then report depth reached separately."""
    work = agent.jit_make_workspace()
    board, st = agent.jit_new_state(fen)
    best_depth = 0
    total_nodes = 0
    started = time.perf_counter()
    for depth in range(1, 40):
        work["info"][agent.NODES] = 0
        work["info"][agent.STOPPED] = 0
        work["info"][agent.REP_LEN] = 0
        work["info"][agent.NODE_LIMIT] = 40_000_000
        t0 = time.perf_counter()
        agent.jit_search_root(
            board, st, work["hist"], work["moves"], work["scores"], work["occ"],
            work["info"], work["killers"], work["hist_heur"], work["counter"], work["rep"],
            work["tt_key"], work["tt_move"], work["tt_score"], work["tt_meta"],
            depth, -31000, 31000, 0, work["out"])
        total_nodes += int(work["info"][agent.NODES])
        best_depth = depth
        if time.perf_counter() - started > seconds:
            break
        if time.perf_counter() - t0 > seconds / 2:
            break
    spent = time.perf_counter() - started
    return total_nodes, spent, best_depth


def theirs_nps(agent, fen: str, seconds: float) -> tuple[int, float, int]:
    eng = agent.Engine() if hasattr(agent, "Engine") else None
    if eng is None:
        from engine.engine import Engine  # type: ignore
        eng = Engine()
    eng.set_position(fen) if hasattr(eng, "set_position") else None
    started = time.perf_counter()
    info = eng.search(max_depth=64, hard_ms=int(seconds * 1000), soft_ms=int(seconds * 1000))
    spent = time.perf_counter() - started
    return int(info.nodes), spent, int(info.depth)


def main() -> None:
    print("loading both engines (each compiles at import)")
    ours = load("ours", ROOT / "agent.py")
    theirs_dir = HERE / "exemplar"
    theirs = load("theirs", theirs_dir / "agent.py", theirs_dir)

    print(f"\n{'position':<14}{'engine':<10}{'nodes':>12}{'nodes/s':>12}{'depth':>8}")
    for label, fen in POSITIONS:
        n, t, d = ours_nps(ours, fen, SECONDS)
        print(f"{label:<14}{'ours':<10}{n:>12,}{n / max(t, 1e-9):>12,.0f}{d:>8}")
        try:
            n2, t2, d2 = theirs_nps(theirs, fen, SECONDS)
            print(f"{'':<14}{'theirs':<10}{n2:>12,}{n2 / max(t2, 1e-9):>12,.0f}{d2:>8}")
        except Exception as exc:
            print(f"{'':<14}{'theirs':<10}  could not drive its search directly: "
                  f"{type(exc).__name__}: {exc}")
    print()


if __name__ == "__main__":
    main()
