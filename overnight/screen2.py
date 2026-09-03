"""Screen search variants against a deeper search's answers, not just node counts.

Fewer nodes is not a virtue on its own. Late move pruning cut nodes and lost 38 Elo,
because the lines it skipped were lines that mattered. What separates a good pruning
change from a reckless one is whether the cheaper search still reaches the conclusion
the expensive search reaches.

So: run the champion deep once, keep its move for each position as the reference, then
ask each variant at shallow depth how often it agrees. Both halves are deterministic, so
two cores answer in minutes what would take a day of games - and the answer is
predictive rather than merely descriptive.

    python screen2.py sc.py sw_*.py

Agreement is a filter, not a ranking. A variant that prunes hard and keeps agreement is
worth games; one that prunes hard and loses agreement is over-pruning and is not.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import chess

REFERENCE_DEPTH = 11
TEST_DEPTH = 8
CACHE = Path(__file__).resolve().parent / "screen_reference.json"


def load(path: str, tag: str):
    spec = importlib.util.spec_from_file_location(f"scr_{tag}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"scr_{tag}"] = module
    spec.loader.exec_module(module)
    if not getattr(module, "JIT_READY", False):
        raise SystemExit(f"{path}: JIT unavailable: {getattr(module, 'JIT_ERROR', '')}")
    return module


def positions(count: int = 24) -> list[str]:
    """A fixed, reproducible spread: openings, middlegames, endgames.

    Seeded so the set is identical on every run - a screen whose positions moved between
    runs would compare variants against different questions.
    """
    rng = random.Random(20260902)
    out: list[str] = []
    while len(out) < count:
        board = chess.Board()
        for _ in range(rng.randint(8, 70)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over() and len(list(board.legal_moves)) > 4:
            out.append(board.fen())
    return out


def search(agent, fen: str, depth: int) -> tuple[int, str]:
    """Exact depth, clock out of the way, tables cleared: the same question every time."""
    work = agent.JIT_STATE["work"]
    board, st = agent.jit_new_state(fen)
    for key in ("tt_key", "tt_move", "tt_score", "tt_meta", "killers", "hist_heur",
                "counter"):
        work[key][:] = 0
    work["info"][agent.NODES] = 0
    work["info"][agent.STOPPED] = 0
    work["info"][agent.REP_LEN] = 0
    work["info"][agent.NODE_LIMIT] = 1 << 40
    best = 0
    for d in range(1, depth + 1):
        agent.jit_search_root(
            board, st, work["hist"], work["moves"], work["scores"], work["occ"],
            work["info"], work["killers"], work["hist_heur"], work["counter"],
            work["rep"], work["tt_key"], work["tt_move"], work["tt_score"],
            work["tt_meta"], d, -31000, 31000, best, work["out"])
        best = int(work["out"][1])
    return int(work["info"][agent.NODES]), agent.jit_move_uci(best) if best else ""


def reference(champion_path: str, fens: list[str]) -> dict[str, str]:
    if CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if cached.get("depth") == REFERENCE_DEPTH and cached.get("fens") == fens:
            print(f"reference: {len(fens)} positions at depth {REFERENCE_DEPTH} (cached)\n",
                  flush=True)
            return cached["moves"]
    print(f"reference: searching {len(fens)} positions at depth {REFERENCE_DEPTH}...",
          flush=True)
    agent = load(champion_path, "ref")
    started = time.monotonic()
    moves = {}
    for fen in fens:
        _, move = search(agent, fen, REFERENCE_DEPTH)
        moves[fen] = move
    CACHE.write_text(json.dumps({"depth": REFERENCE_DEPTH, "fens": fens, "moves": moves}))
    print(f"reference done in {time.monotonic() - started:.0f}s\n", flush=True)
    return moves


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    champion_path, *variants = sys.argv[1:]
    fens = positions()
    truth = reference(champion_path, fens)

    def measure(path: str, tag: str) -> tuple[int, int]:
        agent = load(path, tag)
        nodes = 0
        agree = 0
        for fen in fens:
            count, move = search(agent, fen, TEST_DEPTH)
            nodes += count
            agree += move == truth[fen]
        return nodes, agree

    base_nodes, base_agree = measure(champion_path, "base")
    print(f"at depth {TEST_DEPTH}, against the depth {REFERENCE_DEPTH} answer:\n")
    print(f"{'variant':<20}{'nodes':>12}{'vs base':>10}{'agrees':>9}{'vs base':>10}")
    print(f"{'champion':<20}{base_nodes:>12,}{'1.00x':>10}"
          f"{f'{base_agree}/{len(fens)}':>9}{'-':>10}")

    rows = []
    for path in variants:
        nodes, agree = measure(path, Path(path).stem)
        rows.append((Path(path).stem.replace("sw_", ""), nodes, agree))
        print(f"{Path(path).stem.replace('sw_', ''):<20}{nodes:>12,}"
              f"{nodes / base_nodes:>9.2f}x{f'{agree}/{len(fens)}':>9}"
              f"{agree - base_agree:>+10}", flush=True)

    print("\nWorth games: fewer nodes with agreement held or improved.")
    print("Not worth games: agreement dropped - the saving came out of real lines.")
    keep = [r for r in rows if r[1] < base_nodes * 0.995 and r[2] >= base_agree]
    if keep:
        print("\nshortlist: " + ",".join(name for name, _, _ in
                                         sorted(keep, key=lambda r: r[1])))


if __name__ == "__main__":
    main()
