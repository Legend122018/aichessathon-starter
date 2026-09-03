"""Screen a candidate deterministically, before spending games on it.

Playing games measures Elo, and Elo needs thousands of games because the signal is
buried in noise. But most of what can be wrong with a search change is not noisy at all.
A pruning rule either visits fewer nodes to reach a given depth or it does not. A
reduction rule either still finds the tactic or it does not. Both are exact, repeatable,
and cheap - two cores answer them in minutes where measuring Elo would take a day.

So this does not replace the match. It sorts candidates into "worth playing games
against" and "already known to be wrong", which is the part that does not need a
sixteen core machine.

    python screen.py champion.py try_ttsize.py try_lmrhist.py ...
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import chess

DEPTH = 9

# A spread of middlegames and endgames, plus tactics with a known answer. The quiet
# positions measure how hard the search prunes; the tactical ones check that it still
# sees what it is supposed to see when the pruning gets more aggressive.
QUIET = [
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bqk2r/pp2bppp/2n1pn2/2pp4/3P4/1P2PN2/PBPN1PPP/R2QKB1R w KQkq - 0 8",
    "r2q1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 9",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "8/5ppp/8/5PPP/8/6k1/8/6K1 w - - 0 1",
    "4rrk1/pp1n1ppp/2p2q2/8/2BP4/2P1RN2/PP3PPP/3R2K1 w - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "2rq1rk1/pb1nbppp/1p2pn2/8/2BP4/2N1PN2/PP2QPPP/2RR2K1 w - - 0 1",
]

# fen, the move a working search must find
TACTICS = [
    ("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", None),
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8"),
    ("r1b2rk1/pp1p1ppp/1b1p4/n3P3/2q1P3/2N2N2/PP3PPP/R1BQ1RK1 w - - 0 1", None),
    ("2r3k1/p4p1p/1p2p1p1/8/2q5/P1N1P3/1P3PPP/2RQ2K1 b - - 0 1", None),
    ("8/8/8/8/8/6k1/6p1/6K1 b - - 0 1", None),
    ("3r1rk1/1pp2ppp/p1n5/4P3/2Pq4/P4N2/1P3PPP/R2Q1RK1 b - - 0 1", None),
]


def load(path: str, tag: str):
    spec = importlib.util.spec_from_file_location(f"screen_{tag}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"screen_{tag}"] = module
    spec.loader.exec_module(module)
    if not getattr(module, "JIT_READY", False):
        raise SystemExit(f"{path}: JIT unavailable: {getattr(module, 'JIT_ERROR', '')}")
    return module


def fixed_depth(agent, fen: str, depth: int) -> tuple[int, str]:
    """Search to an exact depth with the clock out of the way, so the node count is a
    property of the search rather than of how busy the machine was."""
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


def measure(agent, label: str) -> dict:
    started = time.monotonic()
    nodes = 0
    for fen in QUIET:
        count, _ = fixed_depth(agent, fen, DEPTH)
        nodes += count
    found = []
    for fen, expected in TACTICS:
        _, move = fixed_depth(agent, fen, DEPTH)
        found.append(move)
        if expected and move != expected:
            print(f"    {label}: missed {expected} in {fen[:32]}... played {move}")
    return {"nodes": nodes, "moves": found, "seconds": time.monotonic() - started}


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    champion_path, *candidates = sys.argv[1:]

    print(f"fixed depth {DEPTH}, {len(QUIET)} quiet positions and {len(TACTICS)} tactical\n",
          flush=True)
    base = measure(load(champion_path, "base"), Path(champion_path).name)
    print(f"{'engine':<22}{'nodes to depth':>16}{'vs champion':>13}"
          f"{'moves changed':>15}{'seconds':>9}")
    print(f"{Path(champion_path).name:<22}{base['nodes']:>16,}{'-':>13}"
          f"{'-':>15}{base['seconds']:>8.1f}s")

    for path in candidates:
        agent = load(path, Path(path).stem)
        result = measure(agent, Path(path).name)
        ratio = result["nodes"] / base["nodes"]
        changed = sum(1 for a, b in zip(base["moves"], result["moves"]) if a != b)
        print(f"{Path(path).name:<22}{result['nodes']:>16,}{ratio:>12.2f}x"
              f"{changed:>10}/{len(TACTICS):<4}{result['seconds']:>8.1f}s", flush=True)

    print("\nFewer nodes to the same depth means the change prunes harder; whether that")
    print("is worth the lines it skips is what the games decide. Moves changed is not")
    print("good or bad on its own - it says how much of the search the change touches.")


if __name__ == "__main__":
    main()
