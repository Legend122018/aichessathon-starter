"""Prove the two evaluations agree, position by position.

agent.py carries two engines with the same meaning and different types, and COMPLIANCE.md
records that an earlier version silently rebound one over the other. A term added to one
half and mistyped in the other is the same class of bug, and it is invisible in a game -
the agent simply plays slightly worse and no test fails.

    python overnight/check_halves.py [agent.py] [positions]
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import chess

HERE = Path(__file__).resolve().parent


def load(path: str):
    spec = importlib.util.spec_from_file_location("halves_agent", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["halves_agent"] = module
    spec.loader.exec_module(module)
    return module


def sample(count: int) -> list[str]:
    rng = random.Random(11)
    out: list[str] = []
    while len(out) < count:
        board = chess.Board()
        for _ in range(rng.randint(4, 80)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if board.is_game_over():
                break
        if not board.is_game_over():
            out.append(board.fen())
    return out


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent / "agent.py")
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    agent = load(path)
    if not getattr(agent, "JIT_READY", False):
        print("the JIT did not build, so there is only one evaluation to check")
        return 1

    worst = 0
    mismatches: list[tuple[str, int, int]] = []
    for fen in sample(count):
        board = chess.Board(fen)
        searcher = agent.Searcher(float("inf"), board.turn)
        searcher.seed(board)
        python_score = searcher.evaluate(board)

        jit_board, jit_state = agent.jit_new_state(fen)
        jit_score = int(agent.jit_evaluate(jit_board, jit_state))

        gap = abs(python_score - jit_score)
        worst = max(worst, gap)
        if gap:
            mismatches.append((fen, python_score, jit_score))

    print(f"\nchecked {count} positions, worst gap {worst} cp")
    if mismatches:
        print(f"{len(mismatches)} disagree. The first few:\n")
        for fen, a, b in mismatches[:5]:
            print(f"  python {a:+6d}   jit {b:+6d}   {fen}")
        return 1
    print("the two evaluations agree exactly\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
