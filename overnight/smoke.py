"""Prove a candidate compiles and plays legally before spending an hour measuring it.

A patch that crashes or emits an illegal move would still produce an SPRT number - a
very bad one - and we would have paid twenty minutes to learn something a thirty second
check catches. Every loss here is a loss on the platform too, so this gate is the same
shape as the real referee: make the move, check it against python-chess, keep the clock.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import time
from pathlib import Path

import chess


def load(path: str, tag: str):
    spec = importlib.util.spec_from_file_location(f"smoke_{tag}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"smoke_{tag}"] = module
    spec.loader.exec_module(module)
    return module


def check(path: str, games: int = 2, base_ms: int = 3000, inc_ms: int = 40) -> None:
    started = time.monotonic()
    agent = load(path, Path(path).stem)
    compiled = time.monotonic() - started
    if not getattr(agent, "JIT_READY", False):
        raise SystemExit(f"{path}: JIT did not come up: {getattr(agent, 'JIT_ERROR', '')}")

    rng = random.Random(7)
    plies = 0
    for game in range(games):
        board = chess.Board()
        for _ in range(rng.randint(2, 6)):
            moves = list(board.legal_moves)
            if moves:
                board.push(rng.choice(moves))
        clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
        for name in ("SEEN", "TT", "HISTORY", "COUNTER"):
            holder = getattr(agent, name, None)
            if isinstance(holder, dict):
                holder.clear()
        fens = getattr(agent, "HISTORY_FENS", None)
        if isinstance(fens, list):
            del fens[:]

        while not board.is_game_over(claim_draw=True) and board.ply() < 120:
            tick = time.monotonic()
            uci = agent.get_move(board.fen(), int(max(0, clock[board.turn])))
            spent = (time.monotonic() - tick) * 1000.0
            clock[board.turn] -= spent
            if clock[board.turn] < 0:
                raise SystemExit(f"{path}: flagged on move {board.fullmove_number}")
            clock[board.turn] += inc_ms
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise SystemExit(f"{path}: illegal move {uci} in {board.fen()}")
            board.push(move)
            plies += 1

    elapsed = time.monotonic() - started
    print(f"  {Path(path).name:16} compiled {compiled:5.1f}s  {plies:4} plies legal  "
          f"{elapsed:5.1f}s total", flush=True)


if __name__ == "__main__":
    for target in sys.argv[1:]:
        check(target)
