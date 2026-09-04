"""Measure what mobility and king safety would cost the search, before writing them.

probe.py asked this question of a neural evaluation and the answer was no. The same
question deserves asking here, because the same trap is open: the evaluation already
accounts for about 82% of node cost, so a term that walks every sliding ray is not
obviously affordable just because it is a good term in other engines.

Mobility counts the squares each knight, bishop, rook and queen can reach. King safety
counts enemy attacks landing on the eight squares around each king. Both are measured on
real positions from the search, against the real evaluation, in the same jitted style the
engine uses.

    python overnight/probe_mobility.py ../agent.py
"""

from __future__ import annotations

import importlib.util
import random
import sys
import time
from pathlib import Path

import chess
from numba import njit

HERE = Path(__file__).resolve().parent


def load(path: str):
    spec = importlib.util.spec_from_file_location("probe_agent", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["probe_agent"] = module
    spec.loader.exec_module(module)
    if not getattr(module, "JIT_READY", False):
        raise SystemExit("the agent is running without its JIT; nothing to measure")
    return module


def positions(count: int) -> list[str]:
    """Positions from real games, not the opening, where mobility actually varies."""
    rng = random.Random(7)
    out: list[str] = []
    while len(out) < count:
        board = chess.Board()
        for _ in range(rng.randint(12, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if board.is_game_over():
                break
        if not board.is_game_over():
            out.append(board.fen())
    return out


def build_probes(agent):
    """Mobility and king safety, written the way the engine would have to write them."""
    knight_steps = agent.KNIGHT_STEPS
    king_steps = agent.KING_STEPS
    bishop_steps = agent.BISHOP_STEPS
    rook_steps = agent.ROOK_STEPS

    @njit(cache=False)
    def on_board(sq):
        return (sq & 0x88) == 0

    @njit(cache=False)
    def mobility(board):
        """Reachable squares per piece, white minus black. One ray walk per slider."""
        score = 0
        for sq in range(128):
            if (sq & 0x88) != 0:
                continue
            piece = board[sq]
            if piece == 0:
                continue
            kind = piece & 7
            colour = piece >> 3
            sign = 1 if colour == 0 else -1
            count = 0
            if kind == 2:
                for i in range(8):
                    to = sq + knight_steps[i]
                    if on_board(to):
                        other = board[to]
                        if other == 0 or (other >> 3) != colour:
                            count += 1
            elif kind == 3 or kind == 5:
                for i in range(4):
                    step = bishop_steps[i]
                    to = sq + step
                    while on_board(to):
                        other = board[to]
                        if other != 0:
                            if (other >> 3) != colour:
                                count += 1
                            break
                        count += 1
                        to += step
            if kind == 4 or kind == 5:
                for i in range(4):
                    step = rook_steps[i]
                    to = sq + step
                    while on_board(to):
                        other = board[to]
                        if other != 0:
                            if (other >> 3) != colour:
                                count += 1
                            break
                        count += 1
                        to += step
            score += sign * count
        return score

    @njit(cache=False)
    def king_zone_pressure(board, white_king, black_king):
        """Enemy pieces whose attacks land next to the king. The eight ring squares are
        tested by walking outward from the ring, which is what a real term would do."""
        total = 0
        for side in range(2):
            king_sq = white_king if side == 0 else black_king
            enemy = 1 - side
            pressure = 0
            for i in range(8):
                ring = king_sq + king_steps[i]
                if not on_board(ring):
                    continue
                # pawns
                if enemy == 0:
                    a, b = ring - 15, ring - 17
                    target = 1
                else:
                    a, b = ring + 15, ring + 17
                    target = 9
                if on_board(a) and board[a] == target:
                    pressure += 1
                if on_board(b) and board[b] == target:
                    pressure += 1
                knight = 2 | (enemy << 3)
                for j in range(8):
                    to = ring + knight_steps[j]
                    if on_board(to) and board[to] == knight:
                        pressure += 1
                for j in range(4):
                    step = bishop_steps[j]
                    to = ring + step
                    while on_board(to):
                        other = board[to]
                        if other != 0:
                            kind = other & 7
                            if (other >> 3) == enemy and (kind == 3 or kind == 5):
                                pressure += 1
                            break
                        to += step
                    step = rook_steps[j]
                    to = ring + step
                    while on_board(to):
                        other = board[to]
                        if other != 0:
                            kind = other & 7
                            if (other >> 3) == enemy and (kind == 4 or kind == 5):
                                pressure += 1
                            break
                        to += step
            total += pressure if side == 0 else -pressure
        return total

    return mobility, king_zone_pressure


def bench(fn, args_list, seconds=1.0):
    fn(*args_list[0])  # compile
    started = time.perf_counter()
    calls = 0
    while time.perf_counter() - started < seconds:
        for args in args_list:
            fn(*args)
            calls += 1
    return (time.perf_counter() - started) / calls * 1e6


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent / "agent.py")
    agent = load(path)
    mobility, king_pressure = build_probes(agent)

    fens = positions(40)
    states = [agent.jit_new_state(f) for f in fens]
    eval_args = [(b, st) for b, st in states]
    mob_args = [(b,) for b, _ in states]
    king_args = [(b, int(st[agent.KING_W]), int(st[agent.KING_B])) for b, st in states]

    classical = bench(agent.jit_evaluate, eval_args)
    mob = bench(mobility, mob_args)
    king = bench(king_pressure, king_args)

    # Node cost, measured the way probe.py measures it.
    work = agent.jit_make_workspace()
    board, st = agent.jit_new_state(chess.STARTING_FEN)
    work["info"][agent.NODE_LIMIT] = 900_000
    work["info"][agent.NODES] = 0
    work["info"][agent.STOPPED] = 0
    work["info"][agent.REP_LEN] = 0
    started = time.perf_counter()
    agent.jit_search_root(
        board, st, work["hist"], work["moves"], work["scores"], work["occ"], work["info"],
        work["killers"], work["hist_heur"], work["counter"], work["rep"], work["tt_key"],
        work["tt_move"], work["tt_score"], work["tt_meta"], 9, -31000, 31000, 0, work["out"])
    spent = time.perf_counter() - started
    nodes = int(work["info"][agent.NODES])
    per_node = spent / nodes * 1e6
    nps = nodes / spent

    evals_per_node = 0.8
    print(f"\nclassical evaluation   {classical:8.3f} us per call")
    print(f"search                 {nps / 1e6:8.2f} M nodes/s ({per_node:.3f} us per node)")
    share = classical * evals_per_node / per_node * 100
    print(f"evaluation is about    {share:8.1f}% of node cost\n")

    print(f"{'term':<26}{'per eval':>11}{'vs classical':>14}{'projected nps':>15}{'verdict':>12}")
    for label, cost in (("mobility", mob),
                        ("king safety", king),
                        ("both", mob + king)):
        node = per_node + cost * evals_per_node
        projected = 1e6 / node
        ratio = projected / nps
        verdict = "viable" if ratio > 0.75 else "marginal" if ratio > 0.6 else "dead"
        print(f"{label:<26}{cost:9.2f}us{cost / classical:12.1f}x"
              f"{projected / 1e6:12.2f}M{ratio * 100:9.0f}%  {verdict}")
    print()
    print("viable  = keeps more than 75% of the node rate, so the term can pay for itself")
    print("dead    = costs more depth than a good evaluation term returns")


if __name__ == "__main__":
    main()
