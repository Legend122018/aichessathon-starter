"""Measure what a neural evaluation would cost the search, before training one.

v5 replaced the hand-written evaluation with a network and scored 8.8%. Two things went
wrong at once: the net was less accurate than what it replaced (217cp mean error against
193cp) and it cost about 40% of the node rate. The second failure is the one worth
checking first, because it is fatal on its own - a perfect evaluation that halves the
depth loses anyway - and because it can be measured without training anything.

So: time the real evaluation, time a quantised forward pass at several sizes, and work
out what each would do to nodes per second. Random weights are fine; multiply-accumulate
does not care what the numbers are.

    python probe.py ../overnight/candidate_r6.py
"""

from __future__ import annotations

import importlib.util
import random
import sys
import time

import chess
import numpy as np
from numba import njit


def load(path: str):
    spec = importlib.util.spec_from_file_location("probe_agent", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["probe_agent"] = module
    spec.loader.exec_module(module)
    if not getattr(module, "JIT_READY", False):
        raise SystemExit(f"JIT unavailable: {getattr(module, 'JIT_ERROR', '')}")
    return module


# The shape every NNUE-style net has at the leaf: the wide first layer is an accumulator
# kept up to date as pieces move, so what each evaluation actually pays for is the dense
# hidden layer and the output. int16 weights with an int32 accumulator is the standard
# quantisation and the one numba vectorises best.
@njit(cache=False)
def forward(acc, w1, b1, w2, b2, hidden, inputs):
    total = 0
    for j in range(hidden):
        s = b1[j]
        base = j * inputs
        for i in range(inputs):
            s += acc[i] * w1[base + i]
        s = s >> 6
        if s < 0:
            s = 0
        elif s > 127:
            s = 127
        total += s * w2[j]
    return (total + b2) >> 6


@njit(cache=False)
def refresh(acc, weights, squares, count, width):
    """Cost of rebuilding an accumulator from scratch, which a search pays on every
    king move and never amortises. Included so the estimate is not flattering."""
    for j in range(width):
        acc[j] = 0
    for k in range(count):
        base = squares[k] * width
        for j in range(width):
            acc[j] += weights[base + j]
    return acc[0]


def bench(fn, *args, seconds=1.2):
    fn(*args)                                   # compile
    calls = 0
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        for _ in range(200):
            fn(*args)
        calls += 200
    return (time.perf_counter() - started) / calls * 1e6      # microseconds per call


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "candidate_r6.py"
    agent = load(path)
    rng = random.Random(11)

    # ---------------------------------------------------------------- classical cost
    positions = []
    for _ in range(40):
        board = chess.Board()
        for _ in range(rng.randint(6, 50)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over():
            positions.append(board.fen())

    states = [agent.jit_new_state(fen) for fen in positions]
    agent.jit_evaluate(states[0][0], states[0][1])
    started = time.perf_counter()
    rounds = 0
    while time.perf_counter() - started < 1.5:
        for board, st in states:
            agent.jit_evaluate(board, st)
        rounds += 1
    classical = (time.perf_counter() - started) / (rounds * len(states)) * 1e6

    # ---------------------------------------------------------------- search cost
    work = agent.JIT_STATE["work"]
    board, st = agent.jit_new_state(chess.STARTING_FEN)
    work["info"][agent.NODE_LIMIT] = 3_000_000
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
    nps = nodes / spent
    per_node = spent / nodes * 1e6

    print(f"classical evaluation   {classical:8.3f} us per call")
    print(f"search                 {nps / 1e6:8.2f} M nodes/s "
          f"({per_node:.3f} us per node, {nodes:,} nodes)")

    # An evaluation is not called at every node - interior nodes that fail high on the
    # transposition table skip it - but quiescence calls it at nearly every leaf, and
    # leaves dominate. 0.8 per node is the conservative end of the usual range.
    evals_per_node = 0.8
    share = classical * evals_per_node / per_node
    print(f"evaluation is about    {share * 100:8.1f}% of node cost "
          f"(at {evals_per_node} evals per node)")
    print()

    # ---------------------------------------------------------------- net cost
    print(f"{'architecture':<24}{'per eval':>11}{'vs classical':>14}"
          f"{'projected nps':>15}{'verdict':>12}")
    for label, inputs, hidden in (("768-128-16-1", 256, 16),
                                  ("768-256-16-1", 512, 16),
                                  ("768-256-32-1", 512, 32),
                                  ("768-512-32-1", 1024, 32)):
        acc = np.random.randint(-64, 64, inputs).astype(np.int32)
        w1 = np.random.randint(-64, 64, hidden * inputs).astype(np.int16)
        b1 = np.random.randint(-64, 64, hidden).astype(np.int32)
        w2 = np.random.randint(-64, 64, hidden).astype(np.int16)
        cost = bench(forward, acc, w1, b1, w2, 0, hidden, inputs)

        wide = np.random.randint(-64, 64, 768 * inputs).astype(np.int16)
        squares = np.random.randint(0, 768, 32).astype(np.int32)
        rebuild = bench(refresh, acc, wide, squares, 32, inputs, seconds=0.6)

        # A king move forces a full rebuild; call it one node in forty.
        total = cost + rebuild / 40.0
        node = per_node - classical * evals_per_node + total * evals_per_node
        projected = 1e6 / node
        ratio = projected / nps
        verdict = "viable" if ratio > 0.75 else "marginal" if ratio > 0.6 else "dead"
        print(f"{label:<24}{total:9.2f}us{total / classical:12.1f}x"
              f"{projected / 1e6:12.2f}M{ratio * 100:9.0f}%  {verdict}")

    print()
    print("viable  = keeps more than 75% of the node rate, so accuracy can pay for it")
    print("dead    = loses more depth than any evaluation gain can return")


if __name__ == "__main__":
    main()
