"""Tune the engine's search parameters without playing thousands of games.

A match is the honest way to measure a change, and it is also the reason nothing gets
tuned: a term worth +10 Elo needs about 2,000 games, which is more than a day on this
hardware. So this measures move quality directly instead. The engine picks a move on each
position of a suite under a fixed clock, and Stockfish says what that move cost against
the best one available. The average of that loss, in centipawns, is the score - lower is
better, and it moves for reasons a match would take hours to see.

It is a proxy, not a verdict. Anything it likes still has to win a real match before it
ships; what it buys is knowing which few candidates are worth a match at all.

    python overnight/tune_search.py --sweep          # one parameter at a time
    python overnight/tune_search.py --baseline       # just score the defaults
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXEMPLAR = HERE / "exemplar"
STOCKFISH = HERE / "stockfish-windows-x86-64-avx2.exe"

# Values to try for each numeric parameter, around the shipped default.
SWEEP = {
    "RFP_MARGIN": [55, 65, 85, 95],
    "RAZOR_MARGIN": [280, 310, 370, 400],
    "FUT_MARGIN": [90, 100, 120, 130],
    "FUT_BASE": [70, 80, 100, 110],
    "NMP_BASE": [3, 5],
    "NMP_DIV": [3, 5, 6],
    "NMP_EVAL_DIV": [140, 160, 200, 220],
    "ASP_WINDOW": [8, 10, 16, 20],
    "DELTA_MARGIN": [100, 115, 145, 160],
    "SEE_QUIET": [-75, -65, -45, -35],
    "SEE_CAP": [-135, -120, -90, -75],
    "SING_DEPTH": [5, 6, 8, 9],
    "SING_MARGIN": [1, 3, 4],
    "HIST_PRUNE_MUL": [-4200, -3700, -2700, -2200],
    "LMR_HIST_DIV": [4000, 5000, 7000, 8000],
    "RFP_DEPTH": [6, 7, 9, 10],
    "FUT_DEPTH": [6, 7, 9, 10],
}


def load_exemplar():
    sys.path.insert(0, str(EXEMPLAR))
    spec = importlib.util.spec_from_file_location("tsx_agent", str(EXEMPLAR / "agent.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["tsx_agent"] = module
    started = time.monotonic()
    spec.loader.exec_module(module)
    print(f"engine compiled in {time.monotonic() - started:.1f}s", flush=True)
    return module


def suite(count: int) -> list[str]:
    """Positions from self-play, past the book and short of the trivial endgame."""
    rng = random.Random(5)
    out: list[str] = []
    while len(out) < count:
        board = chess.Board()
        for _ in range(rng.randint(16, 50)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if board.is_game_over():
                break
        if not board.is_game_over() and chess.popcount(board.occupied) > 8:
            out.append(board.fen())
    return out


class Judge:
    """Stockfish, scoring how much a move cost. Answers are cached: the same position
    comes back constantly across candidates, and each probe is the expensive part."""

    def __init__(self, engine, depth: int):
        self.engine = engine
        # Fixed depth, not fixed time. A time-limited judge gives a different verdict on
        # the same position from one run to the next, which put roughly a centipawn and a
        # half of noise into every score - as much as the parameter effects being hunted.
        self.limit = chess.engine.Limit(depth=depth)
        self.cache: dict[str, int] = {}

    def _score(self, board: chess.Board) -> int:
        key = board.fen()
        if key not in self.cache:
            info = self.engine.analyse(board, self.limit)
            self.cache[key] = info["score"].pov(board.turn).score(mate_score=30000)
        return self.cache[key]

    def loss(self, fen: str, uci: str) -> float:
        """Centipawns given up by playing `uci` instead of the best move."""
        board = chess.Board(fen)
        best = self._score(board)
        board.push(chess.Move.from_uci(uci))
        after = -self._score(board)      # flip: it is now the opponent's turn
        return max(0.0, best - after)


def score_config(agent, judge: Judge, fens: list[str], params: dict[str, int],
                 think_ms: int) -> float:
    engine = agent._STATE.engine
    for name, value in params.items():
        engine.set_param(name, value)
    total = 0.0
    for fen in fens:
        engine.new_game()
        engine.set_position(fen)
        # Fixed nodes, not fixed time. A time limit makes the search nondeterministic -
        # the same configuration scored 20.7 to 23.8 cp across four identical runs, which
        # is larger than any parameter effect being looked for. Counting nodes instead
        # makes a rerun reproduce exactly, so a difference is the parameter and not the
        # scheduler.
        info = engine.search(max_depth=64, nodes=think_ms * 400)
        total += judge.loss(fen, info.bestmove)
    return total / len(fens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=40)
    parser.add_argument("--think-ms", type=int, default=500,
                        help="node budget per move is this times 400")
    parser.add_argument("--judge-depth", type=int, default=12)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--out", default=str(HERE / "REPORT_search_tuning.md"))
    args = parser.parse_args()

    agent = load_exemplar()
    fens = suite(args.positions)
    sf = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
    sf.configure({"Threads": 1, "Hash": 128})
    judge = Judge(sf, args.judge_depth)

    defaults = {name: agent._STATE.engine.get_param(name) for name in SWEEP}
    print(f"{len(fens)} positions, {args.think_ms * 400:,} nodes per move, "
          f"judged by Stockfish at depth {args.judge_depth}\n", flush=True)

    started = time.monotonic()
    base = score_config(agent, judge, fens, defaults, args.think_ms)
    print(f"baseline (shipped defaults): {base:.1f} cp average loss "
          f"[{time.monotonic() - started:.0f}s]\n", flush=True)

    if args.baseline:
        sf.quit()
        return

    lines = ["# Search parameter sweep\n",
             f"Average centipawn loss against Stockfish over {len(fens)} positions at "
             f"{args.think_ms}ms per move. Lower is better; the baseline is the shipped "
             f"configuration.\n",
             f"**baseline {base:.1f} cp**\n",
             "| parameter | default | value | cp loss | change |",
             "| --- | --- | --- | --- | --- |"]
    wins: list[tuple[float, str, int]] = []

    for name, values in SWEEP.items():
        for value in values:
            trial = dict(defaults)
            trial[name] = value
            got = score_config(agent, judge, fens, trial, args.think_ms)
            delta = got - base
            mark = "better" if delta < -0.5 else ("worse" if delta > 0.5 else "level")
            print(f"  {name:<16} {value:>7}  {got:7.1f} cp  {delta:+6.1f}  {mark}", flush=True)
            lines.append(f"| {name} | {defaults[name]} | {value} | {got:.1f} | "
                         f"{delta:+.1f} {mark} |")
            if delta < -0.5:
                wins.append((delta, name, value))

    wins.sort()
    lines.append("\n## Worth a match\n")
    if wins:
        for delta, name, value in wins[:8]:
            lines.append(f"- `{name} = {value}` ({delta:+.1f} cp)")
        lines.append("\nThese are proxy results. Each still has to win a real match "
                     "before it ships.")
    else:
        lines.append("Nothing beat the defaults by more than noise. The shipped "
                     "configuration looks well chosen.")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    sf.quit()


if __name__ == "__main__":
    main()
