"""Tune the engine's search parameters without playing thousands of games.

A match is the honest way to measure a change, and it is also the reason nothing gets
tuned: a term worth +10 Elo needs about 2,000 games, which is more than a day on this
hardware. So this measures move quality directly instead. The engine picks a move on each
position of a suite under a fixed node count, and Stockfish says what that move cost
against the best one available. The average of that loss, in centipawns, is the score -
lower is better, and it moves for reasons a match would take hours to see.

Two rounds of this measured nothing but their own noise, so both causes are now closed:

  * The search ran on a clock and the judge ran on a clock, so neither was reproducible;
    four identical runs scored 23.1, 20.7, 22.6 and 23.8 cp. Fixed nodes and fixed depth
    make a rerun reproduce exactly.
  * Even reproducible, a mean over N positions carries sampling error, and one position
    swinging from 0 to 300 cp moves an 80 position mean by 3.75 cp - larger than any
    effect worth having. So nothing is compared as two means any more. Every candidate
    runs the same positions in the same order as the baseline, and the score is the
    average of the per-position differences, which cancels the positions themselves and
    leaves an error bar you can test against.

It is still a proxy, not a verdict. Anything it likes has to win a real match before it
ships; what it buys is knowing which few candidates are worth a match at all.

    python overnight/tune_search.py --sweep                     # one parameter at a time
    python overnight/tune_search.py --candidates RFP_MARGIN=85  # re-test a shortlist
    python overnight/tune_search.py --baseline                  # just score the defaults
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import statistics
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


def run_config(agent, judge: Judge, fens: list[str], params: dict[str, int],
               think_ms: int) -> tuple[list[float], list[str]]:
    """Per-position loss and chosen move. The losses come back one per position rather
    than averaged, because a candidate is judged against the baseline position by
    position - see the module docstring."""
    engine = agent._STATE.engine
    for name, value in params.items():
        engine.set_param(name, value)
    losses, moves = [], []
    for fen in fens:
        engine.new_game()
        engine.set_position(fen)
        # Fixed nodes, not fixed time: a time limit makes the search nondeterministic, so
        # a rerun of the same configuration lands somewhere else and the difference gets
        # read as the parameter.
        info = engine.search(max_depth=64, nodes=think_ms * 400)
        losses.append(judge.loss(fen, info.bestmove))
        moves.append(info.bestmove)
    return losses, moves


def compare(base: list[float], trial: list[float]) -> tuple[float, float]:
    """Mean paired difference and its standard error, in centipawns.

    Pairing is the whole point. The positions are shared, so subtracting position by
    position removes the suite's own variance - which is enormous, since most moves lose
    nothing and a few lose a queen - and leaves only what the parameter changed. Positions
    where the engine played the same move contribute an exact zero and shrink the error
    bar honestly, rather than adding noise to both sides of a two-mean comparison.
    """
    diffs = [t - b for b, t in zip(base, trial, strict=True)]
    mean = statistics.fmean(diffs)
    if len(diffs) < 2:
        return mean, float("inf")
    return mean, statistics.stdev(diffs) / len(diffs) ** 0.5


def verdict(delta: float, se: float) -> str:
    """Two standard errors, both ways. Anything inside is 'level' - not 'slightly better'.
    Calling a sub-noise wobble a finding is how the first two sweeps produced 33 winners
    out of 64 for an engine that was already tuned."""
    if delta + 2 * se < 0:
        return "BETTER"
    if delta - 2 * se > 0:
        return "worse"
    return "level"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=200)
    parser.add_argument("--think-ms", type=int, default=500,
                        help="node budget per move is this times 400")
    parser.add_argument("--judge-depth", type=int, default=12)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--candidates", default="",
                        help="comma separated NAME=VALUE to re-test at higher precision")
    parser.add_argument("--out", default=str(HERE / "REPORT_search_tuning.md"))
    args = parser.parse_args()

    agent = load_exemplar()
    fens = suite(args.positions)
    sf = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
    sf.configure({"Threads": 1, "Hash": 128})
    judge = Judge(sf, args.judge_depth)

    defaults = {name: agent._STATE.engine.get_param(name) for name in SWEEP}
    print(f"{len(fens)} positions, {args.think_ms * 400:,} nodes per move, "
          f"judged by Stockfish at depth {args.judge_depth}", flush=True)
    print("delta is the mean paired difference against the baseline, +-2 standard "
          "errors\n", flush=True)

    started = time.monotonic()
    base, base_moves = run_config(agent, judge, fens, defaults, args.think_ms)
    base_mean = statistics.fmean(base)
    per_config = time.monotonic() - started
    print(f"baseline (shipped defaults): {base_mean:.1f} cp average loss "
          f"[{per_config:.0f}s]\n", flush=True)

    if args.baseline:
        sf.quit()
        return

    if args.candidates:
        trials = []
        for item in args.candidates.split(","):
            name, _, value = item.partition("=")
            trials.append((name.strip(), int(value)))
    else:
        trials = [(n, v) for n, values in SWEEP.items() for v in values]
    print(f"{len(trials)} configurations to try, about "
          f"{len(trials) * per_config / 60:.0f} min\n", flush=True)

    lines = ["# Search parameter sweep\n",
             f"Average centipawn loss against Stockfish over {len(fens)} positions at "
             f"{args.think_ms * 400:,} nodes per move. Lower is better. `delta` is the "
             "mean paired difference from the shipped configuration, and a candidate "
             "only counts if it clears two standard errors.\n",
             f"**baseline {base_mean:.1f} cp**\n",
             "| parameter | default | value | cp loss | delta | moves changed | verdict |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    wins: list[tuple[float, str, int, float]] = []

    for name, value in trials:
        trial = dict(defaults)
        trial[name] = value
        losses, moves = run_config(agent, judge, fens, trial, args.think_ms)
        delta, se = compare(base, losses)
        changed = sum(1 for a, b in zip(base_moves, moves, strict=True) if a != b)
        mark = verdict(delta, se)
        print(f"  {name:<16}{value:>7}  {statistics.fmean(losses):6.1f} cp  "
              f"{delta:+6.1f} +-{2 * se:4.1f}  {changed:>3}/{len(fens)} moves  {mark}",
              flush=True)
        lines.append(f"| {name} | {defaults[name]} | {value} | "
                     f"{statistics.fmean(losses):.1f} | {delta:+.1f} ±{2 * se:.1f} | "
                     f"{changed}/{len(fens)} | {mark} |")
        if mark == "BETTER":
            wins.append((delta, name, value, se))

    wins.sort()
    lines.append("\n## Worth a match\n")
    if wins:
        for delta, name, value, se in wins[:8]:
            lines.append(f"- `{name} = {value}` ({delta:+.1f} ±{2 * se:.1f} cp)")
        lines.append("\nThese are proxy results on one position suite. Each still has to "
                     "win a real match before it ships.")
    else:
        lines.append("Nothing cleared two standard errors. On this evidence the shipped "
                     "configuration is already at least as good as every neighbour "
                     "tried, and the search is not where the remaining Elo is.")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    sf.quit()


if __name__ == "__main__":
    main()
