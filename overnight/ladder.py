"""Play the agent against Stockfish at a series of strength settings.

Stockfish never ships and is never called at runtime; this is a measuring stick, which
the rules allow. What it produces is a curve rather than a number: the setting where the
score crosses 50% is the engine's strength on Stockfish's own scale.

    python ladder.py --agent try_ownpst.py --from-elo 2200 --to-elo 3000 --games 10

Two things this cannot do, and it is better to say them here than to imply otherwise in
the output. Ten games carry an error bar of roughly ±180 Elo, so a single row means very
little and only the shape across rows is worth reading. And Stockfish's UCI_Elo is a
self-report, not a rating earned against other engines - it is known to run optimistic at
the top of its range, so a crossing at 2600 does not mean 2600 on a published list.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import shutil
import sys
import time

import chess
import chess.engine


def load(path: str):
    spec = importlib.util.spec_from_file_location("ladder_agent", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["ladder_agent"] = module
    spec.loader.exec_module(module)
    if not getattr(module, "JIT_READY", False):
        print(f"warning: {path} is running without its JIT", flush=True)
    return module


def reset(module) -> None:
    stop = getattr(module, "_ponder_stop", None)
    if stop is not None:
        stop()
    for name in ("SEEN", "TT", "HISTORY", "COUNTER"):
        holder = getattr(module, name, None)
        if isinstance(holder, dict):
            holder.clear()
    fens = getattr(module, "HISTORY_FENS", None)
    if isinstance(fens, list):
        del fens[:]
    work = getattr(module, "JIT_STATE", {}).get("work") \
        if hasattr(module, "JIT_STATE") else None
    if work is not None:
        for key in ("tt_key", "tt_move", "tt_score", "tt_meta", "killers", "hist_heur",
                    "counter"):
            if key in work:
                work[key][:] = 0


def opening(rng: random.Random) -> chess.Board:
    for _ in range(60):
        board = chess.Board()
        for _ in range(rng.randint(4, 8)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over():
            return board
    return chess.Board()


def game(module, engine, start: chess.Board, agent_white: bool,
         base_ms: int, inc_ms: int) -> tuple[float, str]:
    """One game. Returns the agent's score and how it ended."""
    board = start.copy()
    clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    reset(module)
    agent_colour = chess.WHITE if agent_white else chess.BLACK

    while True:
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            if outcome.winner is None:
                return 0.5, "draw"
            return (1.0, "win") if outcome.winner == agent_colour else (0.0, "loss")
        if board.ply() >= 300:
            return 0.5, "adjudicated"

        turn = board.turn
        started = time.monotonic()
        if turn == agent_colour:
            try:
                uci = module.get_move(board.fen(), int(max(0, clock[turn])))
                move = chess.Move.from_uci(uci)
            except Exception:
                return 0.0, "agent crashed"
            if move not in board.legal_moves:
                return 0.0, "agent illegal"
        else:
            limit = chess.engine.Limit(
                white_clock=clock[chess.WHITE] / 1000.0,
                black_clock=clock[chess.BLACK] / 1000.0,
                white_inc=inc_ms / 1000.0, black_inc=inc_ms / 1000.0)
            move = engine.play(board, limit).move
            if move is None:
                return 1.0, "engine resigned"

        clock[turn] -= (time.monotonic() - started) * 1000.0
        if clock[turn] < 0:
            return (0.0, "agent flagged") if turn == agent_colour else (1.0, "engine flagged")
        clock[turn] += inc_ms
        board.push(move)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--stockfish", default="/tmp/stockfish/stockfish-ubuntu-x86-64-avx2")
    parser.add_argument("--from-elo", type=int, default=2200)
    parser.add_argument("--to-elo", type=int, default=3000)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--base-ms", type=int, default=8000)
    parser.add_argument("--inc-ms", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.stockfish):
        raise SystemExit(f"no Stockfish at {args.stockfish}")

    module = load(args.agent)
    # One core each, so neither side is measuring the other's scheduling.
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {0})
        except OSError:
            pass
    command = [args.stockfish]
    if shutil.which("taskset"):
        command = ["taskset", "-c", "1", args.stockfish]

    print(f"{args.agent} vs Stockfish, {args.games} games per level, "
          f"{args.base_ms / 1000:g}s + {args.inc_ms / 1000:g}s, one core each\n", flush=True)
    print(f"{'SF UCI_Elo':>11}{'W':>4}{'D':>4}{'L':>4}{'score':>8}{'win rate':>10}"
          f"{'implied gap':>13}{'min':>6}")

    rows = []
    for target in range(args.from_elo, args.to_elo + 1, args.step):
        engine = chess.engine.SimpleEngine.popen_uci(command)
        engine.configure({"Threads": 1, "Hash": 64,
                          "UCI_LimitStrength": True, "UCI_Elo": target})
        rng = random.Random(args.seed + target)
        wins = draws = losses = 0
        endings: dict[str, int] = {}
        started = time.monotonic()
        for index in range(args.games):
            start = opening(rng)
            score, how = game(module, engine, start, index % 2 == 0,
                              args.base_ms, args.inc_ms)
            endings[how] = endings.get(how, 0) + 1
            if score == 1.0:
                wins += 1
            elif score == 0.0:
                losses += 1
            else:
                draws += 1
        engine.quit()

        total = wins + draws + losses
        score = (wins + 0.5 * draws) / total
        win_rate = wins / total
        clamped = min(max(score, 0.01), 0.99)
        gap = -400.0 * __import__("math").log10(1.0 / clamped - 1.0)
        minutes = (time.monotonic() - started) / 60.0
        rows.append((target, wins, draws, losses, score, win_rate, gap))
        print(f"{target:>11}{wins:>4}{draws:>4}{losses:>4}{score:>8.2f}"
              f"{win_rate * 100:>9.0f}%{gap:>+12.0f}{minutes:>6.0f}", flush=True)
        odd = {k: v for k, v in endings.items()
               if k not in ("win", "loss", "draw", "adjudicated")}
        if odd:
            print(f"            {odd}", flush=True)

    print("\nscore counts draws as a half; win rate does not.")
    print("implied gap is the agent's rating minus Stockfish's setting, from the score.")
    print("Ten games is +/- about 180 Elo per row: read the curve, not the rows.")


if __name__ == "__main__":
    main()
