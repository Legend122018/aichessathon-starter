"""Play two engines the way the platform does: one process each, one core each.

`match_fast.py` loads both engines into a single process pinned to a single core, and
alternates calling them. That is fine for measuring changes to the search, because both
sides are affected identically. It is useless for measuring anything that uses the time
between our moves - pondering above all - because the background thread would run on the
core the opponent is thinking on, and we would win partly by slowing them down. That
advantage does not exist on the platform and a match that reports it is lying.

So this mirrors the real thing. Each agent gets its own process and its own core, the
referee lives in the parent and holds the clock, and an agent that keeps working after
returning a move is using time that genuinely belongs to it.

    python arena2.py --candidate try_ponder.py --champion try_ownpst.py --games 200

Two other fidelity fixes come along for the ride: a game that reaches 300 plies is
adjudicated on material rather than called a draw, which is what the rules say, and the
per-move exchange is one request and one reply, which is the shape of the real protocol.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import multiprocessing as mp
import os
import pathlib
import random
import sys
import threading
import time
from pathlib import Path

import chess

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from match_fast import elo, llr, pin  # noqa: E402

PLY_CAP = 300
# Material at adjudication, in the usual units. The king is worth nothing here because
# both sides always have one.
PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
               chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material(board: chess.Board) -> int:
    total = 0
    for _square, piece in board.piece_map().items():
        value = PIECE_VALUE[piece.piece_type]
        total += value if piece.color == chess.WHITE else -value
    return total


def adjudicate(board: chess.Board) -> float:
    """300 plies without a result: material decides, else draw."""
    balance = material(board)
    if balance > 0:
        return 1.0
    if balance < 0:
        return 0.0
    return 0.5


def engine_process(conn, path: str, cores: list[int]) -> None:
    """One agent, one core, for the life of the run.

    The module is imported once and serves every game, which is not what the platform
    does - it starts a fresh process per game - but paying numba's compile cost for each
    of several thousand games would make the run impossible. The per-game state is
    cleared by hand instead, which is the same thing the old harness did.
    """
    pin(cores)
    # The platform puts the zip root first on sys.path, so an agent that ships a package
    # beside it imports cleanly. Do the same here or such an agent cannot be measured.
    root = str(pathlib.Path(path).resolve().parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("arena_agent", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["arena_agent"] = module
    spec.loader.exec_module(module)
    # None means "the engine does not use this convention", which is not a failure.
    flag = getattr(module, "JIT_READY", None)
    conn.send(("ready", None if flag is None else bool(flag)))

    while True:
        message = conn.recv()
        if message[0] == "stop":
            # Let any background search finish before the process goes away.
            for name in ("_ponder_stop", "_stop_ponder"):
                stop = getattr(module, name, None)
                if callable(stop):
                    stop()
            return
        if message[0] == "reset":
            for name in ("_ponder_stop", "_stop_ponder"):
                stop = getattr(module, name, None)
                if callable(stop):
                    stop()
            reset_hook = getattr(module, "new_game", None) or getattr(module, "reset", None)
            if callable(reset_hook):
                with contextlib.suppress(Exception):
                    reset_hook()
            for name in ("SEEN", "TT", "HISTORY", "COUNTER"):
                holder = getattr(module, name, None)
                if isinstance(holder, dict):
                    holder.clear()
            fens = getattr(module, "HISTORY_FENS", None)
            if isinstance(fens, list):
                del fens[:]
            killers = getattr(module, "KILLERS", None)
            if isinstance(killers, list):
                for slot in killers:
                    slot[0] = slot[1] = None
            work = getattr(module, "JIT_STATE", {}).get("work") \
                if hasattr(module, "JIT_STATE") else None
            if work is not None:
                for key in ("tt_key", "tt_move", "tt_score", "tt_meta", "killers",
                            "hist_heur", "counter"):
                    if key in work:
                        work[key][:] = 0
            conn.send(("ok",))
            continue
        if message[0] == "move":
            _, fen, left = message
            try:
                conn.send(("move", module.get_move(fen, left)))
            except Exception as exc:
                conn.send(("error", repr(exc)[:200]))


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


def play(white, black, start: chess.Board, base_ms: int, inc_ms: int) -> float:
    """One game. Score from white's point of view, with the platform's loss rules."""
    board = start.copy()
    clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    for side in (white, black):
        side.send(("reset",))
    for side in (white, black):
        side.recv()

    while True:
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            if outcome.winner is None:
                return 0.5
            return 1.0 if outcome.winner == chess.WHITE else 0.0
        if board.ply() >= PLY_CAP:
            return adjudicate(board)

        mover = white if board.turn == chess.WHITE else black
        started = time.monotonic()
        mover.send(("move", board.fen(), int(max(0, clock[board.turn]))))
        reply = mover.recv()
        clock[board.turn] -= (time.monotonic() - started) * 1000.0
        if reply[0] != "move":
            return 0.0 if board.turn == chess.WHITE else 1.0
        if clock[board.turn] < 0:
            return 0.0 if board.turn == chess.WHITE else 1.0
        try:
            move = chess.Move.from_uci(reply[1])
        except ValueError:
            return 0.0 if board.turn == chess.WHITE else 1.0
        if move not in board.legal_moves:
            return 0.0 if board.turn == chess.WHITE else 1.0
        clock[board.turn] += inc_ms
        board.push(move)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--pairs", type=int, default=0,
                        help="concurrent games; each needs two cores. Default fills the "
                             "physical cores and leaves one pair's worth for the referee")
    parser.add_argument("--base-ms", type=int, default=20000)
    parser.add_argument("--inc-ms", type=int, default=200)
    parser.add_argument("--elo0", type=float, default=-5.0)
    parser.add_argument("--elo1", type=float, default=15.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--max-games", type=int, default=20000)
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--smt-stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    total_cores = os.cpu_count() or 2
    physical = max(1, total_cores // args.smt_stride)
    pairs = args.pairs or max(1, (physical - 1) // 2)

    lower = 0.0 if args.alpha >= 1 else __import__("math").log(args.beta / (1 - args.alpha))
    upper = __import__("math").log((1 - args.beta) / args.alpha)

    print(f"{pairs} concurrent games, two processes each, one core per agent "
          f"({physical} physical cores)", flush=True)
    print(f"SPRT [{args.elo0:g}, {args.elo1:g}] bounds [{lower:.2f}, {upper:.2f}] at "
          f"{args.base_ms / 1000:g}s + {args.inc_ms / 1000:g}s", flush=True)

    slots = []
    for index in range(pairs):
        first = (2 * index * args.smt_stride) % total_cores
        second = ((2 * index + 1) * args.smt_stride) % total_cores
        pair = []
        for path, core in ((args.candidate, first), (args.champion, second)):
            parent_end, child_end = mp.Pipe()
            proc = mp.Process(target=engine_process, args=(child_end, path, [core]),
                              daemon=True)
            proc.start()
            pair.append((parent_end, proc))
        slots.append(pair)

    for pair in slots:
        for conn, _ in pair:
            _tag, ready = conn.recv()
            if ready is None:
                # No readiness flag at all. That is not a problem, it is a different
                # engine: JIT_READY is this repo's own convention and the exemplar warms
                # up at import without setting it. Warning here said the JIT had failed
                # when it had not, on every process of every run.
                print("note: engine exposes no JIT_READY flag; warmed at import",
                      flush=True)
            elif not ready:
                print("warning: an engine came up without its JIT", flush=True)
    print("engines up, playing", flush=True)

    rng = random.Random(args.seed)
    wins = draws = losses = 0
    started = time.monotonic()
    verdict = 2

    def play_slot(pair, start, out: list) -> None:
        """One slot's pair of games: the same opening from both colours."""
        (cand, _), (champ, _) = pair
        out.append(play(cand, champ, start, args.base_ms, args.inc_ms))
        out.append(1.0 - play(champ, cand, start, args.base_ms, args.inc_ms))

    try:
        while wins + draws + losses < args.max_games:
            # The slots run at the same time. They did not before: `play` blocks on a
            # pipe until the engine answers, so iterating over the slots played them one
            # after another and left twelve of the fourteen processes idle. The header
            # has always said "7 concurrent games" and it has never been true, which is
            # most of why a match here costs hours and why nothing in this engine has
            # ever been measured to better than about +-50 Elo.
            #
            # Threads rather than processes because the referee does nothing but wait:
            # every recv releases the GIL, and the actual work is in the child
            # processes, each already pinned to its own core.
            batch = [(pair, opening(rng), []) for pair in slots]
            threads = [threading.Thread(target=play_slot, args=(pair, start, out))
                       for pair, start, out in batch]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            for _pair, _start, out in batch:
                for score in out:
                    if score == 1.0:
                        wins += 1
                    elif score == 0.0:
                        losses += 1
                    else:
                        draws += 1

            total = wins + draws + losses
            statistic = llr(wins, draws, losses, args.elo0, args.elo1)
            rating, margin = elo(wins, draws, losses)
            minutes = (time.monotonic() - started) / 60.0
            print(f"  {total:5} games  +{wins} ={draws} -{losses}  "
                  f"elo {rating:+.0f} +/- {margin:.0f}  LLR {statistic:+.2f}  "
                  f"({minutes:.0f} min)", flush=True)
            if statistic >= upper:
                print("SPRT: candidate is better - accept")
                verdict = 0
                break
            if statistic <= lower:
                print("SPRT: no improvement - reject")
                verdict = 1
                break
            if args.max_minutes and minutes >= args.max_minutes:
                print(f"SPRT: out of time at {args.max_minutes:g} minutes, undecided")
                break
    finally:
        for pair in slots:
            for conn, proc in pair:
                with contextlib.suppress(Exception):
                    conn.send(("stop",))
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.terminate()
    sys.exit(verdict)


if __name__ == "__main__":
    main()
