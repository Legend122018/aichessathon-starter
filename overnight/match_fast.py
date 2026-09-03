"""Play two engines against each other and stop when the answer is clear.

Same statistics as match.py, but the worker pool is created once for the whole match
instead of once per batch. Each worker paid JIT compilation for both engines on every
batch before; now it pays once, and results stream back game by game so the SPRT can
stop the moment it is decided rather than at the next batch boundary.

    python match_fast.py --candidate candidate.py --champion champion.py
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import multiprocessing as mp
import os
import random
import sys
import time

import chess

PLY_CAP = 300

# Filled in once per worker process by setup().
WORKER: dict = {}


def load_agent(path: str, tag: str):
    spec = importlib.util.spec_from_file_location(f"agent_{tag}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"agent_{tag}"] = module
    spec.loader.exec_module(module)
    return module


def reset(module) -> None:
    """Clear everything that must not survive from one game to the next.

    The platform starts a fresh process per game. Reloading the module here would mean
    paying JIT compilation every game, so instead the per-game state is cleared by hand -
    which has to include the JIT workspace, not just the Python tables.
    """
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
    work = getattr(module, "JIT_STATE", {}).get("work") if hasattr(module, "JIT_STATE") else None
    if work is not None:
        for key in ("tt_key", "tt_move", "tt_score", "tt_meta", "killers", "hist_heur",
                    "counter"):
            if key in work:
                work[key][:] = 0


def pin(cores: list[int]) -> None:
    """Give this worker its own cores, on Windows as well as Linux.

    These games are timed on the wall clock, so several running at once are only
    comparable if the operating system stops moving them between cores mid-game. That
    matters more than usual on a 7950X3D, where half the cores have the large cache and
    half do not - an unpinned game can be measured on both and average the difference
    into the result.
    """
    if not cores:
        return
    if hasattr(os, "sched_setaffinity"):          # Linux, WSL
        try:
            os.sched_setaffinity(0, set(cores))
            return
        except OSError:
            pass
    try:                                          # Windows
        import psutil

        psutil.Process().cpu_affinity(list(cores))
    except Exception:
        pass                                      # unpinned: noisier, still correct


def setup(counter, candidate_path: str, champion_path: str, stride: int,
          total_cores: int, cores_per_game: int) -> None:
    """Run once per worker process: claim cores, then compile both engines once."""
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    first = (index * stride) % total_cores
    pin([(first + k) % total_cores for k in range(cores_per_game)])
    WORKER["candidate"] = load_agent(candidate_path, f"cand{index}")
    WORKER["champion"] = load_agent(champion_path, f"champ{index}")


def opening(rng: random.Random) -> chess.Board:
    """A short random opening, kept to positions that are not already decided."""
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
    """Score from white's point of view. Mirrors the platform's loss rules."""
    board = start.copy()
    clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    reset(white)
    reset(black)
    while True:
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            if outcome.winner is None:
                return 0.5
            return 1.0 if outcome.winner == chess.WHITE else 0.0
        if board.ply() >= PLY_CAP:
            return 0.5
        mover = white if board.turn == chess.WHITE else black
        started = time.monotonic()
        try:
            uci = mover.get_move(board.fen(), int(max(0, clock[board.turn])))
            move = chess.Move.from_uci(uci)
        except Exception:
            return 0.0 if board.turn == chess.WHITE else 1.0
        clock[board.turn] -= (time.monotonic() - started) * 1000.0
        if clock[board.turn] < 0:
            return 0.0 if board.turn == chess.WHITE else 1.0
        if move not in board.legal_moves:
            return 0.0 if board.turn == chess.WHITE else 1.0
        clock[board.turn] += inc_ms
        board.push(move)


def pair(job: tuple[int, int, int]) -> tuple[int, int, int]:
    """One opening, played from both sides so colour cannot skew the result."""
    seed, base_ms, inc_ms = job
    candidate = WORKER["candidate"]
    champion = WORKER["champion"]
    start = opening(random.Random(seed))
    wins = draws = losses = 0
    for candidate_is_white in (True, False):
        if candidate_is_white:
            score = play(candidate, champion, start, base_ms, inc_ms)
        else:
            score = 1.0 - play(champion, candidate, start, base_ms, inc_ms)
        if score == 1.0:
            wins += 1
        elif score == 0.0:
            losses += 1
        else:
            draws += 1
    return wins, draws, losses


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def llr(wins: int, draws: int, losses: int, elo0: float, elo1: float) -> float:
    """Log-likelihood ratio under a normal approximation to the score."""
    total = wins + draws + losses
    # Below a few dozen games the variance estimate is unusable - with no draws yet it
    # collapses toward zero and the statistic explodes. Wait rather than report nonsense.
    if total < 32:
        return 0.0
    score = (wins + 0.5 * draws) / total
    variance = (wins + 0.25 * draws) / total - score * score
    variance = max(variance, 0.01)
    s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
    return total * (s1 - s0) * (2 * score - s0 - s1) / (2 * variance)


def elo(wins: int, draws: int, losses: int) -> tuple[float, float]:
    total = wins + draws + losses
    if not total:
        return 0.0, 0.0
    score = (wins + 0.5 * draws) / total
    clamped = min(max(score, 0.5 / (total + 1)), 1 - 0.5 / (total + 1))
    rating = -400.0 * math.log10(1.0 / clamped - 1.0)
    spread = math.sqrt(max(score * (1 - score), 0.01) / total)
    low = min(max(clamped - 1.96 * spread, 0.001), 0.999)
    high = min(max(clamped + 1.96 * spread, 0.001), 0.999)
    margin = ((-400.0 * math.log10(1.0 / high - 1.0)) -
              (-400.0 * math.log10(1.0 / low - 1.0))) / 2
    return rating, margin


def report(outcome: str, wins: int, draws: int, losses: int, statistic: float) -> None:
    """One machine-readable line, so a driver script does not have to parse prose."""
    rating, margin = elo(wins, draws, losses)
    print(f"RESULT {outcome} {wins} {draws} {losses} {rating:.1f} {margin:.1f} "
          f"{statistic:.3f}", flush=True)


def jobs(seed: int, base_ms: int, inc_ms: int, limit: int):
    for index in range(limit):
        yield (seed * 1000003 + index, base_ms, inc_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--workers", type=int,
                        default=max(1, ((os.cpu_count() or 2) // 2) - 1))
    parser.add_argument("--cores-per-game", type=int, default=1)
    parser.add_argument("--smt-stride", type=int, default=2,
                        help="logical cores per physical core; 2 on a Ryzen with SMT. "
                             "Games are spread one per physical core so two of them "
                             "never share one and slow each other unpredictably.")
    parser.add_argument("--base-ms", type=int, default=8000)
    parser.add_argument("--inc-ms", type=int, default=80)
    parser.add_argument("--max-games", type=int, default=20000,
                        help="SPRT almost always stops long before this; a low "
                             "cap silently discards real improvements")
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--print-every", type=int, default=40,
                        help="games between progress lines")
    parser.add_argument("--max-minutes", type=float, default=0.0,
                        help="wall-clock cap; 0 means none. A test stopped by the clock "
                             "reports inconclusive rather than a verdict it has not earned")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    lower = math.log(args.beta / (1 - args.alpha))
    upper = math.log((1 - args.beta) / args.alpha)
    total_cores = os.cpu_count() or 2
    physical = max(1, total_cores // args.smt_stride)
    print(f"{args.workers} concurrent games, one per physical core "
          f"({physical} physical / {total_cores} logical)", flush=True)
    print(f"SPRT [{args.elo0}, {args.elo1}] bounds [{lower:.2f}, {upper:.2f}]", flush=True)
    print("compiling both engines once per worker, then playing continuously", flush=True)

    counter = mp.Value("i", 0)
    wins = draws = losses = 0
    started = time.monotonic()
    reported = 0
    verdict = 1

    pool = mp.Pool(args.workers, initializer=setup,
                   initargs=(counter, args.candidate, args.champion, args.smt_stride,
                             total_cores, args.cores_per_game))
    try:
        stream = pool.imap_unordered(
            pair, jobs(args.seed, args.base_ms, args.inc_ms, args.max_games // 2),
            chunksize=1)
        for w, d, losses_ in stream:
            wins += w
            draws += d
            losses += losses_
            total = wins + draws + losses
            if total - reported < args.print_every and total < args.max_games:
                continue
            reported = total
            statistic = llr(wins, draws, losses, args.elo0, args.elo1)
            rating, margin = elo(wins, draws, losses)
            print(f"  {total:5} games  +{wins} ={draws} -{losses}  "
                  f"elo {rating:+.0f} +/- {margin:.0f}  LLR {statistic:+.2f}  "
                  f"({(time.monotonic() - started) / 60:.0f} min)", flush=True)
            if statistic >= upper:
                print("SPRT: candidate is better - accept")
                report("accept", wins, draws, losses, statistic)
                verdict = 0
                break
            if statistic <= lower:
                print("SPRT: no improvement - reject")
                report("reject", wins, draws, losses, statistic)
                verdict = 1
                break
            if args.max_minutes and (time.monotonic() - started) / 60.0 >= args.max_minutes:
                needed = int(total * upper / statistic) if statistic > 0 else 0
                print(f"SPRT: out of time at {args.max_minutes:g} minutes, undecided"
                      + (f" - about {needed:,} games would settle it" if needed else ""))
                report("inconclusive", wins, draws, losses, statistic)
                verdict = 2
                break
        else:
            statistic = llr(wins, draws, losses, args.elo0, args.elo1)
            if statistic > 0:
                needed = int((wins + draws + losses) * upper / max(statistic, 1e-9))
                print(f"SPRT: inconclusive at the game limit. The trend is positive - "
                      f"about {needed:,} games would settle it. Raise --max-games.")
            else:
                print("SPRT: inconclusive at the game limit - reject to be safe")
            report("inconclusive", wins, draws, losses,
                   llr(wins, draws, losses, args.elo0, args.elo1))
            verdict = 2
    finally:
        pool.terminate()
        pool.join()
    sys.exit(verdict)


if __name__ == "__main__":
    main()
