"""Generate Stockfish-labelled training positions across every core.

The rules allow unrestricted training data, including positions analysed by an existing
engine; what they forbid is an engine choosing moves at run time. Nothing here ships.

    python gen.py --out data --games 40000 --workers 16
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import random
import shutil
import time

import chess
import chess.engine

CLAMP = 2000


def find_stockfish() -> str:
    """Locate Stockfish on Windows or Linux, including a copy sitting next to this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    # Accept whatever name the official Windows build arrives under - it ships as
    # stockfish-windows-x86-64-avx2.exe and renaming it is a common way to get a file
    # called stockfish.exe.exe instead.
    candidates = [
        "stockfish", "stockfish.exe",
        *sorted(glob.glob(os.path.join(here, "stockfish*.exe"))),
        *sorted(glob.glob(os.path.join(here, "stockfish*"))),
        "/usr/games/stockfish", "/usr/bin/stockfish",
    ]
    for candidate in candidates:
        found = shutil.which(candidate) or (candidate if os.path.exists(candidate) else None)
        if found:
            return found
    raise SystemExit(
        "Stockfish not found.\n\n"
        "  Windows: download from https://stockfishchess.org/download/ and put the\n"
        f"           .exe (any stockfish* name is fine) in {here}\n"
        "  Linux:   sudo apt install stockfish\n\n"
        "It is used only to label training positions and to measure strength. It is "
        "never imported by the agent and never goes in the submission."
    )


def worker(index: int, games: int, depth: int, out_dir: str, seed: int,
           run_id: str) -> int:
    rng = random.Random(seed)
    engine = chess.engine.SimpleEngine.popen_uci(find_stockfish())
    engine.configure({"Threads": 1, "Hash": 64})
    limit = chess.engine.Limit(depth=depth)
    # The run id keeps each round's output separate. Without it every round would
    # reuse the same filenames and quietly delete the data collected before it.
    path = os.path.join(out_dir, f"part_{run_id}_{index:02d}.csv")
    written = 0
    with open(path, "w") as handle:
        for game in range(games):
            board = chess.Board()
            for _ in range(rng.randint(6, 12)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
            if board.is_game_over():
                continue
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=6))
                opening = info["score"].relative.score(mate_score=10000)
            except chess.engine.EngineError:
                continue
            if opening is None or abs(opening) > 500:
                continue

            records = []
            while not board.is_game_over(claim_draw=True) and board.ply() < 220:
                try:
                    info = engine.analyse(board, limit)
                except chess.engine.EngineError:
                    break
                score = info["score"].relative.score(mate_score=10000)
                best = info.get("pv", [None])[0]
                if best is None or score is None:
                    break
                # Quiet positions only: a score taken mid-exchange or in check teaches the
                # evaluation about positions it should never be asked to judge.
                if not board.is_check() and not board.is_capture(best):
                    records.append((board.fen(), max(-CLAMP, min(CLAMP, score)), board.turn))
                board.push(best)

            outcome = board.outcome(claim_draw=True)
            winner = outcome.winner if outcome else None
            for fen, score, turn in records:
                result = 0.5 if winner is None else (1.0 if winner == turn else 0.0)
                handle.write(f"{fen},{score},{result}\n")
                written += 1
            if game % 50 == 0:
                handle.flush()
                if index == 0:
                    print(f"  worker 0: game {game}/{games}, {written} positions",
                          flush=True)
    engine.quit()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data")
    parser.add_argument("--games", type=int, default=40000, help="total across all workers")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="data generation is pure throughput, so it is worth "
                             "using the SMT threads as well as the real cores")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_id = time.strftime("%m%d%H%M%S")
    per_worker = max(1, args.games // args.workers)
    print(f"{args.workers} workers x {per_worker} games -> {args.out}", flush=True)
    started = time.monotonic()
    with mp.Pool(args.workers) as pool:
        results = pool.starmap(
            worker,
            [(i, per_worker, args.depth, args.out, args.seed * 1000 + i, run_id)
             for i in range(args.workers)],
        )
    total = sum(results)
    existing = sum(1 for _ in glob.glob(os.path.join(args.out, "part*.csv")))
    elapsed = time.monotonic() - started
    print(f"{total:,} new positions in {elapsed / 60:.1f} min "
          f"({total / max(elapsed, 1):.0f}/s), {existing} data files total", flush=True)


if __name__ == "__main__":
    main()
