"""Build an opening book for the exemplar with local Stockfish.

The entry has no book: `get_move` goes straight from the tablebase probe into the search,
so it calculates its own opening moves from move one. That costs twice. The moves are
worse than known theory - three seconds of search against an opponent playing thirty years
of it - and, more importantly, they cost time. The six rated games all spend about three
seconds a move until roughly move 32 and then have nothing left, playing the remaining 8
to 46 moves without searching at all. Every move answered from a book is answered in
microseconds, so a twelve-ply book hands roughly 30-40 seconds of a 120 second game back
to the middlegame, which is where those games were actually decided.

The book is built here rather than downloaded: Lichess's explorer now needs a token, and
generating it locally means every move in it is one this machine verified at depth, with
no question about provenance or redistribution.

Shape of the tree. We only need a move for positions where we are to move, but we have to
guess our way there through the opponent's choices, so the two sides are treated
differently: at our nodes Stockfish's best move is played and stored, and at opponent
nodes every move within `--margin` of best (up to `--multipv` of them) becomes a child.
That is done once with us as White and once as Black. The walk is breadth-first and
capped by wall clock, so stopping early yields a complete shallow book rather than one
very deep line.

    python overnight/build_book.py --minutes 60 --out overnight/book.npz
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import chess
import chess.engine
import chess.polyglot
import numpy as np

HERE = Path(__file__).resolve().parent
STOCKFISH = HERE / "stockfish-windows-x86-64-avx2.exe"


def analyse(sf, board: chess.Board, depth: int, multipv: int):
    """Top moves with their scores, best first."""
    infos = sf.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    out = []
    for info in infos:
        pv = info.get("pv")
        if not pv:
            continue
        out.append((pv[0], info["score"].pov(board.turn).score(mate_score=30000)))
    return out


def build(sf, our_color: bool, book: dict[int, str], deadline: float,
          depth: int, multipv: int, margin: int, max_plies: int, stats: dict) -> None:
    """Breadth-first walk of the opening tree from the side `our_color` plays.

    Breadth-first matters: the wall-clock cap can fire at any moment, and a book that is
    complete to ply 8 is far more useful than one line explored to ply 20 - the first
    covers most games, the second covers one.
    """
    root = chess.Board()
    queue = collections.deque([(root, 0)])
    seen: set[int] = set()

    while queue and time.monotonic() < deadline:
        board, plies = queue.popleft()
        if plies >= max_plies or board.is_game_over():
            continue
        key = chess.polyglot.zobrist_hash(board)
        if key in seen:
            continue            # openings transpose constantly; never pay twice
        seen.add(key)

        if board.turn == our_color:
            # Our move: one line, the best one. Store it - this is a position the
            # engine will actually be asked about.
            moves = analyse(sf, board, depth, 1)
            if not moves:
                continue
            move, score = moves[0]
            if key not in book:
                book[key] = move.uci()
                stats["stored"] += 1
            child = board.copy(stack=False)
            child.push(move)
            queue.append((child, plies + 1))
        else:
            # Opponent's move: branch over everything plausible, so the book still
            # answers when they deviate. Nothing is stored here - we are not to move.
            moves = analyse(sf, board, depth, multipv)
            if not moves:
                continue
            best = moves[0][1]
            for move, score in moves:
                if best - score > margin:
                    break       # already sorted, so everything after is worse too
                child = board.copy(stack=False)
                child.push(move)
                queue.append((child, plies + 1))
        stats["nodes"] += 1
        if stats["nodes"] % 50 == 0:
            left = max(0.0, deadline - time.monotonic())
            print(f"  {stats['nodes']:>6} analysed  {stats['stored']:>6} positions "
                  f"booked  ply {plies:>2}  queue {len(queue):>6}  "
                  f"{left / 60:.0f} min left", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument("--depth", type=int, default=22)
    parser.add_argument("--multipv", type=int, default=4,
                        help="how many opponent replies to branch over")
    parser.add_argument("--margin", type=int, default=75,
                        help="centipawns from best an opponent move may be and still "
                             "be considered plausible enough to prepare for")
    parser.add_argument("--max-plies", type=int, default=16)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--hash", type=int, default=2048)
    parser.add_argument("--out", default=str(HERE / "book.npz"))
    args = parser.parse_args()

    sf = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
    sf.configure({"Threads": args.threads, "Hash": args.hash})

    book: dict[int, str] = {}
    stats = {"nodes": 0, "stored": 0}
    started = time.monotonic()
    # Half the budget per colour, so the book is not lopsided if the clock runs out.
    per_side = args.minutes * 60 / 2

    for name, color in (("white", chess.WHITE), ("black", chess.BLACK)):
        print(f"\n=== building the {name} side ===", flush=True)
        build(sf, color, book, time.monotonic() + per_side, args.depth,
              args.multipv, args.margin, args.max_plies, stats)

    sf.quit()

    keys = np.fromiter(book.keys(), dtype=np.uint64, count=len(book))
    order = np.argsort(keys)
    moves = np.array([book[int(k)] for k in keys], dtype="S5")
    np.savez_compressed(args.out, keys=keys[order], moves=moves[order])

    size_kb = Path(args.out).stat().st_size / 1024
    print(f"\n{len(book):,} positions booked from {stats['nodes']:,} analysed, "
          f"{time.monotonic() - started:.0f}s")
    print(f"wrote {args.out} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
