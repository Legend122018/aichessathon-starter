"""Check the opening book end to end, through the real agent.

The book is worth having for two reasons and this checks both: that its moves are legal
and actually get played, and that they cost effectively no clock. It also checks the case
that matters most for not losing games - that a missing or corrupt book file degrades to
a normal search instead of throwing, since `get_move` catching the exception would still
mean a move chosen by `next(iter(legal_moves))`.

    python overnight/check_book.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import chess

HERE = Path(__file__).resolve().parent
EXEMPLAR = HERE / "exemplar"

sys.path.insert(0, str(EXEMPLAR))
from engine import book  # noqa: E402

import agent  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> None:
    print(f"book holds {book.size():,} positions\n")
    check("book loaded", book.size() > 0)

    # A game against a book-following opponent, which is the case the book is built for.
    # The agent plays both sides here, so a "book move" is one ply; a single player in a
    # real game gets about half of them.
    board = chess.Board()
    hits, searched, book_ms, search_ms = 0, 0, 0.0, 0.0
    white_saved_ms = 0.0
    # A real clock, not a constant 120s. Handing the agent the same time_left on every
    # call makes each search as long as the first and overstates what the book saves,
    # because the budget is a fraction of what is left and what is left keeps falling.
    clock_ms = 120_000.0
    # And a second clock, for the White player alone, running as if there were no book.
    # This is what the saving has to be measured against. Multiplying book hits by the
    # length of the searches that came after them - the obvious thing, and what this
    # printed first - is circular: those searches are long precisely because the book
    # had already banked the time, so it reports the saving roughly twice over.
    nobook_clock_ms = 120_000.0
    for _ in range(60):
        if board.is_game_over():
            break
        in_book = book.probe_move(board) is not None
        white_to_move = board.turn == chess.WHITE
        # What this move would have cost White with an empty book: the engine's own
        # budget, off a clock that never got the book's time back.
        would_spend, _ = agent._time_budget(nobook_clock_ms, board)
        started = time.perf_counter()
        uci = agent.get_move(board.fen(), int(clock_ms))
        elapsed = (time.perf_counter() - started) * 1000
        clock_ms = clock_ms - elapsed + 500      # the competition's 0.5s increment
        if white_to_move:
            nobook_clock_ms = nobook_clock_ms - would_spend + 500
            if in_book:
                white_saved_ms += would_spend
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            check("legal move returned", False, f"unparseable {uci!r}")
            break
        if move not in board.legal_moves:
            check("legal move returned", False, f"illegal {uci} in {board.fen()}")
            break
        board.push(move)
        if in_book:
            hits += 1
            book_ms += elapsed
        else:
            searched += 1
            search_ms += elapsed
        if searched >= 6:
            break        # past the book; no need to play out a whole game

    print()
    check("book moves were played", hits > 0, f"{hits} plies from book")
    if hits:
        avg = book_ms / hits
        check("book moves are effectively free", avg < 50, f"{avg:.1f}ms average")
        print(f"\n  {hits} book plies at {avg:.1f}ms each, so about {hits // 2} moves "
              "for a single player")
        print(f"  White would have spent {white_saved_ms / 1000:.1f}s on those moves "
              f"without a book,")
        print(f"  which is {white_saved_ms / 1200:.0f}% of a 120s clock moved into the "
              "middlegame")

    # The failure mode that costs games, not Elo: if a bad book raises rather than
    # returning None, get_move's except-branch plays the first legal move it can find.
    print()
    book._book = None
    book._book_missing = False
    original = book._BOOK_PATH
    book._BOOK_PATH = str(HERE / "does-not-exist.npz")
    try:
        check("missing book returns None", book.probe_move(chess.Board()) is None)
        uci = agent.get_move(chess.STARTING_FEN, 120_000)
        legal = {m.uci() for m in chess.Board().legal_moves}
        check("agent still plays without a book", uci in legal, f"played {uci}")
    finally:
        book._BOOK_PATH = original
        book._book = None
        book._book_missing = False

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("All checks passed!")


if __name__ == "__main__":
    main()
