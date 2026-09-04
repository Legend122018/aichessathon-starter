"""Opening book lookup.

Reads the book built by ``overnight/build_book.py`` from ``data/book.npz``: positions we
can be asked about in the opening, each with a move Stockfish picked at depth. Keys are
python-chess's Polyglot Zobrist hashes, which fold in side to move, castling rights and a
capturable en-passant file, so a hit means the whole position matches and not just the
piece placement.

Two reasons this exists. The moves are better than the three seconds of search they
replace, and - the larger effect - they cost no clock at all. Every rated game so far ran
at about three seconds a move until roughly move 32 and then had nothing left, finishing
without searching; a book answers its moves in microseconds and hands that time back to
the middlegame.

Like ``tablebase``, every public function returns ``None`` rather than raising, so callers
can probe unconditionally and fall through to the search. A missing, truncated or corrupt
book file disables the book and changes nothing else.
"""
import os

import chess
import chess.polyglot

_BOOK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "book.npz"
)

_book = None
_book_missing = False


def _get_book():
    """Load the book once, on first probe. Cheap enough not to need the init budget:
    a few thousand entries is a millisecond or two."""
    global _book, _book_missing
    if _book is None and not _book_missing:
        try:
            import numpy as np

            with np.load(_BOOK_PATH) as data:
                keys = data["keys"]
                moves = data["moves"]
            _book = {int(k): m.decode("ascii") for k, m in zip(keys, moves, strict=True)}
        except Exception:
            _book_missing = True
    return _book


def probe_move(board: chess.Board):
    """The book's move for this position, or ``None``.

    The move is verified legal before it is handed back. That is not paranoia about the
    builder: Zobrist keys are 64 bits, so a collision is possible in principle, and the
    cost of checking is one legality test against the cost of forfeiting a game.
    """
    book = _get_book()
    if not book:
        return None
    try:
        uci = book.get(chess.polyglot.zobrist_hash(board))
        if uci is None:
            return None
        move = chess.Move.from_uci(uci)
        return uci if move in board.legal_moves else None
    except Exception:
        return None


def size():
    """How many positions the book holds; 0 if it failed to load. For diagnostics."""
    book = _get_book()
    return len(book) if book else 0
