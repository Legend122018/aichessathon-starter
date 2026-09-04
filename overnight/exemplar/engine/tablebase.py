"""Syzygy endgame tablebase probing.

Wraps ``chess.syzygy`` over the complete 3-piece + 4-piece WDL+DTZ set
shipped at ``data/tablebases``. Every public function returns ``None``
whenever the position is out of scope (castling rights still held, more
pieces than we ship tables for, or the table directory is missing) so
callers can unconditionally try tablebase probing and fall back to search
without special-casing.

Uses ``chess.syzygy``'s non-raising ``get_wdl``/``get_dtz`` accessors, which
already handle the castling-rights and too-many-pieces cases internally
(see ``chess.syzygy.Tablebase.probe_ab``) -- no piece-count pre-check needed
here.
"""
import os

import chess
import chess.syzygy

_TB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tablebases"
)

_tablebase = None
_tablebase_missing = False


def _get_tablebase():
    global _tablebase, _tablebase_missing
    if _tablebase is None and not _tablebase_missing:
        try:
            _tablebase = chess.syzygy.open_tablebase(_TB_DIR)
        except Exception:
            _tablebase_missing = True
    return _tablebase


def _as_board(board_or_fen):
    return board_or_fen if isinstance(board_or_fen, chess.Board) else chess.Board(board_or_fen)


def probe_result(board_or_fen):
    """Game-theoretic result for the side to move: +1 win, 0 draw, -1 loss,
    or None if this position can't be probed (out of scope, or the table
    for its material isn't in our shipped set). Collapses the WDL scale's
    cursed-win/blessed-loss nuance (+-1) into the eventual sign (+-2 -> same
    sign) since we only ship this for move selection, not 50-move-rule
    bookkeeping."""
    tb = _get_tablebase()
    if tb is None:
        return None
    board = _as_board(board_or_fen)
    wdl = tb.get_wdl(board)
    if wdl is None:
        return None
    if wdl > 0:
        return 1
    if wdl < 0:
        return -1
    return 0


def probe_best_move(board_or_fen):
    """UCI move that preserves the position's tablebase result (never turns
    a win into a draw, or a draw into a loss), or None if unprobeable.

    Among result-preserving moves: if winning, prefers the smallest DTZ
    (fastest forced conversion); if losing, prefers the largest DTZ (most
    resistance, giving the opponent the most chances to slip); if drawing,
    DTZ doesn't matter, any result-preserving move is fine.
    """
    tb = _get_tablebase()
    if tb is None:
        return None
    board = _as_board(board_or_fen)
    root_wdl = tb.get_wdl(board)
    if root_wdl is None:
        return None
    root_sign = (root_wdl > 0) - (root_wdl < 0)

    candidates = []
    for move in board.legal_moves:
        board.push(move)
        try:
            child_wdl = tb.get_wdl(board)
            if child_wdl is not None:
                child_sign = (child_wdl > 0) - (child_wdl < 0)
                if -child_sign == root_sign:
                    dtz = tb.get_dtz(board)
                    candidates.append((abs(dtz) if dtz is not None else None, move))
        finally:
            board.pop()

    if not candidates:
        return None
    ranked = [c for c in candidates if c[0] is not None]
    if not ranked:
        return candidates[0][1].uci()
    ranked.sort(key=lambda c: c[0], reverse=(root_sign < 0))
    return ranked[0][1].uci()
