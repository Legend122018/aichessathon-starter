"""AI Chessathon competition entry point.

The harness calls `get_move(fen, time_left_ms)` once per move -- it hands us
only the current position and our remaining clock, never a move list. To keep
the transposition table, killer/history heuristics and repetition history
alive across a game we track our own `python-chess` Board and reconcile it
against each incoming FEN by checking whether it is reachable from our board
by one legal move (the opponent's reply); see `_resync`. Anything that does
not match -- the first call of a game, or an unexpected jump -- starts a
fresh position, which is always correct, just without pre-call history.

All one-off setup (numba JIT compilation of the whole engine) happens at
import time via `_warmup()`, so it is paid for out of the 60s initialization
budget rather than out of the per-move clock.

Two extra sources of strength on top of the raw search:
  - Syzygy tablebase probing (engine/tablebase.py) for <=4-man positions --
    instant and exact, tried before spending any search time.
  - Pondering: after we return our move, a background thread keeps
    searching the position we expect to face next (our own PV's guess at
    the opponent's reply), sharing the same transposition table, using the
    dedicated core the process keeps between moves. See `_start_ponder` /
    `_stop_ponder`.
"""
import threading
import time

import chess

from engine.engine import Engine
from engine import tablebase

TT_BITS = 22

INCREMENT_MS = 500          # matches the competition's 0.5s/move increment
OVERHEAD_MS = 100           # fixed per-move slack for harness/IO/GC jitter
PANIC_MS = 5000             # below this, play near-instantly rather than risk flagging
MIN_MS = 15
TB_MAX_PIECES = 4           # matches the 3+4-piece Syzygy set shipped in data/tablebases


class _State:
    __slots__ = ("engine", "board")


_STATE = _State()
_STATE.engine = Engine(tt_bits=TT_BITS)
_STATE.board = None

# A separate Engine for pondering, sharing only the transposition table
# buffer with _STATE.engine (never its pos/hist/undo arrays). Ponder work
# still pays off -- TT hits carry over -- but _STATE.engine's accumulated
# Zobrist history (needed for real-game repetition detection, see
# engine/search.py's is_repetition) is never touched by the ponder thread,
# so there's no cleanup-ordering bug to get right and _STATE.engine is
# always immediately safe to use in get_move without waiting on a join.
# Never call _PONDER.new_game() -- that zeroes .tt in place, which would
# wipe the shared table out from under _STATE.engine too.
_PONDER = Engine(tt_bits=TT_BITS)
_PONDER.tt = _STATE.engine.tt
_PONDER.tt_size = _STATE.engine.tt_size

_ponder_thread = None       # background Thread, or None when not pondering


def _resync(fen):
    """Bring our tracked board+engine state in line with the given FEN.

    Tries to explain the new FEN as "our last board plus one legal move" so
    the opponent's reply gets pushed into the engine's own state (preserving
    TT entries and Zobrist history for repetition detection). Falls back to
    a full reset -- correct, just colder -- whenever that fails.
    """
    target = chess.Board(fen)
    target_fen = target.fen()
    board = _STATE.board
    if board is not None:
        for mv in board.legal_moves:
            board.push(mv)
            # Compare serialized FEN, not raw .ep_square: python-chess sets
            # .ep_square on any double pawn push regardless of whether it's
            # actually capturable, but .fen() only serializes it when it is
            # -- so a board freshly parsed from a FEN string (like `target`)
            # can have .ep_square=None while an equivalent locally-pushed
            # board has it set, even though the positions are identical.
            # Comparing the two boards' own .fen() output side-steps that
            # inconsistency entirely, since both apply the same rule.
            same = board.fen() == target_fen
            board.pop()
            if same:
                if _STATE.engine.push_uci(mv.uci()):
                    board.push(mv)
                    _STATE.board = board
                    return
                break   # engine desynced from board; fall through to a reset

    _STATE.engine.new_game()
    _STATE.engine.set_position(fen)
    _STATE.board = target


def _time_budget(time_left_ms, board):
    """Split the remaining clock into a soft/hard search budget for one move.

    Conservative first cut at Stage-8 time management: a shrinking
    moves-to-go estimate from the move number, a fixed safety reserve so a
    slow last iteration can never flag, and a panic mode for very low time.
    """
    time_left_ms = max(0.0, float(time_left_ms))
    reserve = OVERHEAD_MS + INCREMENT_MS * 0.5
    usable = max(0.0, time_left_ms - reserve)

    if time_left_ms <= PANIC_MS:
        soft = max(MIN_MS, usable * 0.15)
        hard = max(MIN_MS, usable * 0.30)
        return soft, hard

    moves_to_go = max(8, 45 - min(board.fullmove_number, 40))
    soft = usable / moves_to_go + INCREMENT_MS * 0.8
    hard = min(soft * 3.5, usable * 0.5)
    hard = max(hard, MIN_MS)
    soft = min(soft, hard)
    return soft, hard


# ------------------------------------------------------------------ pondering
def _stop_ponder(timeout=2.0):
    """Signal any running ponder search to halt and wait (briefly) for it.

    Retries the stop signal rather than setting it once: `Engine.search()`
    starts by unconditionally clearing N[ON_STOP] (`_arm_timer`), so a
    single `.stop()` call racing against a ponder thread that hasn't
    reached `.search()` yet can be silently erased -- and since pondering
    runs with hard_ms=None (no internal Timer safety net), a lost stop
    request would otherwise let the search run unbounded, contending for
    the one dedicated core during our real move. Re-asserting every 50ms
    reliably lands after that one-time reset (thread setup before
    `.search()` is microseconds), while `_PONDER` never touching
    `_STATE.engine`'s pos/hist/undo arrays means `_STATE.engine` stays safe
    to use immediately even in the pathological case this doesn't resolve
    within `timeout`.
    """
    global _ponder_thread
    t = _ponder_thread
    if t is None:
        return
    deadline = time.perf_counter() + timeout
    while t.is_alive() and time.perf_counter() < deadline:
        _PONDER.stop()
        t.join(0.05)
    if not t.is_alive():
        _ponder_thread = None


def _ponder_worker(resume_fen, predicted_uci):
    try:
        _PONDER.set_position(resume_fen)
        if not _PONDER.push_uci(predicted_uci):
            return
        _PONDER.search(max_depth=64, hard_ms=None, soft_ms=None)
    except Exception:
        pass


def _start_ponder(info, resume_fen):
    """Kick off a background search on our best guess at the opponent's
    reply, using the PV from the move we just finished searching."""
    global _ponder_thread
    if _ponder_thread is not None and _ponder_thread.is_alive():
        return   # previous ponder hasn't wound down yet; skip this cycle
    if not info or not info.pv or len(info.pv) < 2:
        return
    predicted_uci = info.pv[1]
    t = threading.Thread(target=_ponder_worker, args=(resume_fen, predicted_uci),
                          daemon=True)
    _ponder_thread = t
    t.start()


# ------------------------------------------------------------------ get_move
def get_move(fen: str, time_left_ms: int) -> str:
    _stop_ponder()

    try:
        _resync(fen)
        board = _STATE.board
        legal = list(board.legal_moves)
        if not legal:
            return "0000"          # terminal position; harness should not call us here
        legal_ucis = {m.uci() for m in legal}

        if chess.popcount(board.occupied) <= TB_MAX_PIECES:
            tb_uci = tablebase.probe_best_move(board)
            if tb_uci in legal_ucis:
                board.push_uci(tb_uci)
                _STATE.engine.push_uci(tb_uci)
                return tb_uci

        soft_ms, hard_ms = _time_budget(time_left_ms, board)
        info = _STATE.engine.search(max_depth=64, hard_ms=hard_ms, soft_ms=soft_ms)
        uci = info.bestmove if info.bestmove in legal_ucis else legal[0].uci()
        board.push_uci(uci)
        _STATE.engine.push_uci(uci)
        if not board.is_game_over(claim_draw=False):
            _start_ponder(info, board.fen())
        return uci
    except Exception:
        pass

    # Fallback path: something above failed unexpectedly. Rebuild state from
    # scratch off the FEN alone -- never assume whatever partial state the
    # failed attempt left behind (e.g. `_STATE.board` may still be None, on
    # a first-call failure) is usable.
    try:
        board = chess.Board(fen)
        uci = next(iter(board.legal_moves)).uci()
    except Exception:
        return "0000"
    try:
        board.push_uci(uci)
        _STATE.board = board
        _STATE.engine.new_game()
        _STATE.engine.set_position(fen)
        _STATE.engine.push_uci(uci)
    except Exception:
        pass  # best-effort; next call's _resync will detect and recover
    return uci


def _warmup():
    """Force every njit code path to compile now, charged to the init budget."""
    t0 = time.perf_counter()
    _STATE.engine.set_position(chess.STARTING_FEN)
    _STATE.engine.search(max_depth=8)
    _STATE.engine.new_game()
    _STATE.board = None
    return time.perf_counter() - t0


_warmup()
