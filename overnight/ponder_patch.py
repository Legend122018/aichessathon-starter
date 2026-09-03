"""Add pondering: think on the opponent's clock instead of idling through it.

The rules allow it explicitly - "the process keeps its dedicated core after get_move
returns, and pondering is allowed" - and the engine has never used it. At 120s + 0.5s
the opponent spends roughly as long thinking as we do, so the idle time is on the same
order as the thinking time.

The awkward part is stopping. numba holds the GIL inside a jitted call, so a background
search in one long call would block get_move from even beginning, and every millisecond
of that comes off our clock. So the background search runs in small node-capped slices
and checks a stop flag between them, which bounds how long a stop can take at the cost
of not reaching full depth. What it leaves behind is transposition table entries for the
position we are about to be asked about, which is most of the value: the real search
then starts with its move ordering already right.

    python ponder_patch.py try_ownpst.py try_ponder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

IMPORT_ANCHOR = "import random\nimport time\n"

STATE_ANCHOR = 'JIT_STATE: dict[str, Any] = {}\n'

PONDER_BLOCK = '''
# --------------------------------------------------------------------------------------
# Pondering
# --------------------------------------------------------------------------------------
# The rules allow using the core after get_move returns. One rule shapes the whole
# design: numba holds the GIL inside a jitted call, so the background search must run in
# slices short enough that stopping it never costs meaningful clock.

PONDER: dict[str, Any] = {"thread": None, "stop": True, "fen": "",
                          "hits": 0, "tries": 0}

# Stopping costs at most one slice, because that is the granularity at which the stop
# flag can be read. Forty thousand nodes is roughly thirty milliseconds, so a sixty move
# game gives up under two seconds of a 120 second clock in the worst case.
PONDER_SLICE_NODES = 40000
PONDER_MAX_SLICES = 400


def _ponder_stop() -> bool:
    """Halt the background search and wait for it to actually finish.

    Returns False if it did not stop. That answer matters: a thread still inside
    jit_search_root is still writing to every array the real search is about to use, and
    two searches sharing one workspace produce a move that belongs to neither. The
    handle is deliberately left set in that case, because the thread still owns the
    workspace, and the caller plays the move from the pure-Python engine instead.
    """
    thread = PONDER["thread"]
    if thread is None:
        return True
    PONDER["stop"] = True
    thread.join(timeout=2.0)
    if thread.is_alive():
        return False
    PONDER["thread"] = None
    return True


def _ponder_run(fen: str, rep_keys: list[int]) -> None:
    """Iterative deepening on the predicted position, into the shared table.

    search_root cannot resume a depth it abandoned, so an unfinished depth is simply
    attempted again - the entries the abandoned attempt left behind make the retry
    cheaper, and the search creeps forward rather than stalling. What matters at the end
    is not the depth reached but the table: the next real search starts with its move
    ordering already right.
    """
    try:
        work = JIT_STATE["work"]
        board, st = jit_new_state(fen)
        rep = work["rep"]
        for index, key in enumerate(rep_keys[-80:]):
            rep[index] = key
        base_rep = min(len(rep_keys), 80)
        best = 0
        depth = 1
        slices = 0
        while depth < 40 and slices < PONDER_MAX_SLICES:
            if PONDER["stop"]:
                return
            # Hand the interpreter lock over before taking it again. Without this the
            # loop reacquires it the instant a slice ends, and get_move - which needs
            # the lock merely to begin - can wait through many slices before it runs.
            # That is a direct loss of clock, and it is severe enough to lose games.
            time.sleep(0.002)
            if PONDER["stop"]:
                return
            work["info"][NODES] = 0
            work["info"][STOPPED] = 0
            work["info"][REP_LEN] = base_rep
            work["info"][NODE_LIMIT] = PONDER_SLICE_NODES
            jit_search_root(board, st, work["hist"], work["moves"], work["scores"],
                            work["occ"], work["info"], work["killers"], work["hist_heur"],
                            work["counter"], rep, work["tt_key"], work["tt_move"],
                            work["tt_score"], work["tt_meta"], depth, -31000, 31000, best,
                            work["out"])
            slices += 1
            if work["info"][STOPPED] == 1:
                # The depth did not fit in one slice. Try it again rather than giving up:
                # the entries this attempt wrote make the next one cheaper, so the search
                # creeps forward instead of stalling the moment a depth gets expensive.
                continue
            best = int(work["out"][1])
            depth += 1
    except Exception:
        # A failure here must cost nothing. The move that matters is chosen on the main
        # thread, which does not depend on any of this having worked.
        return


def _ponder_start(fen: str, played: Any) -> None:
    """Guess the reply and search the position it would give, on the opponent's time.

    The guess comes from the transposition table: after our move, the entry for the new
    position holds the best reply the search just found. That is the same prediction a
    principal variation would give, without having to keep one.
    """
    try:
        if not JIT_READY or PONDER["thread"] is not None:
            return
        board_obj = chess.Board(fen)
        board_obj.push(played)
        if board_obj.is_game_over():
            return

        work = JIT_STATE["work"]
        board, st = jit_new_state(board_obj.fen())
        key = jit_full_hash(board, st)
        slot = int(key & TT_MASK)
        if work["tt_meta"][slot] == 0 or work["tt_key"][slot] != key:
            return
        guess = int(work["tt_move"][slot])
        if guess == 0:
            return
        reply = chess.Move.from_uci(jit_move_uci(guess))
        if reply not in board_obj.legal_moves:
            return
        board_obj.push(reply)
        if board_obj.is_game_over():
            return

        rep_keys = []
        for past in HISTORY_FENS[-78:]:
            past_board, past_st = jit_new_state(past)
            rep_keys.append(jit_full_hash(past_board, past_st))
        after_ours, after_ours_st = jit_new_state(board_obj.fen())
        rep_keys.append(jit_full_hash(after_ours, after_ours_st))

        PONDER["stop"] = False
        PONDER["fen"] = board_obj.fen()
        PONDER["tries"] += 1
        thread = threading.Thread(target=_ponder_run, args=(board_obj.fen(), rep_keys),
                                  daemon=True)
        PONDER["thread"] = thread
        thread.start()
    except Exception:
        PONDER["thread"] = None


'''

GET_MOVE_ANCHOR = '''def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in `fen`."""
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if not moves:
        return "0000"
    fallback = moves[0].uci()
'''

GET_MOVE_NEW = '''def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in `fen`."""
    # First, before anything reads the shared workspace: stop the background search that
    # has been running on the opponent's clock. Everything below assumes it has finished.
    if not _ponder_stop():
        # It would not let go of the workspace, which should not be reachable given
        # node-capped slices. Play from the pure-Python engine, which shares none of
        # those arrays, rather than racing a thread for them.
        return _python_get_move(fen, time_left_ms)
    if PONDER["fen"] == fen:
        PONDER["hits"] += 1

    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if not moves:
        return "0000"
    fallback = moves[0].uci()
'''


def patch(source: str) -> str:
    def swap(text: str, old: str, new: str, label: str) -> str:
        seen = text.count(old)
        if seen != 1:
            raise SystemExit(f"{label}: anchor matched {seen} times, expected 1")
        return text.replace(old, new)

    source = swap(source, IMPORT_ANCHOR, "import random\nimport threading\nimport time\n",
                  "import")
    source = swap(source, STATE_ANCHOR, STATE_ANCHOR + PONDER_BLOCK, "ponder-block")
    source = swap(source, GET_MOVE_ANCHOR, GET_MOVE_NEW, "get-move-stop")

    # Start pondering on the way out, wherever the jitted path returns a move it chose.
    old_return = """        uci = _jit_choose(fen, time_left_ms)
        if uci and chess.Move.from_uci(uci) in moves:
            return uci
        return _python_get_move(fen, time_left_ms)"""
    new_return = """        uci = _jit_choose(fen, time_left_ms)
        if uci and chess.Move.from_uci(uci) in moves:
            # Hand the spare core to the position we expect to be asked about next.
            _ponder_start(fen, chess.Move.from_uci(uci))
            return uci
        return _python_get_move(fen, time_left_ms)"""
    return swap(source, old_return, new_return, "ponder-start")


if __name__ == "__main__":
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.write_text(patch(src.read_text()))
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
