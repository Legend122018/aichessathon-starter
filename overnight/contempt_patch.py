"""Score a draw as slightly bad for us, so the search plays on instead of repeating.

Against Stockfish at 2500 to 2900 the engine went three wins, three draws, four losses
per ten. A third of those games ended level, and on a ladder ranked by Elo a draw against
a stronger opponent is worth having while a draw against a weaker one is a win thrown
away. Contempt is the standard lever: tell the search a draw is worth a little less than
nothing, and it will decline a repetition it would otherwise take.

The sign has to follow the side to move, because negamax reports every score from the
mover's point of view. At an even ply we are the mover and a draw is worth minus the
contempt; at an odd ply the opponent is the mover and the same draw is worth plus it. Get
that backwards and the engine hunts for draws instead of avoiding them.

Two values are offered rather than one because the right amount is not guessable. Too
little does nothing; too much makes the engine refuse a draw it should be grateful for
and lose games it had already saved.

    python contempt_patch.py try_ownpst.py 20 try_contempt20.py
"""

from __future__ import annotations

import sys
from pathlib import Path

DRAW_RETURNS = """        alpha_original = alpha
        if ply > 0:
            for i in range(info[REP_LEN]):
                if rep[i] == key:
                    return 0
            if st[3] >= 100:
                return 0
"""

STALEMATE = """        if legal == 0:
            return -JIT_MATE + ply if checked else 0
"""

CONST_ANCHOR = "    JIT_FUTILITY = _np.array([0, 150, 320, 520], dtype=_np.int32)\n"


def patch(source: str, value: int) -> str:
    def swap(text: str, old: str, new: str, label: str) -> str:
        seen = text.count(old)
        if seen != 1:
            raise SystemExit(f"{label}: anchor matched {seen} times, expected 1")
        return text.replace(old, new)

    source = swap(
        source, CONST_ANCHOR,
        CONST_ANCHOR + f"""    # A draw is worth this much less than nothing to us. Signed by ply below, because
    # negamax scores from the mover's point of view and the two sides disagree about
    # whether a draw is good news.
    JIT_CONTEMPT = {value}
""",
        "contempt-const")

    source = swap(
        source, DRAW_RETURNS,
        """        alpha_original = alpha
        if ply > 0:
            for i in range(info[REP_LEN]):
                if rep[i] == key:
                    return -JIT_CONTEMPT if (ply & 1) == 0 else JIT_CONTEMPT
            if st[3] >= 100:
                return -JIT_CONTEMPT if (ply & 1) == 0 else JIT_CONTEMPT
""",
        "draw-returns")

    return swap(
        source, STALEMATE,
        """        if legal == 0:
            if checked:
                return -JIT_MATE + ply
            return -JIT_CONTEMPT if (ply & 1) == 0 else JIT_CONTEMPT
""",
        "stalemate")


if __name__ == "__main__":
    src = Path(sys.argv[1])
    amount = int(sys.argv[2])
    out = Path(sys.argv[3])
    out.write_text(patch(src.read_text(), amount))
    print(f"wrote {out} with contempt {amount}")
