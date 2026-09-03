"""Generate many single-parameter variants of the current engine, for screening.

Every constant in the search was picked once by someone and never revisited. Some of
them are load-bearing and some are arbitrary, and the only way to tell is to vary one at
a time and look. These are cheap to produce and cheap to screen; the expensive part -
playing thousands of games - is reserved for the handful that survive.

The generated names say what they do, so a verdict is readable months from now.

    python sweep.py sprint_champion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# The JIT half only, and each entry a single textual swap: (name, blurb, old, new).
# `old` must appear exactly once, which is checked before anything is written.
SWEEP: list[tuple[str, str, str, str]] = [
    # --- late move reductions: the +35 winner, now varied around its chosen shape ------
    ("lmr_div175", "reduce harder (log divisor 2.25 -> 1.75)",
     "_np.log(_d) * _np.log(_i) / 2.25", "_np.log(_d) * _np.log(_i) / 1.75"),
    ("lmr_div275", "reduce softer (log divisor 2.25 -> 2.75)",
     "_np.log(_d) * _np.log(_i) / 2.25", "_np.log(_d) * _np.log(_i) / 2.75"),
    ("lmr_base100", "reduction base 0.75 -> 1.00",
     "int(0.75 + _np.log(_d)", "int(1.00 + _np.log(_d)"),
    ("lmr_from3", "start reducing at the 4th move, not the 5th",
     "        for _i in range(4, 64):", "        for _i in range(3, 64):"),

    # --- null move: also a winner, also never varied -----------------------------------
    ("null_div4", "null-move reduction grows faster (depth//6 -> depth//4)",
     "JIT_NULL_R - depth // 6", "JIT_NULL_R - depth // 4"),
    ("null_min4", "allow null move from depth 2 rather than 3",
     "and depth >= 3 and static >= beta", "and depth >= 2 and static >= beta"),

    # --- internal iterative reduction: accepted at depth 4, never swept ----------------
    ("iir_from3", "reduce with no table move from depth 3",
     "        if tt_move == 0 and depth >= 4:", "        if tt_move == 0 and depth >= 3:"),
    ("iir_two", "reduce two plies with no table move at high depth",
     "        if tt_move == 0 and depth >= 4:\n            depth -= 1\n",
     "        if tt_move == 0 and depth >= 4:\n            depth -= 2 if depth >= 8 else 1\n"),

    # --- reverse futility: the margin has been 120 since v3 ----------------------------
    ("rfp_90", "reverse futility margin 120 -> 90 per ply",
     "        if (not is_pv) and (not checked) and depth <= 3 and static - 120 * depth >= beta:",
     "        if (not is_pv) and (not checked) and depth <= 3 and static - 90 * depth >= beta:"),
    ("rfp_160", "reverse futility margin 120 -> 160 per ply",
     "        if (not is_pv) and (not checked) and depth <= 3 and static - 120 * depth >= beta:",
     "        if (not is_pv) and (not checked) and depth <= 3 and static - 160 * depth >= beta:"),
    ("rfp_depth5", "reverse futility out to depth 5",
     "        if (not is_pv) and (not checked) and depth <= 3 and static - 120 * depth >= beta:",
     "        if (not is_pv) and (not checked) and depth <= 5 and static - 120 * depth >= beta:"),

    # --- quiescence -------------------------------------------------------------------
    ("delta_120", "tighter delta pruning in quiescence (200 -> 120)",
     "    JIT_DELTA = 200", "    JIT_DELTA = 120"),
    ("delta_300", "looser delta pruning in quiescence (200 -> 300)",
     "    JIT_DELTA = 200", "    JIT_DELTA = 300"),

    # --- move ordering ----------------------------------------------------------------
    ("hist_fast", "history bonus grows faster (depth^2 -> 2*depth^2)",
     "                    bump = hist_heur[h] + depth * depth",
     "                    bump = hist_heur[h] + 2 * depth * depth"),
]


def build(source: str, name: str) -> str:
    for key, _, old, new in SWEEP:
        if key != name:
            continue
        seen = source.count(old)
        if seen != 1:
            raise SystemExit(f"{key}: anchor matched {seen} times, expected 1")
        return source.replace(old, new)
    raise SystemExit(f"unknown variant {name!r}")


def main() -> None:
    champion = Path(sys.argv[1])
    text = champion.read_text()
    # The cloud box that screens these is far slower than the machine that plays them,
    # and would otherwise refuse to compile. Screening copies only; what ships is built
    # from the same source without this line touched.
    forced = text.replace("    return probe * 11.0 < 40.0\n",
                          "    return True  # screening copy\n")
    made = 0
    for key, blurb, _, _ in SWEEP:
        Path(f"sw_{key}.py").write_text(build(forced, key))
        print(f"  sw_{key:<16} {blurb}")
        made += 1
    print(f"\n{made} variants written")


if __name__ == "__main__":
    main()
