"""Single-change search patches, applied as text edits to a built agent file.

Every entry changes exactly one thing. That is the whole point: v4 bundled log-scaled
LMR, razoring and adaptive null-move into one commit, lost 7.5% against v3, and told us
nothing about which of the three was at fault. Tested one at a time, a loss is
attributable and a win is bankable.

These edit the numba half only. That half is what plays; the pure-Python engine is a
fallback for a machine that cannot compile, and the two are allowed to search
differently as long as they evaluate identically - which the tables, shared by both,
guarantee.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------- anchors
# Text that must appear exactly once in the agent file. If an anchor stops matching the
# patch is refused rather than silently doing nothing.

FUTILITY_CONST = "    JIT_FUTILITY = _np.array([0, 150, 320, 520], dtype=_np.int32)\n"

FUTILITY_SKIP = """            if (quiet and (not is_pv) and (not checked) and depth <= 3 and legal > 0
                    and static + JIT_FUTILITY[depth] <= alpha):
                continue
"""

IS_PV_LINE = "        is_pv = (beta - alpha) > 1\n"

LMR_LINE = "                    reduction = 1 if i < 8 else 2\n"

NULL_DEPTH = "                             depth - 1 - JIT_NULL_R, -beta, -beta + 1, ply + 1, False, 0)\n"


def _swap(source: str, old: str, new: str, label: str) -> str:
    seen = source.count(old)
    if seen != 1:
        raise SystemExit(f"{label}: anchor matched {seen} times, expected exactly 1")
    return source.replace(old, new)


# ------------------------------------------------------------------------- the patches

def late_move_pruning(source: str) -> str:
    """Stop searching quiet moves once enough have failed at shallow depth.

    Futility pruning already skips quiet moves whose static score cannot reach alpha.
    This skips them on move count instead: if the first dozen ordered quiets did not
    raise alpha at depth 3, the thirtieth almost never does. It is the one standard
    pruning rule the search does not have.
    """
    source = _swap(
        source, FUTILITY_CONST,
        FUTILITY_CONST + "    JIT_LMP = _np.array([0, 6, 10, 16, 24], dtype=_np.int32)\n",
        "lmp-const")
    return _swap(
        source, FUTILITY_SKIP,
        FUTILITY_SKIP +
        """            if (quiet and (not is_pv) and (not checked) and depth <= 4 and legal > 0
                    and i >= JIT_LMP[depth]):
                continue
""",
        "lmp-skip")


def internal_iterative_reduction(source: str) -> str:
    """Search one ply shallower when the table has no move to try first.

    Move ordering is what makes alpha-beta pay, and with no transposition-table move the
    ordering at this node is guesswork. Rather than spend full depth ordering badly,
    spend one less and let the table entry that search writes order the re-visit.
    """
    return _swap(
        source, IS_PV_LINE,
        IS_PV_LINE +
        """        if tt_move == 0 and depth >= 4:
            depth -= 1
""",
        "iir")


def log_lmr(source: str) -> str:
    """Reduce late quiet moves by a log-scaled amount rather than one ply or two.

    The current rule is flat: one ply for moves 5 to 8, two for anything later,
    regardless of depth. A table over (depth, move index) reduces far more aggressively
    deep in the tree, where the saving compounds, and barely at all near the leaves,
    where a wrong reduction costs a real line. This was in the v4 bundle; it has never
    been measured on its own.
    """
    source = _swap(
        source, FUTILITY_CONST,
        FUTILITY_CONST + """    JIT_LMR = _np.zeros(32 * 64, dtype=_np.int32)
    for _d in range(3, 32):
        for _i in range(4, 64):
            JIT_LMR[_d * 64 + _i] = int(0.75 + _np.log(_d) * _np.log(_i) / 2.25)
""",
        "lmr-table")
    return _swap(
        source, LMR_LINE,
        """                    d_index = depth if depth < 32 else 31
                    m_index = i if i < 64 else 63
                    reduction = JIT_LMR[d_index * 64 + m_index]
                    if reduction > depth - 2:
                        reduction = depth - 2
""",
        "lmr-lookup")


def adaptive_null(source: str) -> str:
    """Cut more off the null-move search as depth grows.

    A fixed reduction of two plies is cautious at depth 12 and about right at depth 4.
    Scaling it with depth is standard. Also in the v4 bundle, also never measured alone.
    """
    return _swap(
        source, NULL_DEPTH,
        "                             depth - 1 - JIT_NULL_R - depth // 6, -beta, -beta + 1,\n"
        "                             ply + 1, False, 0)\n",
        "null-r")


PATCHES = [
    ("lmp", "late move pruning at shallow depth", late_move_pruning),
    ("iir", "search a ply shallower with no table move", internal_iterative_reduction),
    ("lmr", "log-scaled late move reductions", log_lmr),
    ("nullr", "null-move reduction that grows with depth", adaptive_null),
]


def build(champion: Path, name: str, out: Path) -> Path:
    """Write a candidate that differs from `champion` by exactly one change."""
    for key, _, apply in PATCHES:
        if key == name:
            out.write_text(apply(champion.read_text()))
            return out
    raise SystemExit(f"unknown patch {name!r}; have {[p[0] for p in PATCHES]}")


if __name__ == "__main__":
    import sys

    champion = Path(sys.argv[1])
    for key, blurb, _ in PATCHES:
        target = champion.parent / f"try_{key}.py"
        build(champion, key, target)
        print(f"{key:6} {blurb:46} -> {target.name} ({target.stat().st_size:,} bytes)")


# ------------------------------------------------------- batch two, on the sprint winner
# The first batch found that log-scaled reductions were worth +35 and shallow late move
# pruning was worth -38. These build on that result rather than guessing again: two of
# them refine the reduction and pruning rules the first batch validated, and two remove
# ceilings that were set arbitrarily and never measured.

TT_BITS_LINE = "    TT_BITS = 21\n"

ASPIRATION_LINE = "        window = 40\n"

LMR_LOOKUP = """                    reduction = JIT_LMR[d_index * 64 + m_index]
                    if reduction > depth - 2:
                        reduction = depth - 2
"""


def bigger_tt(source: str) -> str:
    """Four times the transposition table.

    Twenty-one bits is two million entries, about 42 MB. The platform allows 2 GB and
    the engine uses a fraction of it, so the table has been small for no reason. A
    deeper search overwrites its own entries; more room means fewer positions searched
    twice.
    """
    return _swap(source, TT_BITS_LINE, "    TT_BITS = 23\n", "tt-bits")


def history_lmr(source: str) -> str:
    """Reduce a quiet move one ply further when it has never caused a cutoff.

    The first version of this rule reduced *less* for moves with a high history score,
    and screening showed it changed the node count by exactly zero. The reason is a
    circular one worth remembering: history is what orders quiet moves, so a move with a
    high history score is sorted early and never reaches the late-move branch at all.
    The condition was unreachable by construction.

    Turned around it is live and standard: among the late quiet moves, the ones with no
    history at all are the least likely to matter, so they lose an extra ply. Screening
    at fixed depth: 7% fewer nodes, no tactic missed. The colour index is `1 - st[0]`
    because this runs after the move is made.
    """
    return _swap(
        source, LMR_LOOKUP,
        LMR_LOOKUP +
        """                    if reduction > 0 and depth >= 6:
                        seen = hist_heur[(1 - st[0]) * 16384 + jit_move_from(move) * 128
                                         + jit_move_to(move)]
                        if seen == 0:
                            reduction += 1
""",
        "lmr-history")


def narrow_aspiration(source: str) -> str:
    """Open the aspiration window at 20 centipawns instead of 40.

    A narrower window fails high or low more often and pays for a re-search when it
    does, but every search that lands inside it prunes harder. Forty was a guess made
    in v3 and never revisited.
    """
    return _swap(source, ASPIRATION_LINE, "        window = 20\n", "aspiration")


def deeper_futility(source: str) -> str:
    """Extend futility pruning from depth 3 to depth 5.

    The margins grow with depth so the rule stays conservative where a quiet move still
    has enough search left to prove itself. Late move pruning at shallow depth measured
    -38, so the shape of this one matters: it prunes on evaluation, which the first
    batch did not test, rather than on move count, which it rejected.
    """
    source = _swap(
        source, FUTILITY_CONST,
        "    JIT_FUTILITY = _np.array([0, 150, 320, 520, 760, 1050], dtype=_np.int32)\n",
        "futility-margins")
    return _swap(
        source,
        "            if (quiet and (not is_pv) and (not checked) and depth <= 3 and legal > 0\n"
        "                    and static + JIT_FUTILITY[depth] <= alpha):\n",
        "            if (quiet and (not is_pv) and (not checked) and depth <= 5 and legal > 0\n"
        "                    and static + JIT_FUTILITY[depth] <= alpha):\n",
        "futility-depth")


PATCHES += [
    ("ttsize", "four times the transposition table", bigger_tt),
    ("lmrhist", "reduce less when history likes the move", history_lmr),
    ("aspwin", "narrower aspiration window", narrow_aspiration),
    ("futility5", "futility pruning to depth 5", deeper_futility),
]


# --------------------------------------------------- batch three, from the wide screen
# Sixteen single-parameter variants were generated and screened against a depth-11
# reference before any of them cost a game. Most were rejected there: the log-reduction
# divisor is already well placed, a bigger reverse-futility margin costs nodes and
# accuracy, and three of the sixteen turned out to be structurally unreachable and
# changed the node count by exactly zero. These are the three that pruned harder while
# still reproducing the deeper search's move.

def iir_from_three(source: str) -> str:
    """Apply the no-table-move reduction from depth 3 rather than depth 4.

    Screened at 0.96x nodes with agreement unchanged, the joint best of the sweep. The
    depth-4 threshold was picked when the rule was first added and never varied.
    """
    return _swap(
        source,
        "        if tt_move == 0 and depth >= 4:\n",
        "        if tt_move == 0 and depth >= 3:\n",
        "iir-from-3")


def reverse_futility_90(source: str) -> str:
    """Cut the reverse-futility margin from 120 centipawns per ply to 90.

    A node whose static score already beats beta by a wide margin is not searched. The
    margin decides how wide "wide" is; 120 has been the number since v3. Screened at
    0.96x nodes, agreement unchanged. The opposite direction, 160, was worse on both
    counts, which is what makes 90 worth a match rather than a coin flip.
    """
    return _swap(
        source,
        "        if (not is_pv) and (not checked) and depth <= 3 and static - 120 * depth >= beta:",
        "        if (not is_pv) and (not checked) and depth <= 3 and static - 90 * depth >= beta:",
        "rfp-90")


def null_from_two(source: str) -> str:
    """Allow the null-move search one ply shallower.

    The weakest of the three on the screen - 0.99x nodes, agreement held - and included
    because it is nearly free to test once the machine is already running.
    """
    return _swap(
        source,
        "and depth >= 3 and static >= beta",
        "and depth >= 2 and static >= beta",
        "null-min-2")


PATCHES += [
    ("iirfrom3", "no-table-move reduction from depth 3", iir_from_three),
    ("rfp90", "reverse futility margin 120 -> 90", reverse_futility_90),
    ("nullmin2", "allow null move from depth 2", null_from_two),
]
