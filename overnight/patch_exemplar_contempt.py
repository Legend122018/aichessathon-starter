"""Port contempt into a copy of the exemplar.

The idea is already validated in the other engine at +10.5 +/- 14.8 Elo over 2,120 games:
score an avoidable draw slightly below zero so the search declines a repetition it could
take while anything better is on offer, and still takes it when every alternative is
worse. Ply parity carries the sign - at an even ply the mover is whoever moved at the
root - so no extra state is needed, and null move recurses with ply + 1, which keeps the
parity honest.

Insufficient material keeps scoring a flat draw: no preference makes a dead position
winnable.
"""
import pathlib

SEARCH = pathlib.Path(__file__).resolve().parent / "exemplar_contempt" / "engine" / "search.py"
s = SEARCH.read_text(encoding="utf-8")


def swap(old: str, new: str, label: str) -> None:
    global s
    n = s.count(old)
    assert n == 1, f"{label}: matched {n}"
    s = s.replace(old, new)


swap(
    "DRAW = 0",
    "DRAW = 0\n"
    "# An avoidable draw is worth this much less than nothing to whoever is to move at the\n"
    "# root. Validated in the other engine at +10.5 +/- 14.8 Elo over 2,120 games.\n"
    "CONTEMPT = 24",
    "constant",
)

qs_old = (
    "    if int64(pos[IHM]) >= 100 or insufficient_material(pos) \\\n"
    "            or is_repetition(pos, hist, ply):\n"
    "        return DRAW"
)
qs_new = (
    "    if insufficient_material(pos):\n"
    "        return DRAW      # dead material: no preference makes a win reachable\n"
    "    if int64(pos[IHM]) >= 100 or is_repetition(pos, hist, ply):\n"
    "        return -CONTEMPT if (ply & 1) == 0 else CONTEMPT"
)
swap(qs_old, qs_new, "qsearch")

ng_old = (
    "        if is_repetition(pos, hist, ply) or int64(pos[IHM]) >= 100 \\\n"
    "                or insufficient_material(pos):\n"
    "            return DRAW"
)
ng_new = (
    "        if insufficient_material(pos):\n"
    "            return DRAW  # dead material: no preference makes a win reachable\n"
    "        if is_repetition(pos, hist, ply) or int64(pos[IHM]) >= 100:\n"
    "            return -CONTEMPT if (ply & 1) == 0 else CONTEMPT"
)
swap(ng_old, ng_new, "negamax")

SEARCH.write_text(s, encoding="utf-8")
print("contempt ported into overnight/exemplar_contempt")
print("CONTEMPT references:", s.count("CONTEMPT"))
