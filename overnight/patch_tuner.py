"""Teach tune.py the mobility term, so Texel fits it with everything else.

The tuner's --check pass compares its features against agent.py's evaluation, so the
feature written here has to match the engine's definition exactly: squares a piece can
reach, a ray stopping at the first blocker, that blocker counted only when it is not ours.
"""
import pathlib

p = pathlib.Path(__file__).resolve().parent / "tune.py"
s = p.read_text(encoding="utf-8")


def swap(old, new, label):
    global s
    n = s.count(old)
    assert n == 1, f"{label}: matched {n}"
    s = s.replace(old, new)


MOB_PIECES = ["knight", "bishop", "rook", "queen"]

swap(
    'EXTRA += [f"passed_mg_{rank}" for rank in range(1, 7)]\n'
    'EXTRA += [f"passed_eg_{rank}" for rank in range(1, 7)]',
    'EXTRA += [f"passed_mg_{rank}" for rank in range(1, 7)]\n'
    'EXTRA += [f"passed_eg_{rank}" for rank in range(1, 7)]\n'
    'EXTRA += [f"mob_mg_{name}" for name in ("knight", "bishop", "rook", "queen")]\n'
    'EXTRA += [f"mob_eg_{name}" for name in ("knight", "bishop", "rook", "queen")]',
    "extra-names",
)

# Feature: white minus black reachable squares, per piece type, in both phases.
swap(
    """    if chess.popcount(board.bishops & white) > 1:
        bump_mg("bishop_pair_mg", 1.0)""",
    """    # Mobility. attacks_mask stops a ray at the first blocker and includes that square,
    # so masking off our own pieces leaves what the engine counts.
    for piece, name in ((chess.KNIGHT, "knight"), (chess.BISHOP, "bishop"),
                        (chess.ROOK, "rook"), (chess.QUEEN, "queen")):
        for square in chess.scan_forward(board.pieces_mask(piece, chess.WHITE)):
            reach = float(chess.popcount(board.attacks_mask(square) & ~white))
            bump_mg(f"mob_mg_{name}", reach)
            bump_eg(f"mob_eg_{name}", reach)
        for square in chess.scan_forward(board.pieces_mask(piece, chess.BLACK)):
            reach = float(chess.popcount(board.attacks_mask(square) & ~black))
            bump_mg(f"mob_mg_{name}", -reach)
            bump_eg(f"mob_eg_{name}", -reach)

    if chess.popcount(board.bishops & white) > 1:
        bump_mg("bishop_pair_mg", 1.0)""",
    "features",
)

# Read the engine's current values into the parameter vector.
swap(
    """    for rank in range(1, 7):
        weights[EXTRA_INDEX[f"passed_mg_{rank}"]] = module.PASSED_MG[rank]
        weights[EXTRA_INDEX[f"passed_eg_{rank}"]] = module.PASSED_EG[rank]
    return weights""",
    """    for rank in range(1, 7):
        weights[EXTRA_INDEX[f"passed_mg_{rank}"]] = module.PASSED_MG[rank]
        weights[EXTRA_INDEX[f"passed_eg_{rank}"]] = module.PASSED_EG[rank]
    for piece, name in ((2, "knight"), (3, "bishop"), (4, "rook"), (5, "queen")):
        weights[EXTRA_INDEX[f"mob_mg_{name}"]] = module.MOBILITY_MG[piece]
        weights[EXTRA_INDEX[f"mob_eg_{name}"]] = module.MOBILITY_EG[piece]
    return weights""",
    "current-parameters",
)

# Emit the fitted values in the shape agent.py declares them.
swap(
    """        handle.write(f"TUNED_PASSED_MG = {tuple(passed_mg)}\\n")
        handle.write(f"TUNED_PASSED_EG = {tuple(passed_eg)}\\n")""",
    """        handle.write(f"TUNED_PASSED_MG = {tuple(passed_mg)}\\n")
        handle.write(f"TUNED_PASSED_EG = {tuple(passed_eg)}\\n")
        mob_mg = [0, 0] + [scalars[f"mob_mg_{n}"] for n in MOB_NAMES] + [0]
        mob_eg = [0, 0] + [scalars[f"mob_eg_{n}"] for n in MOB_NAMES] + [0]
        handle.write(f"TUNED_MOBILITY_MG = {tuple(mob_mg)}\\n")
        handle.write(f"TUNED_MOBILITY_EG = {tuple(mob_eg)}\\n")""",
    "write-tables",
)

swap(
    "PHASE_WEIGHT = (0, 0, 1, 1, 2, 4, 0)",
    'MOB_NAMES = ("knight", "bishop", "rook", "queen")\n\n'
    "PHASE_WEIGHT = (0, 0, 1, 1, 2, 4, 0)",
    "mob-names",
)

p.write_text(s, encoding="utf-8")
print("tune.py now carries the mobility term")
print("EXTRA count is now", s.count('EXTRA += ') , "groups plus the base list")
