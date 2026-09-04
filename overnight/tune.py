"""Tune the evaluation on your own data (Texel's method).

The evaluation is linear in its parameters - the tapered score is just a weighted sum of
piece-square entries - so fitting it is a sparse least-squares problem rather than a slow
black-box search. That is the same procedure that produced the PeSTO tables the engine
currently borrows, so running this replaces them with numbers that are yours.

    python tune.py --check                 # verify feature extraction, tune nothing
    python tune.py --data data --out tuned_tables.py --steps 4000

The --check pass is not optional decoration: it confirms that the features reproduce
agent.py's evaluation exactly. If that fails, anything the tuner produces is meaningless.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import chess
import numpy as np
import scipy.sparse as sp

PIECES = 6
PST_PARAMS = PIECES * 64          # one combined table per piece type, material folded in
EXTRA = [
    "bishop_pair_mg", "bishop_pair_eg", "rook_open_mg", "rook_semi_mg",
    "isolated_mg", "isolated_eg", "doubled_mg", "doubled_eg", "shelter_mg",
]
EXTRA += [f"passed_mg_{rank}" for rank in range(1, 7)]
EXTRA += [f"passed_eg_{rank}" for rank in range(1, 7)]
EXTRA += [f"mob_mg_{name}" for name in ("knight", "bishop", "rook", "queen")]
EXTRA += [f"mob_eg_{name}" for name in ("knight", "bishop", "rook", "queen")]
TOTAL = PST_PARAMS * 2 + len(EXTRA)

MG_BASE = 0
EG_BASE = PST_PARAMS
EXTRA_BASE = PST_PARAMS * 2
EXTRA_INDEX = {name: EXTRA_BASE + i for i, name in enumerate(EXTRA)}

MOB_NAMES = ("knight", "bishop", "rook", "queen")

PHASE_WEIGHT = (0, 0, 1, 1, 2, 4, 0)
TOTAL_PHASE = 24


def features(board: chess.Board) -> dict[int, float]:
    """White-relative feature coefficients for one position.

    Mirrors agent.py's evaluate() term for term. Midgame parameters are scaled by
    phase/24 and endgame ones by (24 - phase)/24, which is the tapering written out.
    """
    raw: dict[int, float] = {}
    phase = 0
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]

    def bump(index: int, amount: float) -> None:
        raw[index] = raw.get(index, 0.0) + amount

    for piece in range(1, 7):
        weight = PHASE_WEIGHT[piece]
        for square in chess.scan_forward(board.pieces_mask(piece, chess.WHITE)):
            bump((piece - 1) * 64 + (square ^ 56), 1.0)
            phase += weight
        for square in chess.scan_forward(board.pieces_mask(piece, chess.BLACK)):
            bump((piece - 1) * 64 + square, -1.0)
            phase += weight
    phase = min(phase, TOTAL_PHASE)

    white_pawns = board.pawns & white
    black_pawns = board.pawns & black
    white_files = [0] * 8
    black_files = [0] * 8
    for square in chess.scan_forward(white_pawns):
        white_files[chess.square_file(square)] += 1
    for square in chess.scan_forward(black_pawns):
        black_files[chess.square_file(square)] += 1

    extra_mg: dict[int, float] = {}
    extra_eg: dict[int, float] = {}

    def bump_mg(name: str, amount: float) -> None:
        extra_mg[EXTRA_INDEX[name]] = extra_mg.get(EXTRA_INDEX[name], 0.0) + amount

    def bump_eg(name: str, amount: float) -> None:
        extra_eg[EXTRA_INDEX[name]] = extra_eg.get(EXTRA_INDEX[name], 0.0) + amount

    # Mobility. attacks_mask stops a ray at the first blocker and includes that square,
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
        bump_mg("bishop_pair_mg", 1.0)
        bump_eg("bishop_pair_eg", 1.0)
    if chess.popcount(board.bishops & black) > 1:
        bump_mg("bishop_pair_mg", -1.0)
        bump_eg("bishop_pair_eg", -1.0)

    for square in chess.scan_forward(board.rooks & white):
        file = chess.square_file(square)
        if white_files[file] == 0:
            bump_mg("rook_semi_mg" if black_files[file] else "rook_open_mg", 1.0)
    for square in chess.scan_forward(board.rooks & black):
        file = chess.square_file(square)
        if black_files[file] == 0:
            bump_mg("rook_semi_mg" if white_files[file] else "rook_open_mg", -1.0)

    for square in chess.scan_forward(white_pawns):
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        blocked = any(
            chess.square_file(other) in range(max(0, file - 1), min(7, file + 1) + 1)
            and chess.square_rank(other) > rank
            for other in chess.scan_forward(black_pawns)
        )
        if not blocked and 1 <= rank <= 6:
            bump_mg(f"passed_mg_{rank}", 1.0)
            bump_eg(f"passed_eg_{rank}", 1.0)
        left = white_files[file - 1] if file > 0 else 0
        right = white_files[file + 1] if file < 7 else 0
        if not left and not right:
            bump_mg("isolated_mg", -1.0)
            bump_eg("isolated_eg", -1.0)
        if any(chess.square_file(o) == file and chess.square_rank(o) > rank
               for o in chess.scan_forward(white_pawns)):
            bump_mg("doubled_mg", -1.0)
            bump_eg("doubled_eg", -1.0)
    for square in chess.scan_forward(black_pawns):
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        relative = 7 - rank
        blocked = any(
            chess.square_file(other) in range(max(0, file - 1), min(7, file + 1) + 1)
            and chess.square_rank(other) < rank
            for other in chess.scan_forward(white_pawns)
        )
        if not blocked and 1 <= relative <= 6:
            bump_mg(f"passed_mg_{relative}", -1.0)
            bump_eg(f"passed_eg_{relative}", -1.0)
        left = black_files[file - 1] if file > 0 else 0
        right = black_files[file + 1] if file < 7 else 0
        if not left and not right:
            bump_mg("isolated_mg", 1.0)
            bump_eg("isolated_eg", 1.0)
        if any(chess.square_file(o) == file and chess.square_rank(o) < rank
               for o in chess.scan_forward(black_pawns)):
            bump_mg("doubled_mg", 1.0)
            bump_eg("doubled_eg", 1.0)

    for colour, files, sign in ((chess.WHITE, white_files, -1.0),
                                (chess.BLACK, black_files, 1.0)):
        king = board.king(colour)
        if king is None:
            continue
        file = chess.square_file(king)
        missing = 0
        if files[file] == 0:
            missing += 1
        if file > 0 and files[file - 1] == 0:
            missing += 1
        if file < 7 and files[file + 1] == 0:
            missing += 1
        bump_mg("shelter_mg", sign * missing)

    mg_scale = phase / TOTAL_PHASE
    eg_scale = (TOTAL_PHASE - phase) / TOTAL_PHASE
    out: dict[int, float] = {}
    for index, value in raw.items():
        out[MG_BASE + index] = value * mg_scale
        out[EG_BASE + index] = value * eg_scale
    for index, value in extra_mg.items():
        out[index] = out.get(index, 0.0) + value * mg_scale
    for index, value in extra_eg.items():
        out[index] = out.get(index, 0.0) + value * eg_scale
    return out


def current_parameters(agent_path: str) -> np.ndarray:
    """The engine's parameters as a vector in the same layout the features use."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ref_agent", agent_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ref_agent"] = module
    spec.loader.exec_module(module)

    weights = np.zeros(TOTAL, dtype=np.float64)
    for piece in range(1, 7):
        for index in range(64):
            weights[MG_BASE + (piece - 1) * 64 + index] = module.MG_TABLE[piece][1][index ^ 56]
            weights[EG_BASE + (piece - 1) * 64 + index] = module.EG_TABLE[piece][1][index ^ 56]
    weights[EXTRA_INDEX["bishop_pair_mg"]] = module.BISHOP_PAIR_MG
    weights[EXTRA_INDEX["bishop_pair_eg"]] = module.BISHOP_PAIR_EG
    weights[EXTRA_INDEX["rook_open_mg"]] = module.ROOK_OPEN_FILE
    weights[EXTRA_INDEX["rook_semi_mg"]] = module.ROOK_SEMI_OPEN_FILE
    weights[EXTRA_INDEX["isolated_mg"]] = module.ISOLATED_MG
    weights[EXTRA_INDEX["isolated_eg"]] = module.ISOLATED_EG
    weights[EXTRA_INDEX["doubled_mg"]] = module.DOUBLED_MG
    weights[EXTRA_INDEX["doubled_eg"]] = module.DOUBLED_EG
    weights[EXTRA_INDEX["shelter_mg"]] = module.SHELTER_PENALTY
    for rank in range(1, 7):
        weights[EXTRA_INDEX[f"passed_mg_{rank}"]] = module.PASSED_MG[rank]
        weights[EXTRA_INDEX[f"passed_eg_{rank}"]] = module.PASSED_EG[rank]
    for piece, name in ((2, "knight"), (3, "bishop"), (4, "rook"), (5, "queen")):
        weights[EXTRA_INDEX[f"mob_mg_{name}"]] = module.MOBILITY_MG[piece]
        weights[EXTRA_INDEX[f"mob_eg_{name}"]] = module.MOBILITY_EG[piece]
    return weights


def check(agent_path: str, samples: int = 300) -> bool:
    """Do the features reproduce the engine's evaluation? Nothing else matters if not."""
    import importlib.util
    import random

    spec = importlib.util.spec_from_file_location("chk_agent", agent_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["chk_agent"] = module
    spec.loader.exec_module(module)

    weights = current_parameters(agent_path)
    random.seed(0)
    worst = 0.0
    for _ in range(samples):
        board = chess.Board()
        for _ in range(random.randint(1, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(random.choice(moves))
        if board.is_game_over():
            continue
        searcher = module.Searcher(1e18, board.turn)
        searcher.seed(board)
        # Compare white-relative, before the tempo bonus and fifty-move fade, which the
        # tuner holds fixed.
        engine_side = searcher.evaluate(board) - module.TEMPO
        engine_white = engine_side if board.turn == chess.WHITE else -engine_side
        engine_white = engine_white * 200 / (200 - board.halfmove_clock)
        predicted = sum(weights[i] * c for i, c in features(board).items())
        worst = max(worst, abs(predicted - engine_white))
    # A couple of centipawns of slack is floor division, not a feature error: the engine
    # floors twice (tapering, and the fifty-move fade) and neither is invertible. Where
    # tapering has no remainder the two agree exactly.
    print(f"feature check: worst disagreement {worst:.2f} cp over {samples} positions")
    return worst < 3.0


def load(data_dir: str, limit: int) -> tuple[sp.csr_matrix, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    targets: list[float] = []
    seen = 0
    for path in sorted(glob.glob(os.path.join(data_dir, "part*.csv"))):
        with open(path) as handle:
            for line in handle:
                if seen >= limit:
                    break
                try:
                    fen, score_text, result_text = line.rsplit(",", 2)
                    board = chess.Board(fen)
                except ValueError:
                    continue
                score = float(score_text)
                if board.turn == chess.BLACK:
                    score = -score
                result = float(result_text)
                if board.turn == chess.BLACK:
                    result = 1.0 - result
                win = 1.0 / (1.0 + math.exp(-score / 400.0))
                for index, coefficient in features(board).items():
                    rows.append(seen)
                    cols.append(index)
                    values.append(coefficient)
                targets.append(0.7 * win + 0.3 * result)
                seen += 1
                if seen % 100_000 == 0:
                    print(f"  featurised {seen}", flush=True)
        if seen >= limit:
            break
    matrix = sp.csr_matrix(
        (np.asarray(values), (np.asarray(rows), np.asarray(cols))),
        shape=(seen, TOTAL), dtype=np.float64,
    )
    return matrix, np.asarray(targets)


def fit_scale(inputs: sp.csr_matrix, targets: np.ndarray, weights: np.ndarray) -> float:
    """Pick the sigmoid scale K that best fits the current parameters."""
    predictions = inputs @ weights
    best_k, best_loss = 400.0, float("inf")
    for k in np.arange(100.0, 700.0, 10.0):
        loss = float(np.mean((1.0 / (1.0 + np.exp(-predictions / k)) - targets) ** 2))
        if loss < best_loss:
            best_loss, best_k = loss, float(k)
    return best_k


def material_only() -> np.ndarray:
    """A starting point that contains nothing anyone else wrote.

    The competition requires that moves come from code you wrote and any model you ship
    is one you trained. The PeSTO tables the engine has been using are neither: they are
    768 numbers hand-tuned by Ronald Friederich for RofChade and published openly. They
    are fine to learn from and awkward to ship.

    So this starts from the only chess knowledge that belongs to nobody - a pawn is worth
    about one, a rook about five - and lets the data supply every square-by-square
    deviation. Textbook material values, no piece-square structure, no structural terms.
    """
    values = (100, 320, 330, 500, 900, 0)      # pawn, knight, bishop, rook, queen, king
    weights = np.zeros(TOTAL, dtype=np.float64)
    for piece in range(1, 7):
        for index in range(64):
            weights[MG_BASE + (piece - 1) * 64 + index] = values[piece - 1]
            weights[EG_BASE + (piece - 1) * 64 + index] = values[piece - 1]
    return weights


def tune(data_dir: str, agent_path: str, out_path: str, steps: int, limit: int,
         batch: int, anchor: float, from_material: bool = False) -> None:
    started = time.monotonic()
    print("featurising positions...", flush=True)
    inputs, targets = load(data_dir, limit)
    count = inputs.shape[0]
    print(f"{count} positions, {TOTAL} parameters", flush=True)

    if from_material:
        print("starting from material only - no borrowed piece-square values", flush=True)
        weights = material_only()
    else:
        weights = current_parameters(agent_path)
    start = weights.copy()
    scale = fit_scale(inputs, targets, weights)
    baseline = float(np.mean(
        (1.0 / (1.0 + np.exp(-(inputs @ weights) / scale)) - targets) ** 2))
    print(f"sigmoid scale K = {scale:.0f}   starting loss {baseline:.6f}", flush=True)

    rng = np.random.default_rng(0)
    order = rng.permutation(count)
    split = int(count * 0.95)
    train_ids, valid_ids = order[:split], order[split:]
    valid_inputs = inputs[valid_ids]
    valid_targets = targets[valid_ids]

    momentum = np.zeros_like(weights)
    velocity = np.zeros_like(weights)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    rate = 2.0
    best = float("inf")
    best_weights = weights.copy()

    for step in range(1, steps + 1):
        ids = rng.choice(train_ids, size=min(batch, len(train_ids)), replace=False)
        block = inputs[ids]
        block_targets = targets[ids]
        raw = block @ weights
        predicted = 1.0 / (1.0 + np.exp(-raw / scale))
        error = predicted - block_targets
        gradient = block.T @ (2.0 * error * predicted * (1.0 - predicted) / scale)
        gradient /= len(ids)

        momentum = beta1 * momentum + (1 - beta1) * gradient
        velocity = beta2 * velocity + (1 - beta2) * gradient ** 2
        corrected1 = momentum / (1 - beta1 ** step)
        corrected2 = velocity / (1 - beta2 ** step)
        weights -= rate * corrected1 / (np.sqrt(corrected2) + epsilon)
        # Decoupled pull toward the starting values, in parameter units rather than
        # gradient units. Piece-square tables can express most of what the structural
        # terms express, so unanchored the redundant ones drift somewhere meaningless
        # while the loss barely notices.
        weights -= anchor * (weights - start)

        if step % 250 == 0 or step == steps:
            predicted = 1.0 / (1.0 + np.exp(-(valid_inputs @ weights) / scale))
            loss = float(np.mean((predicted - valid_targets) ** 2))
            if loss < best:
                best = loss
                best_weights = weights.copy()
            print(f"  step {step:5}/{steps}  validation {loss:.6f}"
                  f"  ({time.monotonic() - started:.0f}s)", flush=True)
        if step % 1500 == 0:
            rate *= 0.5

    improvement = (baseline - best) / baseline * 100
    print(f"best validation {best:.6f} (started {baseline:.6f}, {improvement:+.1f}%)")
    write_tables(best_weights, out_path)
    print(f"wrote {out_path}")


def write_tables(weights: np.ndarray, path: str) -> None:
    """Emit tables in exactly the shape agent.py declares them."""
    def block(name: str, base: int) -> str:
        lines = [f"{name} = ("]
        for piece in range(6):
            values = [round(float(weights[base + piece * 64 + i])) for i in range(64)]
            lines.append("    (")
            for rank in range(8):
                row = ", ".join(f"{v:5}" for v in values[rank * 8:(rank + 1) * 8])
                lines.append(f"        {row},")
            lines.append("    ),")
        lines.append(")")
        return "\n".join(lines)

    scalars = {name: round(float(weights[EXTRA_INDEX[name]])) for name in EXTRA}
    passed_mg = [0] + [scalars[f"passed_mg_{r}"] for r in range(1, 7)] + [0]
    passed_eg = [0] + [scalars[f"passed_eg_{r}"] for r in range(1, 7)] + [0]

    with open(path, "w") as handle:
        handle.write('"""Evaluation parameters fitted to our own games with Texel\'s method.\n\n'
                     'Generated by tune.py. Tables are written rank 8 first, in the same\n'
                     'orientation agent.py reads them: white at [square ^ 56], black at\n'
                     '[square]. Material value is folded into the tables.\n"""\n\n')
        handle.write(block("TUNED_MG", MG_BASE) + "\n\n")
        handle.write(block("TUNED_EG", EG_BASE) + "\n\n")
        handle.write(f"TUNED_BISHOP_PAIR_MG = {scalars['bishop_pair_mg']}\n")
        handle.write(f"TUNED_BISHOP_PAIR_EG = {scalars['bishop_pair_eg']}\n")
        handle.write(f"TUNED_ROOK_OPEN = {scalars['rook_open_mg']}\n")
        handle.write(f"TUNED_ROOK_SEMI_OPEN = {scalars['rook_semi_mg']}\n")
        handle.write(f"TUNED_ISOLATED_MG = {scalars['isolated_mg']}\n")
        handle.write(f"TUNED_ISOLATED_EG = {scalars['isolated_eg']}\n")
        handle.write(f"TUNED_DOUBLED_MG = {scalars['doubled_mg']}\n")
        handle.write(f"TUNED_DOUBLED_EG = {scalars['doubled_eg']}\n")
        handle.write(f"TUNED_SHELTER = {scalars['shelter_mg']}\n")
        handle.write(f"TUNED_PASSED_MG = {tuple(passed_mg)}\n")
        handle.write(f"TUNED_PASSED_EG = {tuple(passed_eg)}\n")
        mob_mg = [0, 0] + [scalars[f"mob_mg_{n}"] for n in MOB_NAMES] + [0]
        mob_eg = [0, 0] + [scalars[f"mob_eg_{n}"] for n in MOB_NAMES] + [0]
        handle.write(f"TUNED_MOBILITY_MG = {tuple(mob_mg)}\n")
        handle.write(f"TUNED_MOBILITY_EG = {tuple(mob_eg)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--agent", default="reference_agent.py")
    parser.add_argument("--out", default="tuned_tables.py")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=3_000_000)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--anchor", type=float, default=2.5e-4,
                        help="per-step pull toward the starting values, in parameter units")
    parser.add_argument("--from-material", action="store_true",
                        help="ignore the current tables and fit every piece-square value "
                             "from the data, so nothing in the shipped file traces to "
                             "another engine's published numbers")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        sys.exit(0 if check(args.agent) else 1)
    if not check(args.agent):
        raise SystemExit("feature extraction disagrees with the engine; refusing to tune")
    tune(args.data, args.agent, args.out, args.steps, args.limit, args.batch,
         args.anchor, args.from_material)


if __name__ == "__main__":
    main()
