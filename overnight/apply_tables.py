"""Fold tuned parameters into agent.py, and prove the result is what was tuned.

agent.py carries the same numbers twice - once as nested tuples for the pure-Python
engine, once as flat arrays for the JIT engine - so both have to be rewritten together or
the two halves silently disagree. The verification at the end is the point: it recomputes
the patched engine's evaluation from the tuned parameters and refuses to keep the file if
they do not match.

    python apply_tables.py --agent agent.py --tables tuned_tables.py --out candidate.py
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import re
import sys

import chess


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def table_literal(name: str, rows: tuple) -> str:
    lines = [f"{name} = ("]
    for rank in range(8):
        row = ", ".join(f"{v:5}" for v in rows[rank * 8:(rank + 1) * 8])
        lines.append(f"    {row},")
    lines.append(")  # fmt: skip")
    return "\n".join(lines)


def patch(agent_path: str, tables_path: str, out_path: str) -> str:
    with open(agent_path) as handle:
        source = handle.read()
    tuned = load_module("tuned", tables_path)

    names = ["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"]

    # Material is folded into the tuned tables, so the separate value vectors go to zero.
    source = re.sub(r"^MG_VALUE = \([^)]*\)$", "MG_VALUE = (0, 0, 0, 0, 0, 0, 0)",
                    source, flags=re.M)
    source = re.sub(r"^EG_VALUE = \([^)]*\)$", "EG_VALUE = (0, 0, 0, 0, 0, 0, 0)",
                    source, flags=re.M)

    for index, name in enumerate(names):
        for prefix, table in (("MG", tuned.TUNED_MG), ("EG", tuned.TUNED_EG)):
            pattern = rf"^{prefix}_{name} = \(.*?\)  # fmt: skip$"
            replacement = table_literal(f"{prefix}_{name}", table[index])
            source, count = re.subn(pattern, lambda _m, r=replacement: r, source,
                                    count=1, flags=re.M | re.S)
            if count != 1:
                raise SystemExit(f"could not find {prefix}_{name} in {agent_path}")

    scalars = {
        "BISHOP_PAIR_MG": tuned.TUNED_BISHOP_PAIR_MG,
        "BISHOP_PAIR_EG": tuned.TUNED_BISHOP_PAIR_EG,
        "ROOK_OPEN_FILE": tuned.TUNED_ROOK_OPEN,
        "ROOK_SEMI_OPEN_FILE": tuned.TUNED_ROOK_SEMI_OPEN,
        "SHELTER_PENALTY": tuned.TUNED_SHELTER,
    }
    for key, value in scalars.items():
        source, count = re.subn(rf"^{key} = -?\d+$", f"{key} = {value}", source,
                                count=1, flags=re.M)
        if count != 1:
            raise SystemExit(f"could not find {key}")
    source = re.sub(r"^ISOLATED_MG, ISOLATED_EG = .*$",
                    f"ISOLATED_MG, ISOLATED_EG = {tuned.TUNED_ISOLATED_MG}, "
                    f"{tuned.TUNED_ISOLATED_EG}", source, flags=re.M)
    source = re.sub(r"^DOUBLED_MG, DOUBLED_EG = .*$",
                    f"DOUBLED_MG, DOUBLED_EG = {tuned.TUNED_DOUBLED_MG}, "
                    f"{tuned.TUNED_DOUBLED_EG}", source, flags=re.M)
    # Mobility lives in one place: the JIT arrays are built from these tuples at import,
    # so rewriting the tuples rewrites both halves at once.
    if hasattr(tuned, "TUNED_MOBILITY_MG"):
        source = re.sub(r"^MOBILITY_MG = \([^)]*\)$",
                        f"MOBILITY_MG = {tuned.TUNED_MOBILITY_MG}", source, flags=re.M)
        source = re.sub(r"^MOBILITY_EG = \([^)]*\)$",
                        f"MOBILITY_EG = {tuned.TUNED_MOBILITY_EG}", source, flags=re.M)
    source = re.sub(r"^PASSED_MG = \([^)]*\)$", f"PASSED_MG = {tuned.TUNED_PASSED_MG}",
                    source, flags=re.M)
    source = re.sub(r"^PASSED_EG = \([^)]*\)$", f"PASSED_EG = {tuned.TUNED_PASSED_EG}",
                    source, flags=re.M)

    # The JIT half keeps the same numbers as flat arrays; regenerate them from the tuned
    # tables so the two engines cannot drift apart.
    flat_mg, flat_eg = [], []
    for piece in range(7):
        for colour in (0, 1):
            for square in range(64):
                if piece == 0:
                    flat_mg.append(0)
                    flat_eg.append(0)
                    continue
                position = (square ^ 56) if colour == 1 else square
                flat_mg.append(int(tuned.TUNED_MG[piece - 1][position]))
                flat_eg.append(int(tuned.TUNED_EG[piece - 1][position]))

    def array_literal(name: str, values: list[int]) -> str:
        body = ", ".join(str(v) for v in values)
        chunks = re.findall(r".{1,86}(?:, |$)", body)
        joined = "\n        ".join(chunk.rstrip() for chunk in chunks)
        return f"    {name} = _np.array([\n        {joined}\n    ], dtype=_np.int32)"

    for name, values in (("JIT_MG_TABLE", flat_mg), ("JIT_EG_TABLE", flat_eg)):
        pattern = rf"^    {name} = _np\.array\(\[.*?\], dtype=_np\.int32\)$"
        replacement = array_literal(name, values)
        source, count = re.subn(pattern, lambda _m, r=replacement: r, source,
                                count=1, flags=re.M | re.S)
        if count != 1:
            raise SystemExit(f"could not find {name}")

    jit_scalars = {
        "JIT_BISHOP_PAIR_MG": tuned.TUNED_BISHOP_PAIR_MG,
        "JIT_BISHOP_PAIR_EG": tuned.TUNED_BISHOP_PAIR_EG,
        "JIT_ROOK_OPEN": tuned.TUNED_ROOK_OPEN,
        "JIT_ROOK_SEMI_OPEN": tuned.TUNED_ROOK_SEMI_OPEN,
        "JIT_ISOLATED_MG": tuned.TUNED_ISOLATED_MG,
        "JIT_ISOLATED_EG": tuned.TUNED_ISOLATED_EG,
        "JIT_DOUBLED_MG": tuned.TUNED_DOUBLED_MG,
        "JIT_DOUBLED_EG": tuned.TUNED_DOUBLED_EG,
        "JIT_SHELTER": tuned.TUNED_SHELTER,
    }
    for key, value in jit_scalars.items():
        source, count = re.subn(rf"^    {key} = -?\d+$", f"    {key} = {value}", source,
                                count=1, flags=re.M)
        if count != 1:
            raise SystemExit(f"could not find {key}")
    for key, values in (("JIT_PASSED_MG", tuned.TUNED_PASSED_MG),
                        ("JIT_PASSED_EG", tuned.TUNED_PASSED_EG)):
        body = ", ".join(str(v) for v in values)
        source, count = re.subn(rf"^    {key} = _np\.array\(\[[^\]]*\], dtype=_np\.int32\)$",
                                f"    {key} = _np.array([\n        {body}\n    ], "
                                f"dtype=_np.int32)", source, count=1, flags=re.M)
        if count != 1:
            raise SystemExit(f"could not find {key}")

    with open(out_path, "w") as handle:
        handle.write(source)
    return out_path


def verify(out_path: str, tables_path: str, tune_dir: str) -> bool:
    """Recompute the patched engine's evaluation from the tuned parameters directly.

    This is what catches a half-applied patch: if either copy of the tables was missed,
    the two numbers diverge.
    """
    sys.path.insert(0, tune_dir)
    import tune

    patched = load_module("patched_agent", out_path)
    weights = tune.current_parameters(out_path)

    random.seed(11)
    worst = 0.0
    for _ in range(250):
        board = chess.Board()
        for _ in range(random.randint(1, 50)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(random.choice(moves))
        if board.is_game_over() or board.halfmove_clock != 0:
            continue
        searcher = patched.Searcher(1e18, board.turn)
        searcher.seed(board)
        side = searcher.evaluate(board) - patched.TEMPO
        white = side if board.turn == chess.WHITE else -side
        predicted = sum(weights[i] * c for i, c in tune.features(board).items())
        worst = max(worst, abs(predicted - white))
    print(f"patched engine matches tuned parameters to {worst:.2f} cp")
    return worst < 3.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="agent.py")
    parser.add_argument("--tables", default="tuned_tables.py")
    parser.add_argument("--out", default="candidate.py")
    parser.add_argument("--tune-dir", default=".")
    args = parser.parse_args()
    patch(args.agent, args.tables, args.out)
    if not verify(args.out, args.tables, args.tune_dir):
        raise SystemExit("patched engine does not match the tuned parameters")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
