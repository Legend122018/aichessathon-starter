"""Is the trained net more accurate than the evaluation it would replace?

This is the gate the net has to pass before any of it goes near agent.py. v5 failed it:
217cp mean error against the classical evaluation's 193cp, and it was slower too. Speed is
now settled - probe.py says 768-128-16-1 keeps 95% of the node rate - so accuracy decides.

Both are scored against the same Stockfish labels, on the validation rows train.py held
out, so neither has seen them fitted.

    python overnight/net_accuracy.py overnight/net128.safetensors
"""

from __future__ import annotations

import glob
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

import chess
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCALE = 400.0
FEATURES = 768


def read_safetensors(path: str) -> dict[str, np.ndarray]:
    with open(path, "rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
        blob = handle.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        out[name] = np.frombuffer(blob[start:end], dtype=np.float32).reshape(meta["shape"])
    return out


def active_features(board: chess.Board, mover: chess.Color) -> list[int]:
    """Identical to train.py's feature map, or the weights mean nothing."""
    active: list[int] = []
    for piece in range(1, 7):
        for colour in (chess.WHITE, chess.BLACK):
            base = (0 if colour == mover else 384) + (piece - 1) * 64
            for square in chess.scan_forward(board.pieces_mask(piece, colour)):
                active.append(base + (square if mover == chess.WHITE else square ^ 56))
    return active


def load_agent(path: str):
    spec = importlib.util.spec_from_file_location("acc_agent", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["acc_agent"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    net_path = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "net128.safetensors")
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else 25000

    tensors = read_safetensors(net_path)
    w1, b1 = tensors["w1"], tensors["b1"]
    w2, b2 = tensors["w2"], tensors["b2"]
    w3, b3 = tensors["w3"], tensors["b3"]
    print(f"net {w1.shape[0]}-{w1.shape[1]}-{w2.shape[1]}-1 "
          f"({sum(t.size for t in tensors.values()):,} parameters)")

    # train.py's validation split, reproduced exactly.
    total = len(np.load(str(ROOT / "overnight/data/targets.npy")))
    rng = np.random.default_rng(0)
    order = rng.permutation(total)
    held_out = set(order[int(total * 0.97):].tolist())
    print(f"{total:,} positions, {len(held_out):,} held out of training")

    rows: list[tuple[str, float]] = []
    index = 0
    for path in sorted(glob.glob(str(ROOT / "overnight/data/part*.csv"))):
        with open(path) as handle:
            for line in handle:
                if index in held_out and len(rows) < sample:
                    try:
                        fen, score_text, _ = line.rsplit(",", 2)
                        rows.append((fen, float(score_text)))
                    except ValueError:
                        pass
                index += 1
        if len(rows) >= sample:
            break
    print(f"scoring {len(rows):,} held-out positions\n")

    agent = load_agent(str(ROOT / "agent.py"))
    tempo = agent.TEMPO

    net_err = 0.0
    cls_err = 0.0
    net_sq = 0.0
    cls_sq = 0.0
    counted = 0
    for fen, label in rows:
        board = chess.Board(fen)

        acc = b1.copy()
        for i in active_features(board, board.turn):
            acc += w1[i]
        hidden = np.clip(acc, 0.0, 1.0)
        mid = np.clip(hidden @ w2 + b2, 0.0, 1.0)
        raw = float((mid @ w3 + b3).ravel()[0])
        win = 1.0 / (1.0 + math.exp(-raw))
        win = min(max(win, 1e-6), 1 - 1e-6)
        net_cp = -SCALE * math.log(1.0 / win - 1.0)

        jb, jst = agent.jit_new_state(fen)
        cls_cp = float(agent.jit_evaluate(jb, jst)) - tempo

        # Labels are from the mover's point of view, same as both evaluations.
        net_err += abs(net_cp - label)
        cls_err += abs(cls_cp - label)
        target = 1.0 / (1.0 + math.exp(-label / SCALE))
        net_sq += (win - target) ** 2
        cls_sq += (1.0 / (1.0 + math.exp(-cls_cp / SCALE)) - target) ** 2
        counted += 1

    print(f"{'evaluation':<24}{'mean error':>13}{'win-prob MSE':>15}")
    print(f"{'classical (shipping)':<24}{cls_err / counted:11.1f}cp{cls_sq / counted:15.5f}")
    print(f"{'net 768-128-16-1':<24}{net_err / counted:11.1f}cp{net_sq / counted:15.5f}")
    print()
    if net_err < cls_err:
        print("  The net is the more accurate evaluation. Worth integrating.")
    else:
        gap = (net_err - cls_err) / counted
        print(f"  The net is {gap:.0f}cp worse per position. Not worth integrating:")
        print("  it has to be more accurate to pay for any of its cost.")


if __name__ == "__main__":
    main()
