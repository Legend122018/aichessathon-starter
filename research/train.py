"""Train the evaluation network, in numpy.

Feature set is 768 inputs: 6 piece types x own/opponent x 64 squares, always from one
side's point of view, so a single set of weights serves both colours. That is also what
makes the accumulator cheap to keep up to date inside the search - a move touches two or
three features, not all of them.

Architecture is 768 -> HIDDEN -> NARROW -> 1 with clipped ReLU. The first layer is a plain
sum of the active feature rows, which is precisely the accumulator the engine maintains
incrementally at run time.

Weights are written as safetensors, which is a JSON header over raw little-endian floats,
so the agent reads them back with numpy alone and needs no extra dependency.

    python train.py --prep
    python train.py --fit --hidden 128 --narrow 16 --out /root/data/net128.safetensors
"""

from __future__ import annotations

import argparse
import array
import glob
import json
import math
import struct
import time

import chess
import numpy as np
import scipy.sparse as sp

FEATURES = 768
SCALE = 400.0
LAMBDA = 0.7
DATA = "/root/data"


def features(board: chess.Board, mover: chess.Color) -> list[int]:
    """Active input indices from `mover`'s point of view."""
    active: list[int] = []
    for piece in range(1, 7):
        for colour in (chess.WHITE, chess.BLACK):
            base = (0 if colour == mover else 384) + (piece - 1) * 64
            for square in chess.scan_forward(board.pieces_mask(piece, colour)):
                active.append(base + (square if mover == chess.WHITE else square ^ 56))
    return active


def prep() -> None:
    # Plain int lists would cost well over a gigabyte at this scale; array keeps the
    # same data in four bytes per entry.
    rows = array.array("i")
    cols = array.array("i")
    targets = array.array("f")
    seen = 0
    for path in sorted(glob.glob(f"{DATA}/part*.csv")):
        with open(path) as handle:
            for line in handle:
                try:
                    fen, score_text, result_text = line.rsplit(",", 2)
                    board = chess.Board(fen)
                except ValueError:
                    continue
                score = float(score_text)
                result = float(result_text)
                # Blend the engine score with how the game actually finished: the score is
                # sharper, the result is unbiased, and neither alone trains as well.
                win = 1.0 / (1.0 + math.exp(-score / SCALE))
                for index in features(board, board.turn):
                    rows.append(seen)
                    cols.append(index)
                targets.append(LAMBDA * win + (1.0 - LAMBDA) * result)
                seen += 1
                if seen % 100_000 == 0:
                    print(f"prepared {seen}", flush=True)

    matrix = sp.csr_matrix(
        (
            np.ones(len(rows), dtype=np.float32),
            (np.frombuffer(rows, dtype=np.int32), np.frombuffer(cols, dtype=np.int32)),
        ),
        shape=(seen, FEATURES),
        dtype=np.float32,
    )
    sp.save_npz(f"{DATA}/inputs.npz", matrix)
    np.save(f"{DATA}/targets.npy", np.frombuffer(targets, dtype=np.float32))
    print(f"prepared {seen} positions", flush=True)


def save_safetensors(path: str, tensors: dict[str, np.ndarray]) -> None:
    """Minimal safetensors writer: JSON header describing raw little-endian blocks."""
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, array in tensors.items():
        contiguous = np.ascontiguousarray(array, dtype=np.float32)
        payload = contiguous.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(contiguous.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        blobs.append(payload)
        offset += len(payload)
    encoded = json.dumps(header).encode("utf-8")
    padding = (8 - len(encoded) % 8) % 8
    encoded += b" " * padding
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for blob in blobs:
            handle.write(blob)


def fit(hidden: int, narrow: int, epochs: int, batch: int, out: str) -> None:
    inputs = sp.load_npz(f"{DATA}/inputs.npz").tocsr()
    targets = np.load(f"{DATA}/targets.npy")
    total = inputs.shape[0]

    rng = np.random.default_rng(0)
    order = rng.permutation(total)
    split = int(total * 0.97)
    train_ids, valid_ids = order[:split], order[split:]
    print(f"{total} positions: {len(train_ids)} train, {len(valid_ids)} validation")

    w1 = rng.uniform(-0.06, 0.06, (FEATURES, hidden)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    limit2 = math.sqrt(6.0 / (hidden + narrow))
    w2 = rng.uniform(-limit2, limit2, (hidden, narrow)).astype(np.float32)
    b2 = np.zeros(narrow, dtype=np.float32)
    limit3 = math.sqrt(6.0 / (narrow + 1))
    w3 = rng.uniform(-limit3, limit3, (narrow, 1)).astype(np.float32)
    b3 = np.zeros(1, dtype=np.float32)

    params = [w1, b1, w2, b2, w3, b3]
    moments = [np.zeros_like(p) for p in params]
    velocities = [np.zeros_like(p) for p in params]
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    step = 0

    def forward(block: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        acc = np.clip(block @ w1 + b1, 0.0, 1.0)
        mid = np.clip(acc @ w2 + b2, 0.0, 1.0)
        return acc, mid, (mid @ w3 + b3).ravel()

    valid_inputs = inputs[valid_ids]
    valid_targets = targets[valid_ids]
    best = float("inf")
    learning_rate = 3e-3

    for epoch in range(epochs):
        started = time.monotonic()
        shuffled = train_ids[rng.permutation(len(train_ids))]
        running = 0.0
        batches = 0
        for start in range(0, len(shuffled), batch):
            ids = shuffled[start : start + batch]
            block = inputs[ids]
            target = targets[ids]
            count = len(ids)

            acc, mid, raw = forward(block)
            predicted = 1.0 / (1.0 + np.exp(-raw))
            error = predicted - target
            running += float(np.mean(error * error))
            batches += 1

            d_raw = (2.0 / count) * error * predicted * (1.0 - predicted)
            d_raw = d_raw.reshape(-1, 1).astype(np.float32)
            g_w3 = mid.T @ d_raw
            g_b3 = d_raw.sum(axis=0)
            d_mid = (d_raw @ w3.T) * ((mid > 0.0) & (mid < 1.0))
            g_w2 = acc.T @ d_mid
            g_b2 = d_mid.sum(axis=0)
            d_acc = (d_mid @ w2.T) * ((acc > 0.0) & (acc < 1.0))
            g_w1 = block.T @ d_acc
            g_b1 = d_acc.sum(axis=0)

            step += 1
            correction1 = 1.0 - beta1**step
            correction2 = 1.0 - beta2**step
            for index, gradient in enumerate((g_w1, g_b1, g_w2, g_b2, g_w3, g_b3)):
                gradient = np.asarray(gradient, dtype=np.float32).reshape(
                    params[index].shape
                )
                moments[index] = beta1 * moments[index] + (1 - beta1) * gradient
                velocities[index] = beta2 * velocities[index] + (1 - beta2) * gradient**2
                adjusted = (moments[index] / correction1) / (
                    np.sqrt(velocities[index] / correction2) + epsilon
                )
                params[index] -= learning_rate * adjusted

        if (epoch + 1) % max(8, epochs // 4) == 0:
            learning_rate *= 0.45

        _, _, raw = forward(valid_inputs)
        predicted = 1.0 / (1.0 + np.exp(-raw))
        validation = float(np.mean((predicted - valid_targets) ** 2))
        print(
            f"epoch {epoch + 1}/{epochs}  train {running / max(1, batches):.5f}"
            f"  valid {validation:.5f}  {time.monotonic() - started:.0f}s",
            flush=True,
        )
        if validation < best:
            best = validation
            save_safetensors(
                out,
                {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3},
            )
    print(f"best validation {best:.5f} -> {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--narrow", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--out", default="/root/data/net.safetensors")
    args = parser.parse_args()
    if args.prep:
        prep()
    if args.fit:
        fit(args.hidden, args.narrow, args.epochs, args.batch, args.out)


if __name__ == "__main__":
    main()
