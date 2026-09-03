"""Generate Stockfish-labelled training positions.

Cloud container only. The rules allow unrestricted training data, including positions
analysed by an existing engine; what they forbid is an engine choosing moves at run time.
Nothing here ships.

Each record is `fen,score,result` where score is centipawns from the side to move and
result is the game outcome from the same side's point of view.

    python gen_data.py --out /root/data/part0.csv --games 700 --seed 0
"""

from __future__ import annotations

import argparse
import random

import chess
import chess.engine

STOCKFISH = "/usr/games/stockfish"
CLAMP = 2000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--games", type=int, default=700)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=220)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    engine.configure({"Threads": 1, "Hash": 64})
    limit = chess.engine.Limit(depth=args.depth)

    written = 0
    with open(args.out, "w") as handle:
        for game in range(args.games):
            board = chess.Board()

            # A random opening keeps the data from collapsing onto one narrow line.
            for _ in range(rng.randint(6, 12)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
            if board.is_game_over():
                continue

            # Throw away openings that random play already decided.
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=6))
            except chess.engine.EngineError:
                continue
            opening = info["score"].relative.score(mate_score=10000)
            if opening is None or abs(opening) > 500:
                continue

            records: list[tuple[str, int, chess.Color]] = []
            while not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
                try:
                    info = engine.analyse(board, limit)
                except chess.engine.EngineError:
                    break
                score = info["score"].relative.score(mate_score=10000)
                best = info.get("pv", [None])[0]
                if best is None or score is None:
                    break
                # Quiet positions only: a score taken mid-exchange or in check teaches the
                # net to evaluate positions no static evaluation should be asked about.
                if not board.is_check() and not board.is_capture(best):
                    records.append((board.fen(), max(-CLAMP, min(CLAMP, score)), board.turn))
                board.push(best)

            outcome = board.outcome(claim_draw=True)
            if outcome is None:
                winner: chess.Color | None = None
            else:
                winner = outcome.winner

            for fen, score, turn in records:
                if winner is None:
                    result = 0.5
                else:
                    result = 1.0 if winner == turn else 0.0
                handle.write(f"{fen},{score},{result}\n")
                written += 1

            if game % 25 == 0:
                handle.flush()
                print(f"game {game}/{args.games}  positions {written}", flush=True)

    engine.quit()
    print(f"done: {written} positions -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
