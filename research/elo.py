"""Calibrate an agent against Stockfish at a fixed UCI_Elo.

Lives outside the project directory on purpose: Stockfish is a measuring stick and a
future source of training labels, never something the submission may touch or ship.

    python elo.py --agent /root/aichessathon-starter --elo 1800 --games 20
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import chess
import chess.engine

STOCKFISH = "/usr/games/stockfish"
PLY_CAP = 300


def load_agent(directory: str) -> ModuleType:
    """Import a fresh copy of `directory/agent.py`, as the platform does per game."""
    path = Path(directory) / "agent.py"
    spec = importlib.util.spec_from_file_location(f"agent_{time.monotonic_ns()}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Tally:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / max(1, self.games)

    def note(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def material(board: chess.Board, side: chess.Color) -> int:
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    return sum(v * len(board.pieces(p, side)) for p, v in values.items())


def play_game(
    agent_dir: str,
    engine: chess.engine.SimpleEngine,
    agent_white: bool,
    base_ms: int,
    inc_ms: int,
    sf_time: float,
) -> tuple[float, str]:
    """Return (score for the agent, reason). Mirrors the harness clock and loss rules."""
    agent = load_agent(agent_dir)
    board = chess.Board()
    clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    agent_colour = chess.WHITE if agent_white else chess.BLACK

    while True:
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            assert outcome is not None
            if outcome.winner is None:
                return 0.5, outcome.termination.name.lower()
            won = outcome.winner == agent_colour
            return (1.0 if won else 0.0), outcome.termination.name.lower()

        if board.ply() >= PLY_CAP:
            mine = material(board, agent_colour)
            theirs = material(board, not agent_colour)
            if mine > theirs + 2:
                return 1.0, "adjudication"
            if theirs > mine + 2:
                return 0.0, "adjudication"
            return 0.5, "adjudication"

        turn = board.turn
        if turn == agent_colour:
            start = time.monotonic()
            try:
                uci = agent.get_move(board.fen(), int(clock[turn]))
                move = chess.Move.from_uci(uci)
            except Exception:  # a crash is a loss, exactly as on the platform
                return 0.0, "crash"
            spent = (time.monotonic() - start) * 1000.0
            clock[turn] -= spent
            if clock[turn] < 0:
                return 0.0, "flag"
            if move not in board.legal_moves:
                return 0.0, "illegal"
        else:
            if sf_time > 0:
                limit = chess.engine.Limit(time=sf_time)
            else:
                limit = chess.engine.Limit(
                    white_clock=clock[chess.WHITE] / 1000.0,
                    black_clock=clock[chess.BLACK] / 1000.0,
                    white_inc=inc_ms / 1000.0,
                    black_inc=inc_ms / 1000.0,
                )
            try:
                result = engine.play(board, limit)
            except chess.engine.EngineError:
                return 0.5, "engine_error"
            if result.move is None:
                return 1.0, "opponent_resigned"
            move = result.move

        clock[turn] += inc_ms
        board.push(move)


def performance(score: float, opponent_elo: int, games: int) -> tuple[float, float]:
    """Performance rating and a rough 95% margin, in Elo."""
    clamped = min(max(score, 0.5 / (games + 1)), 1 - 0.5 / (games + 1))
    delta = -400.0 * math.log10(1.0 / clamped - 1.0)
    stderr = math.sqrt(max(clamped * (1 - clamped), 0.01) / games)
    low = min(max(clamped - 1.96 * stderr, 0.001), 0.999)
    high = min(max(clamped + 1.96 * stderr, 0.001), 0.999)
    span = (-400.0 * math.log10(1.0 / high - 1.0)) - (-400.0 * math.log10(1.0 / low - 1.0))
    return opponent_elo + delta, span / 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="/root/aichessathon-starter")
    parser.add_argument("--elo", type=int, default=1800)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--base-ms", type=int, default=20000)
    parser.add_argument("--inc-ms", type=int, default=500)
    parser.add_argument(
        "--sf-time",
        type=float,
        default=0.1,
        help="fixed seconds per move for Stockfish; 0 gives it the same clock instead",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    engine.configure(
        {"Threads": 1, "Hash": 16, "UCI_LimitStrength": True, "UCI_Elo": args.elo}
    )
    tally = Tally()
    try:
        for index in range(args.games):
            agent_white = index % 2 == 0
            score, reason = play_game(
                args.agent, engine, agent_white, args.base_ms, args.inc_ms, args.sf_time
            )
            tally.note(reason)
            if score == 1.0:
                tally.wins += 1
            elif score == 0.5:
                tally.draws += 1
            else:
                tally.losses += 1
            if not args.quiet:
                colour = "W" if agent_white else "B"
                print(
                    f"game {index + 1}/{args.games} ({colour}): {score} {reason}"
                    f"   running {tally.score:.1%}",
                    flush=True,
                )
    finally:
        engine.quit()

    rating, margin = performance(tally.score, args.elo, tally.games)
    print(
        f"\n{args.agent} vs Stockfish UCI_Elo {args.elo}"
        f" at {args.base_ms}ms+{args.inc_ms}ms over {tally.games} games"
    )
    print(f"+{tally.wins} ={tally.draws} -{tally.losses}, score {tally.score:.1%}")
    print(f"performance rating {rating:.0f} +/- {margin:.0f}")
    print("terminations:", ", ".join(f"{k} {v}" for k, v in sorted(tally.reasons.items())))


if __name__ == "__main__":
    main()
