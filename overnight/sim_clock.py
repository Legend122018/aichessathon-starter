"""Simulate the clock over a long game, for the shipped budget and a proposed one.

The rated games all show the same shape: about three seconds a move until roughly move 32,
then nothing left and the rest of the game played instantly. This reproduces that from the
formula alone, so a fix can be checked without playing a game.

    python overnight/sim_clock.py
"""

from __future__ import annotations

OVERHEAD_MS = 100
INCREMENT_MS = 500
PANIC_MS = 5000
MIN_MS = 15
BASE_MS = 120_000


def shipped(time_left_ms: float, move_number: int) -> float:
    reserve = OVERHEAD_MS + INCREMENT_MS * 0.5
    usable = max(0.0, time_left_ms - reserve)
    if time_left_ms <= PANIC_MS:
        return max(MIN_MS, usable * 0.15)
    moves_to_go = max(8, 45 - min(move_number, 40))
    soft = usable / moves_to_go + INCREMENT_MS * 0.8
    hard = min(soft * 3.5, usable * 0.5)
    return min(soft, max(hard, MIN_MS))


def proposed(time_left_ms: float, move_number: int) -> float:
    """Two changes. The moves-to-go floor rises from 8 to 24, so the share of the clock
    bet on one move stops growing as the game goes long; and the budget is capped at a
    small fraction of what is left, so the clock decays instead of hitting a wall."""
    reserve = OVERHEAD_MS + INCREMENT_MS * 0.5
    usable = max(0.0, time_left_ms - reserve)
    if time_left_ms <= PANIC_MS:
        return max(MIN_MS, usable * 0.15)
    moves_to_go = max(24, 48 - min(move_number, 24))
    soft = usable / moves_to_go + INCREMENT_MS * 0.8
    soft = min(soft, usable * 0.08)
    hard = min(soft * 3.5, usable * 0.25)
    return min(soft, max(hard, MIN_MS))


def run(budget, moves: int = 85) -> tuple[list[float], int | None, int]:
    clock = float(BASE_MS)
    trace, panic_at, instant = [], None, 0
    for move in range(1, moves + 1):
        spend = budget(clock, move)
        if clock <= PANIC_MS and panic_at is None:
            panic_at = move
        if spend < 200:
            instant += 1
        clock = clock - spend + INCREMENT_MS
        clock = max(0.0, clock)
        trace.append(clock)
    return trace, panic_at, instant


def main() -> None:
    for name, budget in (("shipped", shipped), ("proposed", proposed)):
        trace, panic_at, instant = run(budget)
        marks = [10, 20, 30, 40, 50, 60, 70, 85]
        line = "  ".join(f"m{m}:{trace[m - 1] / 1000:5.1f}s" for m in marks)
        print(f"\n{name:<9} {line}")
        print(f"{'':<9} panic at move {panic_at if panic_at else '-'}, "
              f"{instant} moves played without searching")
    print("\nA game that reaches move 85 is normal: the six rated games ran 44 to 85.\n")


if __name__ == "__main__":
    main()
