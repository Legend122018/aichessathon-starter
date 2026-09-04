"""Stop the engine spending its whole clock by move thirty.

All six rated games show the same shape: about three seconds a move until roughly move 32,
then the clock is gone and the rest of the game - between 8 and 46 moves - is played
without searching. The cause is in `_time_budget`:

    moves_to_go = max(8, 45 - min(board.fullmove_number, 40))

The floor of 8 means that late in a game the engine bets an eighth of everything it has
left on a single move, and the 400ms it adds on top is more than the 500ms increment it
earns back. The estimate assumes the game ends near move 45; the rated games ran 44 to 85.

Two changes: raise the floor to 24 so the share bet on one move stops growing, and cap the
budget at 8% of what is left so the clock decays smoothly instead of hitting a wall.
Simulated over an 85 move game this leaves 10.9s at move 70 where the shipped formula is
in panic mode from move 51.
"""
import pathlib

AGENT = pathlib.Path(__file__).resolve().parent / "exemplar_timefix" / "agent.py"
s = AGENT.read_text(encoding="utf-8")

old = """    moves_to_go = max(8, 45 - min(board.fullmove_number, 40))
    soft = usable / moves_to_go + INCREMENT_MS * 0.8
    hard = min(soft * 3.5, usable * 0.5)
    hard = max(hard, MIN_MS)
    soft = min(soft, hard)
    return soft, hard"""

new = """    # The floor here is what decides whether the clock lasts the game. At 8 the engine
    # bets an eighth of its remaining time on one move, and the 400ms added below is more
    # than the 500ms increment it earns, so from about move 40 the clock collapses and the
    # rest of the game is played without searching. Every rated game showed that shape.
    moves_to_go = max(24, 48 - min(board.fullmove_number, 24))
    soft = usable / moves_to_go + INCREMENT_MS * 0.8
    # And never bet more than this much of what is left, so the clock decays towards the
    # increment instead of running out: 10.9s still in hand at move 70, against 3.7s.
    soft = min(soft, usable * 0.08)
    hard = min(soft * 3.5, usable * 0.25)
    hard = max(hard, MIN_MS)
    soft = min(soft, hard)
    return soft, hard"""

assert s.count(old) == 1, f"anchor matched {s.count(old)} times"
AGENT.write_text(s.replace(old, new), encoding="utf-8")
print("time management patched in overnight/exemplar_timefix")
