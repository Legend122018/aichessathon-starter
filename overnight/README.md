# Overnight training system

Runs unattended. Generates its own training data, tunes the evaluation on it, and then
**plays matches to decide whether the change is real** — nothing is promoted because a
loss function went down.

## Setup (once)

**Windows**

```
pip install python-chess numpy scipy numba psutil
```

Then download Stockfish from https://stockfishchess.org/download/ and put
`stockfish.exe` in this folder. `psutil` is what lets the match runner pin each game to
its own cores on Windows; without it everything still works, just with noisier timings.

**Linux or WSL2**

```bash
sudo apt update && sudo apt install -y stockfish python3-pip
pip install python-chess numpy scipy numba
```

Stockfish is used only to label training positions and to measure strength. It is never
imported by the agent and never goes in the zip — the competition rules allow
engine-analysed positions as training data, and forbid an engine choosing moves at run
time.

## Run it

```
cd overnight
python overnight.py --hours 8 --workers 16
```

Wake up to `REPORT.md`. If a candidate was promoted, `champion.py` is the new engine:

```bash
cp champion.py ../agent.py
cd .. && uv run python -m harness.package     # builds submission.zip
```

`archive/` keeps every champion it replaced, so nothing is lost.

## What each round does

| Stage | What happens | Roughly |
|---|---|---|
| 1 | Generate Stockfish-labelled positions on all cores | ~35 min for 3M |
| 2 | Texel-tune the evaluation on everything collected | ~15 min |
| 3 | Fold the numbers into a candidate, verify the patch | seconds |
| 4 | SPRT match, candidate vs champion | 20-60 min |

Then it loops with more data. Later rounds tune on a bigger pile, which is the whole
point: the first rounds will probably fail, and that is expected.

## What "fail" means

I ran stage 2-4 here on 567k positions. The tuned evaluation fit the data **9.2% better**
and then **lost the match at −117 Elo**. The system rejected it, correctly.

That is the single most important thing to understand about this pipeline. A better fit
is not a better engine — 789 free parameters need far more than half a million positions
to beat tables that were themselves tuned on tens of millions. Your machine can generate
~170k positions a minute, so an overnight run should reach 20-50M. That is the regime
where tuning starts to win, and the SPRT will tell you when it does rather than you
having to guess.

## Why the matches are set up the way they are

Engine games are wall-clock timed, so playing several at once is only honest if each game
owns its cores — `match.py` pins every worker. Each opening is played from both sides so
colour cannot skew the result. And SPRT stops the moment the answer is decided instead of
grinding through a fixed count, which is what makes several rounds per night possible.

The default bounds test "is this at least +8 Elo" at 5% error either way. Tighten `--elo1`
if you want to catch smaller gains, but expect matches to take much longer.

## Knobs worth touching

| Flag | Default | Note |
|---|---|---|
| `--workers` | cores − 1 | data generation |
| `--games-per-round` | 6000 | more data per round, fewer rounds |
| `--match-base-ms` | 8000 | longer games are more representative, and slower |
| `--max-match-games` | 1200 | SPRT usually stops well before this |

## Files

- `gen.py` — parallel Stockfish-labelled data generation
- `tune.py` — Texel tuning; `--check` proves its features reproduce the engine exactly
- `apply_tables.py` — folds tuned numbers into both copies of the tables, then verifies
- `match.py` — core-pinned parallel matches with SPRT
- `overnight.py` — the loop
