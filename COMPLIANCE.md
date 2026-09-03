# Compliance record

Every rule on aichessathon.com/docs and aichessathon.com/terms, checked against the
shipped `agent.py`. Written to be read by a judge: the final asks each team to walk
through how its agent was built, and this is that walkthrough.

Last checked against the live rules on 2 September 2026.

## Submission format

| Rule | Status |
| --- | --- |
| Zip, 50 MB unzipped at most | 103 KB unzipped, 25 KB zipped |
| `agent.py` at the root, not in a folder | yes, `harness/package.py` builds it that way |
| `get_move(fen: str, time_left_ms: int) -> str`, UCI | yes |
| 6 uploads per team per day | tracked manually; gate locally before each upload |
| No file shadowing a module we import | the zip holds `agent.py` and `requirements.txt` only - no `chess.py`, `types.py`, `random.py`, `time.py` |

## Environment

The container ships Python 3.12 plus torch 2.13.0+cpu, numpy 2.5.2, python-chess
1.11.2, onnxruntime 1.29.0 and numba 0.67.0. Nothing else installs and a
`requirements.txt` in the zip is ignored.

`agent.py` imports exactly: `__future__`, `collections`, `random`, `time`, `typing` from
the standard library, and `chess`, `numba`, `numpy` from the allowed set. It imports
nothing else. The local `uv` environment pins the same versions the platform does, so
local measurements transfer.

No native binaries. The file is pure Python source; the speed comes from numba, which
compiles in process at import.

## Runtime limits

| Limit | Status |
| --- | --- |
| 1 dedicated core | single threaded; no threads are created |
| 2 GB memory | transposition table is 42 MB, everything else is small |
| No network, either direction | no socket, urllib, subprocess or http import; verified by AST scan |
| No GPU | not used, and measured not to be worth using (see below) |
| Read-only filesystem, 256 MB at `/tmp` | the agent opens no files and writes nothing |
| 128 processes | one |
| 4096 bytes of output per move | one fallback message on an unhandled error, truncated to 200 characters. It was unbounded: a numba `TypingError` repr alone runs to several kilobytes, and passing the cap loses the game |
| 60 s init budget, or the game is lost | see below |

### The init budget

Compilation is the only thing that can blow it. The agent times a trivial numba compile
at import and extrapolates: if the estimate does not fit comfortably inside the budget it
skips the JIT entirely and plays with the pure-Python engine, which is weaker but always
finishes.

Measured on a deliberately slow two-core machine: probe 3.5 s, guard estimate 38 s
(assuming an 11x multiplier), real import 27.9 s, true multiplier 8.0x. The guard is
conservative in the safe direction - it gives up on the JIT sooner than it strictly has
to, rather than risking the budget. On a fast machine the whole import takes about 16 s.

## Originality

> "Your moves come from code you wrote and any model you ship is one you trained."
> "Training data is unrestricted, including positions annotated by an existing engine."
> "The ban covers only what ships inside the submission."

- No third-party engine ships, is called, or is wrapped. There is no Stockfish, Lc0 or
  Maia code, binary, or pip package anywhere in the submission.
- The search - negamax, alpha-beta, quiescence, transposition table, move ordering,
  null-move pruning, late move reductions, the numba port - is original code.
- **The evaluation tables were rebuilt from scratch.** Earlier versions used the PeSTO
  piece-square tables, which are 768 values hand-tuned by Ronald Friederich for RofChade
  and published openly. They are fine to learn from and awkward to ship, because they are
  neither code we wrote nor a model we trained. Every value now comes from fitting our
  own data: the starting point is textbook material only (100/320/330/500/900, no
  square-by-square structure at all) and Texel's method supplies every deviation.
  Provenance of the fit is in `tune.py --from-material`, and the result differs from the
  old tables by 31 cp per square on average, up to 158 cp - an independent fit, not a
  perturbation.
- Training data is self-play positions labelled by Stockfish, run locally before the
  event. The rules permit this explicitly; no Stockfish code or output ships in the zip.
- Stockfish was also used locally as a rating yardstick. It never ships and is never
  called at runtime.

## Readability

> "What you ship has to be source a judge can read." Obfuscated agents are disqualified.

`agent.py` is 2,543 lines of annotated Python with 128 comment lines explaining the
non-obvious parts. It passes `ruff` and `mypy --strict` clean. No minification, no
encoded blobs, no generated identifiers.

The one thing worth explaining to a reader: the file contains two engines. A pure-Python
one, and a numba-compiled one under `if HAVE_NUMBA:` whose names are all prefixed `JIT_`
or `jit_`. The prefixes exist because the two halves define functions with the same
meaning and different types, and an earlier version silently rebound one over the other.
The compiled half plays; the Python half is the fallback when compilation would not fit
the init budget.

## Fair play

No attempt is made to read hidden match data, other entrants' systems, organiser
infrastructure, or credentials. No pairing manipulation, no result coordination, no
impersonation. The agent interacts with the platform only through `get_move`.

## Things deliberately not done

- **No neural network.** A model is not required, and one was measured to be a bad trade
  here: a 768-256-32-1 net costs 4.7x the classical evaluation per call and keeps 31% of
  the node rate on one core. The measurement is reproducible with `probe.py`. An earlier
  attempt scored 8.8% against its predecessor, which matches.
- **No GPU training.** Follows from the above: there is nothing to train that would
  survive the speed cost.
- **No opening book reliance.** Every game starts from a curated near-level position
  rather than the initial position, so the book rarely fires. It is retained because it
  costs nothing, not because it does much.

## Open items

- Pondering is explicitly allowed and is not implemented. This is a strength opportunity,
  not a compliance gap.
- The local harness adjudicates a 300-ply game as a draw; the platform adjudicates on
  material first. Local results are very slightly biased against an engine that reaches
  ply 300 ahead.
