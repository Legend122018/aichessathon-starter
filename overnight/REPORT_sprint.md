# Search sprint

Ran 361 minutes from `try_ownpst.py`, SPRT [-5, 8] at 5s + 0.05s, 60 minutes per candidate.

| change | what it does | verdict | result |
| --- | --- | --- | --- |
| `ttsize` | four times the transposition table | inconclusive | +1287 =1093 -1300  -1 +/- 11 Elo |
| `lmrhist` | reduce less when history likes the move | inconclusive | +1279 =1161 -1240  +4 +/- 11 Elo |
| `aspwin` | narrower aspiration window | accept | +604 =459 -537  +15 +/- 17 Elo |
| `futility5` | futility pruning to depth 5 | inconclusive | +1309 =1063 -1308  +0 +/- 11 Elo |
| `iirfrom3` | no-table-move reduction from depth 3 | reject | +872 =770 -918  -6 +/- 14 Elo |
| `rfp90` | reverse futility margin 120 -> 90 | inconclusive | +1327 =1075 -1278  +5 +/- 11 Elo |

**1 of 6 promoted.** The current engine is `sprint_champion.py`.

Measured against the starting engine at 20s + 0.2s: **-4 +/- 26 Elo** over +225 =223 -232.

`inconclusive` means the clock ran out, not that the change is worthless: at these bounds a gain near +8 Elo needs a few thousand games to prove and this run gives each candidate a few hundred. Re-run a promising one on its own with `--only <name> --minutes 90` to settle it.
