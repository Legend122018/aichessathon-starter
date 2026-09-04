
## Unattended run, started 2026-09-04 02:36

### 1. exemplar vs ours
Decides which engine everything else should be built on.

- ran 201 minutes
- SPRT: candidate is better - accept
- full log: `stage_exemplar.log`

### 2. tuned mobility vs champion
Mobility fitted by Texel over all 4.55M positions, folded in and verified.

- ran 161 minutes
- SPRT: out of time at 150 minutes, undecided
- full log: `stage_mobility.log`

Nothing was promoted. Read the verdicts and decide.

### 3. contempt ported into the exemplar

Contempt is the one validated idea agent.py had that the exemplar lacks.

- ran 178 minutes
- +38 =93 -37 over 168 games, +2 +/- 53 Elo, undecided
- full log: `stage_contempt.log`

The error bar is the point. A +/- 53 Elo interval cannot resolve a +10 Elo term, so this
match is not evidence for contempt or against it - it is evidence that the question needs
about 2,000 games, or roughly 33 hours on this hardware. Contempt is left out of the
recommended upload on the principle that a strong engine does not take unvalidated
changes.

## Recommendation

Upload `submission_exemplar.zip`. It beat agent.py by +311 +/- 187 Elo over 42 games at
the real control, losing one game in forty-two, and it passes the rules check. Read the
platform's validation log afterwards: it is the authority on whether the engine/ and data/
subfolders are accepted, which is the one thing that cannot be checked locally.
