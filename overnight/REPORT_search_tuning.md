# Search parameter sweep

Average centipawn loss against Stockfish over 300 positions at 200,000 nodes per move. Lower is better. `delta` is the mean paired difference from the shipped configuration, and a candidate only counts if it clears two standard errors.

**baseline 22.8 cp**

| parameter | default | value | cp loss | delta | moves changed | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| RFP_MARGIN | 75 | 55 | 21.1 | -1.6 ±3.7 | 39/300 | level |
| RFP_MARGIN | 75 | 65 | 21.6 | -1.2 ±3.3 | 41/300 | level |
| RFP_MARGIN | 75 | 85 | 21.0 | -1.8 ±3.1 | 37/300 | level |
| RFP_MARGIN | 75 | 95 | 20.6 | -2.2 ±4.5 | 47/300 | level |
| RAZOR_MARGIN | 340 | 280 | 20.8 | -2.0 ±2.9 | 41/300 | level |
| RAZOR_MARGIN | 340 | 310 | 22.9 | +0.1 ±4.1 | 45/300 | level |
| RAZOR_MARGIN | 340 | 370 | 20.9 | -1.9 ±4.3 | 39/300 | level |
| RAZOR_MARGIN | 340 | 400 | 21.1 | -1.6 ±4.2 | 41/300 | level |
| FUT_MARGIN | 110 | 90 | 20.2 | -2.5 ±3.0 | 39/300 | level |
| FUT_MARGIN | 110 | 100 | 22.2 | -0.6 ±4.1 | 26/300 | level |
| FUT_MARGIN | 110 | 120 | 22.1 | -0.7 ±3.0 | 26/300 | level |
| FUT_MARGIN | 110 | 130 | 22.6 | -0.1 ±2.8 | 25/300 | level |
| FUT_BASE | 90 | 70 | 20.4 | -2.4 ±2.8 | 32/300 | level |
| FUT_BASE | 90 | 80 | 22.3 | -0.5 ±4.0 | 24/300 | level |
| FUT_BASE | 90 | 100 | 21.6 | -1.2 ±2.6 | 23/300 | level |
| FUT_BASE | 90 | 110 | 23.6 | +0.8 ±4.0 | 29/300 | level |
| NMP_BASE | 4 | 3 | 19.3 | -3.5 ±3.1 | 34/300 | BETTER |
| NMP_BASE | 4 | 5 | 21.8 | -1.0 ±3.1 | 37/300 | level |
| NMP_DIV | 4 | 3 | 21.1 | -1.7 ±3.1 | 31/300 | level |
| NMP_DIV | 4 | 5 | 22.0 | -0.8 ±3.0 | 31/300 | level |
| NMP_DIV | 4 | 6 | 23.4 | +0.6 ±4.5 | 34/300 | level |
| NMP_EVAL_DIV | 180 | 140 | 22.4 | -0.4 ±1.2 | 15/300 | level |
| NMP_EVAL_DIV | 180 | 160 | 22.5 | -0.3 ±0.9 | 12/300 | level |
| NMP_EVAL_DIV | 180 | 200 | 23.2 | +0.4 ±1.7 | 14/300 | level |
| NMP_EVAL_DIV | 180 | 220 | 21.9 | -0.8 ±1.8 | 17/300 | level |
| ASP_WINDOW | 12 | 8 | 19.4 | -3.4 ±3.1 | 43/300 | BETTER |
| ASP_WINDOW | 12 | 10 | 19.2 | -3.6 ±3.2 | 36/300 | BETTER |
| ASP_WINDOW | 12 | 16 | 21.1 | -1.7 ±2.9 | 43/300 | level |
| ASP_WINDOW | 12 | 20 | 20.3 | -2.5 ±4.4 | 56/300 | level |
| DELTA_MARGIN | 130 | 100 | 21.1 | -1.7 ±4.2 | 39/300 | level |
| DELTA_MARGIN | 130 | 115 | 21.8 | -1.0 ±2.8 | 28/300 | level |
| DELTA_MARGIN | 130 | 145 | 21.2 | -1.6 ±4.1 | 32/300 | level |
| DELTA_MARGIN | 130 | 160 | 23.0 | +0.2 ±4.4 | 37/300 | level |
| SEE_QUIET | -55 | -75 | 22.7 | -0.1 ±4.4 | 39/300 | level |
| SEE_QUIET | -55 | -65 | 21.1 | -1.6 ±4.2 | 33/300 | level |
| SEE_QUIET | -55 | -45 | 23.1 | +0.3 ±4.4 | 32/300 | level |
| SEE_QUIET | -55 | -35 | 22.9 | +0.1 ±4.5 | 43/300 | level |
| SEE_CAP | -105 | -135 | 21.4 | -1.4 ±4.5 | 42/300 | level |
| SEE_CAP | -105 | -120 | 21.6 | -1.2 ±4.3 | 37/300 | level |
| SEE_CAP | -105 | -90 | 19.7 | -3.1 ±2.8 | 20/300 | BETTER |
| SEE_CAP | -105 | -75 | 23.3 | +0.5 ±4.3 | 34/300 | level |
| SING_DEPTH | 7 | 5 | 20.4 | -2.4 ±3.2 | 40/300 | level |
| SING_DEPTH | 7 | 6 | 21.5 | -1.3 ±3.5 | 38/300 | level |
| SING_DEPTH | 7 | 8 | 20.8 | -2.0 ±4.2 | 42/300 | level |
| SING_DEPTH | 7 | 9 | 18.3 | -4.5 ±3.1 | 39/300 | BETTER |
| SING_MARGIN | 2 | 1 | 20.3 | -2.5 ±3.0 | 37/300 | level |
| SING_MARGIN | 2 | 3 | 21.2 | -1.6 ±4.2 | 29/300 | level |
| SING_MARGIN | 2 | 4 | 22.2 | -0.5 ±4.0 | 44/300 | level |
| HIST_PRUNE_MUL | -3200 | -4200 | 22.8 | +0.0 ±0.0 | 0/300 | level |
| HIST_PRUNE_MUL | -3200 | -3700 | 22.8 | +0.0 ±0.0 | 0/300 | level |
| HIST_PRUNE_MUL | -3200 | -2700 | 22.8 | +0.0 ±0.0 | 0/300 | level |
| HIST_PRUNE_MUL | -3200 | -2200 | 22.8 | +0.0 ±0.0 | 0/300 | level |
| LMR_HIST_DIV | 6000 | 4000 | 21.7 | -1.1 ±4.3 | 27/300 | level |
| LMR_HIST_DIV | 6000 | 5000 | 20.8 | -2.0 ±2.5 | 27/300 | level |
| LMR_HIST_DIV | 6000 | 7000 | 22.9 | +0.1 ±1.3 | 17/300 | level |
| LMR_HIST_DIV | 6000 | 8000 | 22.3 | -0.5 ±1.5 | 19/300 | level |
| RFP_DEPTH | 8 | 6 | 23.9 | +1.1 ±4.0 | 23/300 | level |
| RFP_DEPTH | 8 | 7 | 22.4 | -0.4 ±2.0 | 16/300 | level |
| RFP_DEPTH | 8 | 9 | 21.8 | -1.0 ±1.9 | 9/300 | level |
| RFP_DEPTH | 8 | 10 | 21.8 | -1.0 ±1.9 | 13/300 | level |
| FUT_DEPTH | 8 | 6 | 21.5 | -1.3 ±1.7 | 11/300 | level |
| FUT_DEPTH | 8 | 7 | 22.8 | +0.0 ±0.6 | 5/300 | level |
| FUT_DEPTH | 8 | 9 | 22.1 | -0.6 ±1.9 | 4/300 | level |
| FUT_DEPTH | 8 | 10 | 22.0 | -0.8 ±1.9 | 2/300 | level |

## Worth a match

- `SING_DEPTH = 9` (-4.5 ±3.1 cp)
- `ASP_WINDOW = 10` (-3.6 ±3.2 cp)
- `NMP_BASE = 3` (-3.5 ±3.1 cp)
- `ASP_WINDOW = 8` (-3.4 ±3.1 cp)
- `SEE_CAP = -90` (-3.1 ±2.8 cp)

These are proxy results on one position suite. Each still has to win a real match before it ships.
