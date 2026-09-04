# Search parameter sweep

Average centipawn loss against Stockfish over 40 positions at 500ms per move. Lower is better; the baseline is the shipped configuration.

**baseline 24.1 cp**

| parameter | default | value | cp loss | change |
| --- | --- | --- | --- | --- |
| RFP_MARGIN | 75 | 55 | 29.6 | +5.4 worse |
| RFP_MARGIN | 75 | 65 | 25.0 | +0.9 worse |
| RFP_MARGIN | 75 | 85 | 22.8 | -1.4 better |
| RFP_MARGIN | 75 | 95 | 26.5 | +2.4 worse |
| RAZOR_MARGIN | 340 | 280 | 24.1 | +0.0 level |
| RAZOR_MARGIN | 340 | 310 | 19.4 | -4.7 better |
| RAZOR_MARGIN | 340 | 370 | 21.6 | -2.5 better |
| RAZOR_MARGIN | 340 | 400 | 21.1 | -3.0 better |
| FUT_MARGIN | 110 | 90 | 19.3 | -4.8 better |
| FUT_MARGIN | 110 | 100 | 23.3 | -0.8 better |
| FUT_MARGIN | 110 | 120 | 26.6 | +2.4 worse |
| FUT_MARGIN | 110 | 130 | 26.8 | +2.7 worse |
| FUT_BASE | 90 | 70 | 25.6 | +1.5 worse |
| FUT_BASE | 90 | 80 | 22.6 | -1.5 better |
| FUT_BASE | 90 | 100 | 26.6 | +2.4 worse |
| FUT_BASE | 90 | 110 | 25.2 | +1.1 worse |
| NMP_BASE | 4 | 3 | 22.1 | -2.0 better |
| NMP_BASE | 4 | 5 | 27.0 | +2.9 worse |
| NMP_DIV | 4 | 3 | 23.4 | -0.8 better |
| NMP_DIV | 4 | 5 | 25.4 | +1.3 worse |
| NMP_DIV | 4 | 6 | 27.7 | +3.6 worse |
| NMP_EVAL_DIV | 180 | 140 | 27.2 | +3.1 worse |
| NMP_EVAL_DIV | 180 | 160 | 23.3 | -0.8 better |
| NMP_EVAL_DIV | 180 | 200 | 24.1 | +0.0 level |
| NMP_EVAL_DIV | 180 | 220 | 25.5 | +1.4 worse |
| ASP_WINDOW | 12 | 8 | 20.1 | -4.0 better |
| ASP_WINDOW | 12 | 10 | 21.6 | -2.5 better |
| ASP_WINDOW | 12 | 16 | 18.6 | -5.6 better |
| ASP_WINDOW | 12 | 20 | 22.4 | -1.7 better |
| DELTA_MARGIN | 130 | 100 | 22.2 | -1.9 better |
| DELTA_MARGIN | 130 | 115 | 28.6 | +4.5 worse |
| DELTA_MARGIN | 130 | 145 | 21.6 | -2.6 better |
| DELTA_MARGIN | 130 | 160 | 23.6 | -0.5 level |
| SEE_QUIET | -55 | -75 | 27.3 | +3.2 worse |
| SEE_QUIET | -55 | -65 | 19.2 | -4.9 better |
| SEE_QUIET | -55 | -45 | 23.1 | -1.0 better |
| SEE_QUIET | -55 | -35 | 21.1 | -3.0 better |
| SEE_CAP | -105 | -135 | 19.6 | -4.5 better |
| SEE_CAP | -105 | -120 | 25.2 | +1.1 worse |
| SEE_CAP | -105 | -90 | 21.6 | -2.5 better |
| SEE_CAP | -105 | -75 | 24.0 | -0.1 level |
| SING_DEPTH | 7 | 5 | 22.7 | -1.4 better |
| SING_DEPTH | 7 | 6 | 25.8 | +1.7 worse |
| SING_DEPTH | 7 | 8 | 14.1 | -10.1 better |
| SING_DEPTH | 7 | 9 | 22.1 | -2.1 better |
| SING_MARGIN | 2 | 1 | 21.6 | -2.5 better |
| SING_MARGIN | 2 | 3 | 20.9 | -3.3 better |
| SING_MARGIN | 2 | 4 | 24.9 | +0.8 worse |
| HIST_PRUNE_MUL | -3200 | -4200 | 24.1 | +0.0 level |
| HIST_PRUNE_MUL | -3200 | -3700 | 24.1 | +0.0 level |
| HIST_PRUNE_MUL | -3200 | -2700 | 21.6 | -2.5 better |
| HIST_PRUNE_MUL | -3200 | -2200 | 24.1 | +0.0 level |
| LMR_HIST_DIV | 6000 | 4000 | 25.2 | +1.1 worse |
| LMR_HIST_DIV | 6000 | 5000 | 22.3 | -1.8 better |
| LMR_HIST_DIV | 6000 | 7000 | 24.1 | -0.1 level |
| LMR_HIST_DIV | 6000 | 8000 | 24.9 | +0.8 worse |
| RFP_DEPTH | 8 | 6 | 21.8 | -2.4 better |
| RFP_DEPTH | 8 | 7 | 22.7 | -1.4 better |
| RFP_DEPTH | 8 | 9 | 24.0 | -0.1 level |
| RFP_DEPTH | 8 | 10 | 24.3 | +0.1 level |
| FUT_DEPTH | 8 | 6 | 25.0 | +0.9 worse |
| FUT_DEPTH | 8 | 7 | 22.8 | -1.4 better |
| FUT_DEPTH | 8 | 9 | 22.5 | -1.6 better |
| FUT_DEPTH | 8 | 10 | 21.4 | -2.7 better |

## Worth a match

- `SING_DEPTH = 8` (-10.1 cp)
- `ASP_WINDOW = 16` (-5.6 cp)
- `SEE_QUIET = -65` (-4.9 cp)
- `FUT_MARGIN = 90` (-4.8 cp)
- `RAZOR_MARGIN = 310` (-4.7 cp)
- `SEE_CAP = -135` (-4.5 cp)
- `ASP_WINDOW = 8` (-4.0 cp)
- `SING_MARGIN = 3` (-3.3 cp)

These are proxy results. Each still has to win a real match before it ships.
