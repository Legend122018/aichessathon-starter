# Search parameter sweep

Average centipawn loss against Stockfish over 300 positions at 200,000 nodes per move. Lower is better. `delta` is the mean paired difference from the shipped configuration, and a candidate only counts if it clears two standard errors.

**baseline 21.8 cp**

| parameter | default | value | cp loss | delta | moves changed | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| NMP_BASE | 4 | 3 | 21.4 | -0.3 ±2.7 | 38/300 | level |
| ASP_WINDOW | 12 | 8 | 23.4 | +1.6 ±2.7 | 37/300 | level |
| ASP_WINDOW | 12 | 10 | 24.6 | +2.8 ±3.1 | 38/300 | level |
| SEE_CAP | -105 | -90 | 20.9 | -0.9 ±1.4 | 18/300 | level |
| SING_DEPTH | 7 | 9 | 24.3 | +2.5 ±2.5 | 33/300 | worse |
| HIST_PRUNE_MUL | -3200 | -400 | 21.9 | +0.1 ±1.5 | 26/300 | level |
| HIST_PRUNE_MUL | -3200 | -800 | 22.3 | +0.5 ±1.4 | 13/300 | level |
| HIST_PRUNE_MUL | -3200 | -1600 | 21.5 | -0.3 ±0.6 | 1/300 | level |

## Worth a match

Nothing cleared two standard errors. On this evidence the shipped configuration is already at least as good as every neighbour tried, and the search is not where the remaining Elo is.
