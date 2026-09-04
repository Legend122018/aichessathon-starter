# Search parameter sweep

Average centipawn loss against Stockfish over 300 positions at 200,000 nodes per move. Lower is better. `delta` is the mean paired difference from the shipped configuration, and a candidate only counts if it clears two standard errors.

**baseline 25.7 cp**

| parameter | default | value | cp loss | delta | moves changed | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| NMP_BASE | 4 | 3 | 26.1 | +0.4 ±2.1 | 43/300 | level |
| ASP_WINDOW | 12 | 8 | 24.1 | -1.6 ±3.8 | 51/300 | level |
| ASP_WINDOW | 12 | 10 | 26.3 | +0.6 ±3.0 | 43/300 | level |
| SEE_CAP | -105 | -90 | 25.0 | -0.6 ±0.9 | 19/300 | level |
| SING_DEPTH | 7 | 9 | 26.8 | +1.2 ±3.1 | 44/300 | level |
| HIST_PRUNE_MUL | -3200 | -400 | 25.6 | -0.0 ±2.8 | 38/300 | level |
| HIST_PRUNE_MUL | -3200 | -800 | 25.9 | +0.2 ±1.3 | 13/300 | level |
| HIST_PRUNE_MUL | -3200 | -1600 | 25.5 | -0.2 ±0.5 | 4/300 | level |

## Worth a match

Nothing cleared two standard errors. On this evidence the shipped configuration is already at least as good as every neighbour tried, and the search is not where the remaining Elo is.
