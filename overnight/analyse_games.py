"""Read the rated games and work out what actually happened on the clock.

An engine that searches to depth 26 does not rate 1812, so the rating is being set by
something other than playing strength. The clocks are where that shows: this reports, per
side per game, how the time was spent, when a side dropped under the panic threshold, and
how many moves it then played nearly instantly.

    python overnight/analyse_games.py "C:/path/to/*.pgn"
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

CLK = re.compile(r"\[%clk\s+(\d+):(\d{2}):([\d.]+)\]")
PANIC_MS = 5000          # the exemplar's threshold: below this it plays near-instantly
INSTANT_MS = 200         # a move this fast was not searched


def clocks(text: str) -> tuple[list[float], list[float]]:
    """Seconds remaining after each move, split into the two sides by move order."""
    values = [int(h) * 3600 + int(m) * 60 + float(s) for h, m, s in CLK.findall(text)]
    body = text.split("]\n\n")[-1]
    first_is_black = bool(re.match(r"\s*\d+\.\.\.", body))
    white, black = [], []
    for index, value in enumerate(values):
        to = (black if first_is_black else white) if index % 2 == 0 else \
             (white if first_is_black else black)
        to.append(value)
    return white, black


def spent(series: list[float], increment: float = 0.5) -> list[float]:
    """Time actually consumed per move: the clock fell by this much before the increment."""
    out = []
    for i in range(1, len(series)):
        out.append(series[i - 1] - series[i] + increment)
    return out


def describe(name: str, series: list[float]) -> dict:
    used = spent(series)
    instant = sum(1 for u in used if u * 1000 < INSTANT_MS)
    panicked = next((i for i, v in enumerate(series) if v * 1000 < PANIC_MS), None)
    return {
        "side": name,
        "moves": len(series),
        "start": series[0] if series else 0.0,
        "end": series[-1] if series else 0.0,
        "median_ms": sorted(used)[len(used) // 2] * 1000 if used else 0.0,
        "instant_moves": instant,
        "panic_at": panicked,
        "after_panic": (len(series) - panicked) if panicked is not None else 0,
    }


def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else \
        str(Path.home() / "Downloads" / "aichessathon-round-*.pgn")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no PGNs matched {pattern}")

    print(f"\n{'game':<26}{'side':<7}{'median/mv':>11}{'instant':>9}"
          f"{'panic@':>8}{'after':>7}{'end clk':>9}  result")
    for path in paths:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        result = re.search(r'\[Result "([^"]+)"\]', text)
        term = re.search(r'\[Termination "([^"]+)"\]', text)
        white, black = clocks(text)
        label = Path(path).stem.replace("aichessathon-round-", "r")
        for name, series in (("White", white), ("Black", black)):
            if not series:
                continue
            d = describe(name, series)
            panic = str(d["panic_at"]) if d["panic_at"] is not None else "-"
            print(f"{label:<26}{name:<7}{d['median_ms']:>10.0f}m{d['instant_moves']:>9}"
                  f"{panic:>8}{d['after_panic']:>7}{d['end']:>8.1f}s  "
                  f"{result.group(1) if result else '?'} "
                  f"({term.group(1) if term else '?'})")
        print()


if __name__ == "__main__":
    main()
