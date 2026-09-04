"""Run the queued experiments in order, unattended, and write down what happened.

Matches cannot overlap - they are wall-clock timed and two at once measure the machine
rather than the engines - so this runs them one after another and appends each verdict to
REPORT_tonight.md as it lands. If a stage fails, the next one still runs.

Everything here only measures. Nothing is promoted into submission.zip: that decision
waits for a human to read the verdicts.

    python overnight/tonight.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable
REPORT = HERE / "REPORT_tonight.md"

# The real control. The exemplar treats anything under five seconds as panic and plays
# instantly, so a fast match measures its panic path rather than its strength.
REAL = ["--base-ms", "120000", "--inc-ms", "500"]
FAST = ["--base-ms", "20000", "--inc-ms", "200"]

STAGES = [
    {
        "name": "exemplar vs ours",
        "why": "Decides which engine everything else should be built on.",
        "log": HERE / "stage_exemplar.log",
        "cmd": [PY, str(HERE / "arena2.py"),
                "--candidate", str(HERE / "exemplar" / "agent.py"),
                "--champion", str(ROOT / "agent.py"),
                *REAL, "--elo0", "-10", "--elo1", "10",
                "--max-minutes", "200", "--seed", "31"],
    },
    {
        "name": "tuned mobility vs champion",
        "why": "Mobility fitted by Texel over all 4.55M positions, folded in and verified.",
        "log": HERE / "stage_mobility.log",
        "cmd": [PY, str(HERE / "arena2.py"),
                "--candidate", str(HERE / "cand_tuned_mob.py"),
                "--champion", str(ROOT / "agent.py"),
                *FAST, "--elo0", "-5", "--elo1", "10",
                "--max-minutes", "150", "--seed", "37"],
    },
]


def note(text: str) -> None:
    stamp = time.strftime("%H:%M")
    with open(REPORT, "a", encoding="utf-8") as handle:
        handle.write(f"{text}\n")
    print(f"[{stamp}] {text}", flush=True)


def verdict_from(log: Path) -> str:
    """The last thing the match said about the result."""
    if not log.exists():
        return "no log written"
    lines = [ln.strip() for ln in log.read_text(encoding="utf-8").splitlines()
             if ln.strip() and "without its JIT" not in ln]
    for line in reversed(lines):
        if line.startswith("RESULT") or line.startswith("SPRT"):
            return line
    return lines[-1] if lines else "no output"


def main() -> None:
    note(f"\n## Unattended run, started {time.strftime('%Y-%m-%d %H:%M')}\n")
    for index, stage in enumerate(STAGES, 1):
        note(f"### {index}. {stage['name']}")
        note(f"{stage['why']}\n")
        started = time.monotonic()
        try:
            with open(stage["log"], "w", encoding="utf-8") as handle:
                subprocess.run(stage["cmd"], stdout=handle, stderr=subprocess.STDOUT,
                               check=False, cwd=str(ROOT))
        except Exception as exc:  # a broken stage must not take the rest of the night
            note(f"- stage failed to start: {type(exc).__name__}: {exc}\n")
            continue
        minutes = (time.monotonic() - started) / 60
        note(f"- ran {minutes:.0f} minutes")
        note(f"- {verdict_from(stage['log'])}")
        note(f"- full log: `{stage['log'].name}`\n")
    note("Nothing was promoted. Read the verdicts and decide.\n")


if __name__ == "__main__":
    main()
