"""A fixed-length run that tests one search change at a time and keeps what wins.

The overnight script tuned evaluation tables. Nine rounds of it produced one promotable
result worth about eight Elo, because the tables it starts from are already well tuned -
that seam is mined out. The search is not: it has no late move pruning, its late move
reductions are a flat one-or-two plies regardless of depth, and it never reduces when
the transposition table gives it nothing to try first.

So this run tests search changes instead, one per candidate, promoting each winner
before testing the next so the gains compound rather than competing. Every candidate is
compiled and played for legality before a single rated game, because a crash measured
for twenty minutes is twenty minutes bought for nothing.

    python sprint.py --minutes 90

What ninety minutes can and cannot settle is fixed by arithmetic, not by hope. At these
bounds a change worth about +15 Elo or better is decided inside its slot, and anything
worse than -5 is rejected quickly. A change worth +8 will run out of clock undecided -
which is the honest answer, and the reason the report says so rather than guessing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import patches  # noqa: E402


def normalise(source: Path, target: Path) -> Path:
    """Rewrite with one newline convention so a candidate diff is only the patch."""
    target.write_text(source.read_text())
    return target


def run(command: list[str], log) -> tuple[int, str]:
    """Stream a subprocess to the console and the log, returning its RESULT line."""
    log.write("$ " + " ".join(command) + "\n")
    log.flush()
    result = ""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    assert process.stdout
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log.write(line)
        log.flush()
        if line.startswith("RESULT "):
            result = line.strip()
    process.wait()
    return process.returncode, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", default=str(HERE / "candidate_r6.py"),
                        help="starting engine; defaults to the round 6 build, which is "
                             "the strongest measured version")
    parser.add_argument("--minutes", type=float, default=90.0)
    parser.add_argument("--slot-minutes", type=float, default=0.0,
                        help="cap per candidate; 0 splits the budget evenly")
    parser.add_argument("--base-ms", type=int, default=4000)
    parser.add_argument("--inc-ms", type=int, default=40)
    parser.add_argument("--elo0", type=float, default=-5.0)
    parser.add_argument("--elo1", type=float, default=15.0)
    parser.add_argument("--workers", type=int,
                        default=max(1, ((os.cpu_count() or 2) // 2) - 1))
    parser.add_argument("--only", default="",
                        help="comma separated patch names, in order; default is all")
    parser.add_argument("--confirm-minutes", type=float, default=0.0,
                        help="after the candidates, replay the finished engine against "
                             "the one it started from at a slower time control. The "
                             "sprint tests each change against the champion of the "
                             "moment; this measures what the night was worth in total, "
                             "at a control closer to the competition's")
    parser.add_argument("--confirm-base-ms", type=int, default=20000)
    parser.add_argument("--confirm-inc-ms", type=int, default=200)
    args = parser.parse_args()

    wanted = [p for p in patches.PATCHES
              if not args.only or p[0] in args.only.split(",")]
    if not wanted:
        raise SystemExit(f"no patches matched {args.only!r}")
    slot = args.slot_minutes or (args.minutes / len(wanted))

    # Two files, deliberately. `sprint_start.py` is an immutable snapshot of what the run
    # began with; `sprint_champion.py` is promoted into as candidates win. Without the
    # snapshot, starting a run from sprint_champion.py would leave the confirmation match
    # playing that file against itself and reporting a solemn zero.
    start = normalise(Path(args.champion), HERE / "sprint_start.py")
    champion = HERE / "sprint_champion.py"
    shutil.copy(start, champion)
    started = time.monotonic()
    outcomes = []

    with open(HERE / "sprint.log", "w") as log:
        print(f"{len(wanted)} candidates, {slot:.0f} minutes each, "
              f"{args.workers} concurrent games", flush=True)
        print(f"starting from {Path(args.champion).name}, "
              f"SPRT [{args.elo0:g}, {args.elo1:g}] at "
              f"{args.base_ms / 1000:g}s + {args.inc_ms / 1000:g}s\n", flush=True)

        for key, blurb, _ in wanted:
            left = args.minutes - (time.monotonic() - started) / 60.0
            if left < 4:
                print(f"[{key}] skipped - {left:.0f} minutes left", flush=True)
                outcomes.append((key, blurb, "skipped", ""))
                continue

            print(f"\n=== {key}: {blurb} ===", flush=True)
            candidate = HERE / f"try_{key}.py"
            patches.build(champion, key, candidate)

            code, _ = run([sys.executable, str(HERE / "smoke.py"), str(candidate)], log)
            if code != 0:
                print(f"[{key}] rejected before playing: it does not run cleanly",
                      flush=True)
                outcomes.append((key, blurb, "broken", ""))
                continue

            code, result = run([
                sys.executable, str(HERE / "match_fast.py"),
                "--candidate", str(candidate), "--champion", str(champion),
                "--workers", str(args.workers),
                "--base-ms", str(args.base_ms), "--inc-ms", str(args.inc_ms),
                "--elo0", str(args.elo0), "--elo1", str(args.elo1),
                "--max-minutes", f"{min(slot, left):.2f}",
            ], log)

            fields = result.split()
            verdict = fields[1] if len(fields) > 2 else "unknown"
            detail = (f"+{fields[2]} ={fields[3]} -{fields[4]}  "
                      f"{float(fields[5]):+.0f} +/- {float(fields[6]):.0f} Elo"
                      if len(fields) > 6 else "")
            outcomes.append((key, blurb, verdict, detail))

            if verdict == "accept":
                shutil.copy(candidate, champion)
                print(f"[{key}] promoted - later candidates build on it", flush=True)
            else:
                print(f"[{key}] discarded", flush=True)

    kept = [o for o in outcomes if o[2] == "accept"]
    confirm = ""
    if kept and args.confirm_minutes:
        print(f"\n=== confirming: the night's engine against the one it started from, "
              f"{args.confirm_base_ms / 1000:g}s + {args.confirm_inc_ms / 1000:g}s ===",
              flush=True)
        with open(HERE / "sprint.log", "a") as log:
            _, result = run([
                sys.executable, str(HERE / "match_fast.py"),
                "--candidate", str(champion), "--champion", str(start),
                "--workers", str(args.workers),
                "--base-ms", str(args.confirm_base_ms),
                "--inc-ms", str(args.confirm_inc_ms),
                "--elo0", "0", "--elo1", "10",
                "--max-minutes", f"{args.confirm_minutes:.2f}",
            ], log)
        fields = result.split()
        if len(fields) > 6:
            confirm = (f"Measured against the starting engine at "
                       f"{args.confirm_base_ms / 1000:g}s + "
                       f"{args.confirm_inc_ms / 1000:g}s: "
                       f"**{float(fields[5]):+.0f} +/- {float(fields[6]):.0f} Elo** "
                       f"over +{fields[2]} ={fields[3]} -{fields[4]}.")

    spent = (time.monotonic() - started) / 60.0
    lines = [
        "# Search sprint",
        "",
        f"Ran {spent:.0f} minutes from `{Path(args.champion).name}`, "
        f"SPRT [{args.elo0:g}, {args.elo1:g}] at "
        f"{args.base_ms / 1000:g}s + {args.inc_ms / 1000:g}s, "
        f"{slot:.0f} minutes per candidate.",
        "",
        "| change | what it does | verdict | result |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| `{k}` | {b} | {v} | {d} |" for k, b, v, d in outcomes]
    lines += [
        "",
        f"**{len(kept)} of {len(outcomes)} promoted.** The current engine is "
        "`sprint_champion.py`.",
        "",
    ]
    if confirm:
        lines += [confirm, ""]
    lines += [
        "`inconclusive` means the clock ran out, not that the change is worthless: at "
        "these bounds a gain near +8 Elo needs a few thousand games to prove and this "
        "run gives each candidate a few hundred. Re-run a promising one on its own with "
        "`--only <name> --minutes 90` to settle it.",
    ]
    (HERE / "REPORT_sprint.md").write_text("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines[3:]), flush=True)
    print(f"\nfull log: {HERE / 'sprint.log'}", flush=True)
    if kept:
        print("\nTo ship the result:\n"
              f"  Copy-Item sprint_champion.py champion.py\n"
              f"  Copy-Item sprint_champion.py ..\\agent.py", flush=True)


if __name__ == "__main__":
    main()
