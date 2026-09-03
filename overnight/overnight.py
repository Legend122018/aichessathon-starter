"""Run the whole loop unattended: generate, tune, test, promote, repeat.

Nothing is promoted because a loss function improved. Every candidate has to win an SPRT
match against the current champion before it replaces it, and the champion is only ever
overwritten after that test passes. If every candidate fails, you wake up to the engine
you went to bed with, plus a report saying what was tried.

    python overnight.py --hours 8 --workers 16

Stages, repeated until the clock runs out:
  1. generate more Stockfish-labelled positions across every core
  2. tune the evaluation on everything collected so far
  3. fold the tuned numbers into a candidate and verify the patch
  4. play the candidate against the champion under SPRT
  5. promote on a pass, discard on a fail, and go round again with more data
"""

from __future__ import annotations

import argparse
import datetime
import glob
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(command: list[str], log) -> int:
    log.write(f"\n$ {' '.join(command)}\n")
    log.flush()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    for line in process.stdout:
        sys.stdout.write(line)
        log.write(line)
        log.flush()
    return process.wait()


def count_positions(data_dir: str) -> int:
    total = 0
    for path in glob.glob(os.path.join(data_dir, "part*.csv")):
        with open(path, "rb") as handle:
            total += sum(1 for _ in handle)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="data generation workers; SMT threads are worth using here")
    parser.add_argument("--smt-stride", type=int, default=2)
    parser.add_argument("--data", default=os.path.join(HERE, "data"))
    parser.add_argument("--champion", default=os.path.join(HERE, "champion.py"))
    parser.add_argument("--games-per-round", type=int, default=6000,
                        help="Stockfish self-play games generated each round")
    parser.add_argument("--match-base-ms", type=int, default=8000)
    parser.add_argument("--match-inc-ms", type=int, default=80)
    parser.add_argument("--max-match-games", type=int, default=20000,
                        help="SPRT stops on its own; this only exists so a "
                             "pathological match cannot run forever")
    args = parser.parse_args()

    os.makedirs(args.data, exist_ok=True)
    archive = os.path.join(HERE, "archive")
    os.makedirs(archive, exist_ok=True)
    deadline = time.time() + args.hours * 3600
    python = sys.executable

    # Concurrent games for the match stage. Each game needs two cores to be timed
    # honestly, and one core is left for the operating system.
    # Matches get one physical core each. Data generation is throughput and can use
    # every thread; matches are timed, and a game sharing a physical core with another
    # game is measuring the scheduler as much as the engine.
    cores = os.cpu_count() or 2
    match_workers = max(1, (cores // args.smt_stride) - 1)

    report_path = os.path.join(HERE, "REPORT.md")
    log_path = os.path.join(HERE, "overnight.log")
    history: list[str] = []
    promotions = 0
    round_index = 0

    with open(log_path, "a") as log:
        log.write(f"\n\n===== run started {datetime.datetime.now():%Y-%m-%d %H:%M} =====\n")
        while time.time() < deadline:
            round_index += 1
            remaining = (deadline - time.time()) / 3600
            print(f"\n===== round {round_index} ({remaining:.1f}h left) =====", flush=True)

            print("[1/4] generating positions", flush=True)
            run([python, os.path.join(HERE, "gen.py"), "--out", args.data,
                 "--games", str(args.games_per_round), "--workers", str(args.workers),
                 "--seed", str(round_index)], log)
            positions = count_positions(args.data)
            print(f"      {positions:,} positions banked", flush=True)
            if time.time() >= deadline:
                break

            print("[2/4] tuning the evaluation", flush=True)
            tables = os.path.join(HERE, f"tuned_r{round_index}.py")
            if run([python, os.path.join(HERE, "tune.py"), "--data", args.data,
                    "--agent", args.champion, "--out", tables,
                    "--steps", "6000", "--limit", "4000000"], log) != 0:
                history.append(f"round {round_index}: tuning failed")
                continue

            print("[3/4] building the candidate", flush=True)
            candidate = os.path.join(HERE, f"candidate_r{round_index}.py")
            if run([python, os.path.join(HERE, "apply_tables.py"),
                    "--agent", args.champion, "--tables", tables,
                    "--out", candidate, "--tune-dir", HERE], log) != 0:
                history.append(f"round {round_index}: patch verification failed")
                continue

            print("[4/4] testing the candidate against the champion", flush=True)
            verdict = run([python, os.path.join(HERE, "match.py"),
                           "--candidate", candidate, "--champion", args.champion,
                           "--workers", str(match_workers),
                           "--smt-stride", str(args.smt_stride),
                           "--base-ms", str(args.match_base_ms),
                           "--inc-ms", str(args.match_inc_ms),
                           "--max-games", str(args.max_match_games),
                           "--seed", str(round_index)], log)

            if verdict == 0:
                stamp = datetime.datetime.now().strftime("%H%M")
                shutil.copy(args.champion,
                            os.path.join(archive, f"champion_before_r{round_index}_{stamp}.py"))
                shutil.copy(candidate, args.champion)
                promotions += 1
                history.append(f"round {round_index}: PROMOTED "
                               f"({positions:,} positions)")
                print("      promoted: the champion has been replaced", flush=True)
            else:
                history.append(f"round {round_index}: rejected "
                               f"({positions:,} positions)")
                print("      rejected: champion unchanged", flush=True)

            with open(report_path, "w") as report:
                report.write("# Overnight run\n\n")
                report.write(f"Finished round {round_index} at "
                             f"{datetime.datetime.now():%H:%M}.\n\n")
                report.write(f"- positions collected: **{count_positions(args.data):,}**\n")
                report.write(f"- candidates promoted: **{promotions}**\n")
                report.write(f"- champion: `{os.path.basename(args.champion)}`\n\n")
                report.write("## Rounds\n\n")
                for line in history:
                    report.write(f"- {line}\n")
                report.write("\n## What to do next\n\n")
                if promotions:
                    report.write("The champion improved. Package it and upload:\n\n")
                    report.write("```\ncp champion.py ../agent.py\n"
                                 "cd .. && uv run python -m harness.package\n```\n")
                else:
                    report.write("No candidate beat the champion. That is a real result, "
                                 "not a failure: more data is the usual fix, so let it "
                                 "run longer before changing the approach.\n")
                report.write("\nFull log: `overnight.log`\n")

    print(f"\ndone: {promotions} promotion(s) over {round_index} round(s)")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
