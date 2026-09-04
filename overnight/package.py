"""Build the submission zip from the exemplar tree.

`agent.py` has to sit at the zip root, with the `engine` package and `data` beside it.
Compiled caches are excluded: `__pycache__` is bytecode for this machine's interpreter,
it is dead weight against the 50 MB limit, and a stale .pyc next to an edited .py is a
way to ship code nobody reviewed.

    python overnight/package.py            # writes submission_exemplar.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = HERE / "exemplar"

SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def entries(source: Path):
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        yield path, path.relative_to(source).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--out", default=str(ROOT / "submission_exemplar.zip"))
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    if not (source / "agent.py").exists():
        raise SystemExit(f"no agent.py in {source}")

    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path, name in entries(source):
            zf.write(path, name)
            total += path.stat().st_size

    print(f"{out.name}: {len(zf.namelist())} files, "
          f"{total / 1024 / 1024:.2f} MB unzipped, "
          f"{out.stat().st_size / 1024 / 1024:.2f} MB zipped")
    print(f"headroom against the 50 MB limit: "
          f"{(50 * 1024 * 1024 - total) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
