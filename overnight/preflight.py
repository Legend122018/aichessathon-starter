"""Check a submission zip against the published rules, mechanically.

Every check comes from the competition documentation rather than from memory, so a rule
change shows up as a failing line instead of a wrong assumption.

This was originally written for a single flat agent.py and has been generalised, because
the entry is now a package: agent.py at the root with `engine/` and `data/` beside it.
Two of the old checks were not rules at all but artefacts of that shape - "nothing inside
a folder", and treating any filesystem import as a violation. Subdirectories are allowed,
and reading your own bundled data is not only allowed but required, since that is how the
tablebases and the opening book are loaded. What is actually banned is reaching outside
the process: network, subprocess, and installing anything.

The import checks now read every Python file in the zip rather than agent.py alone. A
banned import in engine/search.py is exactly as disqualifying as one in agent.py, and the
old version could not see it.

    python overnight/preflight.py
    python overnight/preflight.py --zip submission.zip --source .
"""

from __future__ import annotations

import argparse
import ast
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The container ships these and nothing installs at validation.
ALLOWED_PACKAGES = {"torch", "numpy", "chess", "onnxruntime", "numba"}
STDLIB_OK = {
    "__future__", "collections", "collections.abc", "random", "threading", "time",
    "typing", "math", "dataclasses", "itertools", "functools", "heapq", "array", "struct",
    "os", "os.path",
}
# Reaching outside the process. `os` is deliberately absent: the entry has to open its
# own tablebase and book files, and forbidding that would forbid the entry.
BANNED = {
    "socket", "urllib", "http", "requests", "httpx", "subprocess", "ftplib", "telnetlib",
    "asyncio", "multiprocessing", "ctypes", "shutil", "pip", "importlib", "pickle",
}
# A file named after a module we import shadows the real one, because the zip is first
# on sys.path.
SHADOW_RISK = {
    "chess.py", "types.py", "random.py", "time.py", "typing.py", "numpy.py", "numba.py",
    "threading.py", "collections.py", "torch.py", "onnxruntime.py", "math.py", "os.py",
}
NATIVE_SUFFIXES = {".exe", ".dll", ".so", ".dylib", ".pyd", ".o", ".a"}
NATIVE_MAGIC = (b"MZ", b"\x7fELF", b"\xca\xfe\xba\xbe", b"\xcf\xfa\xed\xfe")

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail) -> None:
    results.append((ok, name, detail if isinstance(detail, str) else ", ".join(detail)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default=str(ROOT / "submission_exemplar.zip"))
    parser.add_argument("--source", default=str(ROOT / "overnight" / "exemplar"),
                        help="the tree the zip was built from, to catch a stale zip")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    source = Path(args.source)
    if not zip_path.exists():
        print(f"no zip at {zip_path} - run `uv run python overnight/package.py` first")
        return 1

    zf = zipfile.ZipFile(zip_path)
    names = zf.namelist()
    unzipped = sum(i.file_size for i in zf.infolist())
    pyfiles = [n for n in names if n.endswith(".py")]

    check("agent.py" in names, "agent.py at the zip root", ", ".join(names[:4]) + " ...")
    check(unzipped <= 50 * 1024 * 1024, "50 MB unzipped at most",
          f"{unzipped / 1024 / 1024:.2f} MB unzipped, "
          f"{zip_path.stat().st_size / 1024 / 1024:.2f} MB zipped, "
          f"{(50 * 1024 * 1024 - unzipped) / 1024 / 1024:.1f} MB spare")
    check(not [n for n in names if Path(n).name in SHADOW_RISK],
          "no module-shadowing filenames", "no chess.py / types.py / os.py etc.")
    check(not [n for n in names if "__pycache__" in n], "no compiled caches shipped",
          "no __pycache__ entries")

    native = []
    for n in names:
        if Path(n).suffix.lower() in NATIVE_SUFFIXES:
            native.append(n)
            continue
        if any(zf.read(n)[:4].startswith(m) for m in NATIVE_MAGIC):
            native.append(n)
    check(not native, "no native binaries", native or "every entry is source or data")

    # Every Python file, not just agent.py: a banned import in engine/search.py
    # disqualifies the entry exactly as hard, and is exactly as easy to miss.
    imports: set[str] = set()
    trees: dict[str, ast.Module] = {}
    for name in pyfiles:
        tree = ast.parse(zf.read(name).decode("utf-8"), filename=name)
        trees[name] = tree
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imports.add(node.module)

    outside = sorted(
        i for i in imports
        if i.split(".")[0] not in ALLOWED_PACKAGES
        and i not in STDLIB_OK
        and not i.startswith("engine")
    )
    check(not outside, f"imports stay inside the environment ({len(pyfiles)} files)",
          outside or ", ".join(sorted(imports)))

    reached = sorted(i for i in imports if i.split(".")[0] in BANNED)
    check(not reached, "no network, subprocess or dynamic import",
          reached or "none imported")

    agent_tree = trees.get("agent.py")
    fn = next((n for n in (agent_tree.body if agent_tree else [])
               if isinstance(n, ast.FunctionDef) and n.name == "get_move"), None)
    fn_args = [a.arg for a in fn.args.args] if fn else []
    check(fn_args == ["fen", "time_left_ms"], "get_move(fen, time_left_ms)",
          f"signature: {fn_args or 'MISSING'}")

    # The 4 KB per-move output cap loses the game outright when exceeded, so any print
    # that could carry an exception's text has to be sliced.
    prints = [n for tree in trees.values() for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id == "print"]
    unbounded = [p for p in prints if p.args and not isinstance(p.args[0], ast.Subscript)]
    check(not unbounded, "output stays under the 4 KB per-move cap",
          f"{len(prints)} print call(s), {len(unbounded)} untruncated")

    # The rule bans shipping, calling or wrapping another engine - not saying its name.
    # A grep cannot tell those apart, and it just failed the entry over a docstring
    # explaining where the book came from, which is the opposite of what the rule wants:
    # that sentence is the disclosure. So look for the thing itself. Any engine name
    # reached from live code - a string the program uses, an attribute, a variable -
    # fails. A name in a comment or docstring is reported for a human to read, because
    # documenting provenance is the honest behaviour and should not be quietly deleted
    # to make a check go green.
    engine_names = ("stockfish", "lc0", "leela", "maia", "komodo", "houdini")
    live: list[str] = []
    for name, tree in trees.items():
        docstrings = {ast.get_docstring(n, clean=False) for n in ast.walk(tree)
                      if isinstance(n, ast.Module | ast.ClassDef | ast.FunctionDef)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in docstrings:
                    continue
                hit = [e for e in engine_names if e in node.value.lower()]
                if hit:
                    live.append(f"{name}: string {node.value[:40]!r}")
            elif isinstance(node, ast.Name | ast.Attribute):
                ident = (node.id if isinstance(node, ast.Name) else node.attr).lower()
                if any(e in ident for e in engine_names):
                    live.append(f"{name}: identifier {ident}")
    check(not live, "no third-party engine reachable from code",
          live[:3] or "no engine name in any executable path")

    mentions = [n for n in pyfiles
                if any(e in zf.read(n).decode("utf-8").lower() for e in engine_names)]
    check(True, "engine names in prose (disclosure, not use)",
          f"{', '.join(mentions)} - read these before uploading" if mentions
          else "none")

    stale = []
    for name in names:
        on_disk = source / name
        if not on_disk.exists():
            stale.append(f"{name} not in the source tree")
            continue
        if zf.read(name).replace(b"\r\n", b"\n") != on_disk.read_bytes().replace(b"\r\n", b"\n"):
            stale.append(f"{name} differs")
    check(not stale, "zip matches the working tree",
          stale[:3] or f"all {len(names)} entries identical")

    width = max(len(name) for _, name, _ in results)
    failed = sum(1 for ok, _, _ in results if not ok)
    print(f"\n  {zip_path.name} - {zip_path.stat().st_size / 1024 / 1024:.2f} MB\n")
    for ok, name, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
    print()
    if failed:
        print(f"  {failed} check(s) failed - do not upload this\n")
        return 1
    print(f"  all {len(results)} checks passed\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
