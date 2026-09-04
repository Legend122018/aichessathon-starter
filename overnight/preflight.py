"""Check submission.zip against the published rules, mechanically.

Every check here comes from the competition documentation rather than from memory, so a
rule change shows up as a failing line instead of a wrong assumption.

    python overnight/preflight.py
"""

from __future__ import annotations

import ast
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "submission.zip"
AGENT = ROOT / "agent.py"

# The container ships these and nothing installs at validation.
ALLOWED_PACKAGES = {"torch", "numpy", "chess", "onnxruntime", "numba"}
STDLIB_OK = {
    "__future__", "collections", "collections.abc", "random", "threading", "time",
    "typing", "math", "dataclasses", "itertools", "functools", "heapq", "array", "struct",
}
# A file named after a module we import shadows the real one, because the zip is first
# on sys.path.
SHADOW_RISK = {
    "chess.py", "types.py", "random.py", "time.py", "typing.py", "numpy.py", "numba.py",
    "threading.py", "collections.py", "torch.py", "onnxruntime.py", "math.py",
}
NATIVE_SUFFIXES = {".exe", ".dll", ".so", ".dylib", ".pyd", ".o", ".a", ".bin"}
NATIVE_MAGIC = (b"MZ", b"\x7fELF", b"\xca\xfe\xba\xbe", b"\xcf\xfa\xed\xfe")

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str) -> None:
    results.append((ok, name, detail))


def main() -> int:
    if not ZIP.exists():
        print(f"no submission.zip at {ZIP} - run `uv run python -m harness.package` first")
        return 1

    zf = zipfile.ZipFile(ZIP)
    names = zf.namelist()
    unzipped = sum(i.file_size for i in zf.infolist())

    check("agent.py" in names, "agent.py at the zip root",
          f"entries: {', '.join(names)}")
    check(not any("/" in n or "\\" in n for n in names), "nothing inside a folder",
          "all entries are at the root")
    check(unzipped <= 50 * 1024 * 1024, "50 MB unzipped at most",
          f"{unzipped / 1024:.0f} KB unzipped, {ZIP.stat().st_size / 1024:.0f} KB zipped")
    check(not [n for n in names if Path(n).name in SHADOW_RISK], "no module-shadowing filenames",
          "no chess.py / types.py / random.py etc.")

    native = []
    for n in names:
        if Path(n).suffix.lower() in NATIVE_SUFFIXES:
            native.append(n)
            continue
        head = zf.read(n)[:4]
        if any(head.startswith(m) for m in NATIVE_MAGIC):
            native.append(n)
    check(not native, "no native binaries", native or "every entry is source or text")

    source = zf.read("agent.py").decode("utf-8")
    tree = ast.parse(source)

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    outside = sorted(
        i for i in imports
        if i.split(".")[0] not in ALLOWED_PACKAGES and i not in STDLIB_OK
    )
    check(not outside, "imports stay inside the fixed environment",
          outside or ", ".join(sorted(imports)))

    banned = {"socket", "urllib", "http", "requests", "subprocess", "ftplib", "telnetlib",
              "asyncio", "multiprocessing", "ctypes", "os", "sys", "pathlib", "shutil"}
    reached = sorted(i for i in imports if i.split(".")[0] in banned)
    check(not reached, "no network, subprocess or filesystem reach",
          reached or "none imported")

    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "get_move"), None)
    args = [a.arg for a in fn.args.args] if fn else []
    check(args == ["fen", "time_left_ms"], "get_move(fen, time_left_ms)",
          f"signature: {args or 'MISSING'}")

    prints = [n for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id == "print"]
    bounded = all(isinstance(p.args[0], ast.Subscript) for p in prints if p.args)
    check(bounded, "output stays under the 4 KB per-move cap",
          f"{len(prints)} print call(s), each truncated")

    engines = [w for w in ("stockfish", "lc0", "leela", "maia", "komodo")
               if w in source.lower()]
    check(not engines, "no third-party engine in the submission",
          engines or "no engine name appears in agent.py")

    zipped_agent = source.replace("\r\n", "\n")
    on_disk = AGENT.read_text(encoding="utf-8").replace("\r\n", "\n")
    check(zipped_agent == on_disk, "zip matches the working agent.py",
          "identical" if zipped_agent == on_disk else "THE ZIP IS STALE - rebuild it")

    width = max(len(name) for _, name, _ in results)
    failed = 0
    print(f"\n  submission.zip - {ZIP.stat().st_size / 1024:.0f} KB\n")
    for ok, name, detail in results:
        if not ok:
            failed += 1
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name.ljust(width)}  {detail}")
    print()
    if failed:
        print(f"  {failed} check(s) failed - do not upload this\n")
        return 1
    print("  ready to upload\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
