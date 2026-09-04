"""Let arena2 run an agent that ships a package beside it, not just a lone file.

A submission may carry `engine/` or `data/` next to agent.py, and the platform puts the
zip root on sys.path so those imports resolve. arena2 loaded by file path only, so such an
agent failed at import. The per-game reset is also widened: it now finds whichever name an
agent uses to stop its background search, and asks the agent to reset itself if it exposes
a hook, before falling back to clearing the containers it recognises.
"""
import pathlib

p = pathlib.Path(__file__).resolve().parent / "arena2.py"
s = p.read_text(encoding="utf-8")


def swap(old, new, label):
    global s
    n = s.count(old)
    assert n == 1, f"{label}: matched {n}"
    s = s.replace(old, new)


swap(
    """    pin(cores)
    spec = importlib.util.spec_from_file_location("arena_agent", path)""",
    """    pin(cores)
    # The platform puts the zip root first on sys.path, so an agent that ships a package
    # beside it imports cleanly. Do the same here or such an agent cannot be measured.
    root = str(pathlib.Path(path).resolve().parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("arena_agent", path)""",
    "sys-path",
)

swap(
    """        if message[0] == "reset":
            stop = getattr(module, "_ponder_stop", None)
            if stop is not None:
                stop()""",
    """        if message[0] == "reset":
            for name in ("_ponder_stop", "_stop_ponder"):
                stop = getattr(module, name, None)
                if callable(stop):
                    stop()
            reset_hook = getattr(module, "new_game", None) or getattr(module, "reset", None)
            if callable(reset_hook):
                try:
                    reset_hook()
                except Exception:
                    pass""",
    "reset",
)

swap(
    """        if message[0] == "stop":
            # Let any background search finish before the process goes away.
            stop = getattr(module, "_ponder_stop", None)
            if stop is not None:
                stop()
            return""",
    """        if message[0] == "stop":
            # Let any background search finish before the process goes away.
            for name in ("_ponder_stop", "_stop_ponder"):
                stop = getattr(module, name, None)
                if callable(stop):
                    stop()
            return""",
    "stop",
)

if "import pathlib" not in s:
    swap("import os\n", "import os\nimport pathlib\n", "import")

p.write_text(s, encoding="utf-8")
print("arena2 can now load a package-style agent")
