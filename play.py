# ruff: noqa: E501 - the page below is an HTML template; wrapping it would only hurt it.
"""Play against the agent in your browser.

Runs a small local server that drives the real agent.py - the same get_move the platform
calls, on a real clock - and serves a board you can click. Nothing leaves your machine.

    uv run python play.py
    python play.py            # if you are not using uv

Then open http://127.0.0.1:8765 (it tries to open itself).
"""

from __future__ import annotations

import contextlib
import http.server
import importlib.util
import json
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path
from types import ModuleType
from typing import Any

import chess

PORT = 8765
HERE = Path(__file__).resolve().parent

# Engine clock presets, in milliseconds. The first is the competition time control.
LEVELS = {
    "full": (120_000, 500),
    "30s": (30_000, 300),
    "10s": (10_000, 100),
    "2s": (2_000, 50),
}


def load_agent() -> ModuleType:
    """Import a fresh copy of agent.py, the way the platform does at the start of a game."""
    if not (HERE / "agent.py").exists():
        raise SystemExit(
            f"No agent.py next to this script.\n\n"
            f"  looked in: {HERE}\n\n"
            f"play.py drives the real agent, so it has to sit in the project folder\n"
            f"beside agent.py. Run it from there:\n\n"
            f'  cd "C:\\Users\\Rohan Sharma\\PycharmProjects\\aichessathon"\n'
            f"  python play.py\n"
        )
    spec = importlib.util.spec_from_file_location(
        f"agent_{time.monotonic_ns()}", HERE / "agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Game:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset("white", "full")

    def reset(self, human_colour: str, level: str) -> None:
        base, increment = LEVELS.get(level, LEVELS["full"])
        self.board = chess.Board()
        self.human = chess.WHITE if human_colour == "white" else chess.BLACK
        self.base = base
        self.increment = increment
        self.clock = float(base)
        self.agent = load_agent()
        self.last_engine_move: str | None = None
        self.last_think_ms = 0.0
        self.flagged = False

    def engine_move(self) -> None:
        """Let the agent move, on a real clock, exactly as it would in a rated game."""
        if self.board.is_game_over(claim_draw=True) or self.flagged:
            return
        started = time.monotonic()
        uci = self.agent.get_move(self.board.fen(), int(max(0, self.clock)))
        spent = (time.monotonic() - started) * 1000.0
        self.clock -= spent
        self.last_think_ms = spent
        if self.clock < 0:
            self.flagged = True
            return
        self.clock += self.increment
        move = chess.Move.from_uci(uci)
        if move not in self.board.legal_moves:
            self.flagged = True
            return
        self.board.push(move)
        self.last_engine_move = uci

    def status(self) -> str:
        if self.flagged:
            return "The engine lost on time or played an illegal move - you win."
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return "Your move." if self.board.turn == self.human else "Engine to move."
        if outcome.winner is None:
            return f"Draw - {outcome.termination.name.replace('_', ' ').lower()}."
        who = "You win" if outcome.winner == self.human else "The engine wins"
        return f"{who} - {outcome.termination.name.replace('_', ' ').lower()}."

    def state(self) -> dict[str, Any]:
        legal: dict[str, list[str]] = {}
        if self.board.turn == self.human and not self.flagged:
            for move in self.board.legal_moves:
                legal.setdefault(chess.square_name(move.from_square), []).append(
                    chess.square_name(move.to_square)
                )
        legal = {square: sorted(set(targets)) for square, targets in legal.items()}
        last = self.board.peek().uci() if self.board.move_stack else None
        return {
            "fen": self.board.fen(),
            "humanIsWhite": self.human == chess.WHITE,
            "turnIsHuman": self.board.turn == self.human and not self.flagged,
            "legal": legal,
            "status": self.status(),
            "over": self.board.is_game_over(claim_draw=True) or self.flagged,
            "clock": max(0.0, self.clock) / 1000.0,
            "thinkMs": round(self.last_think_ms),
            "lastMove": last,
            "check": self.board.is_check(),
            "moves": [m.uci() for m in self.board.move_stack],
        }


GAME = Game()

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Play the agent</title><style>
:root{--bg:#f6f5f2;--fg:#23211d;--muted:#6d675e;--light:#ebe6dc;--dark:#b08c65;
--from:#d7e3a8;--to:#c9d98a;--line:#d9d3c7;--card:#fffdf9}
@media (prefers-color-scheme:dark){:root{--bg:#1a1917;--fg:#eae6df;--muted:#9b9488;
--light:#4a453d;--dark:#7d6448;--from:#5d6b3a;--to:#6f8043;--line:#332f2a;--card:#232120}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
display:flex;justify-content:center;padding:24px 16px}
.wrap{width:100%;max-width:560px}
h1{font-size:19px;margin:0 0 2px;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13px;margin-bottom:18px}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:14px}
select,button{font:inherit;font-size:13px;padding:7px 11px;border-radius:8px;
border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--fg);color:var(--bg);border-color:var(--fg);font-weight:560}
#board{display:grid;grid-template-columns:repeat(8,1fr);aspect-ratio:1;width:100%;
border-radius:10px;overflow:hidden;border:1px solid var(--line);
box-shadow:0 1px 3px rgba(0,0,0,.09)}
.sq{display:flex;align-items:center;justify-content:center;position:relative;
font-size:clamp(26px,7.2vw,42px);line-height:1;cursor:pointer;user-select:none}
.sq.l{background:var(--light)} .sq.d{background:var(--dark)}
.sq.from{background:var(--from)!important} .sq.to{background:var(--to)!important}
.sq.sel{box-shadow:inset 0 0 0 3px var(--fg)}
.sq .dot{position:absolute;width:26%;height:26%;border-radius:50%;
background:rgba(0,0,0,.24);pointer-events:none}
@media (prefers-color-scheme:dark){.sq .dot{background:rgba(255,255,255,.3)}}
.sq.chk{box-shadow:inset 0 0 0 3px #d05a4a}
.info{display:flex;justify-content:space-between;gap:12px;margin-top:14px;
font-size:13px;color:var(--muted);flex-wrap:wrap}
#status{color:var(--fg);font-weight:560}
#promo{display:none;gap:6px;margin-top:10px;align-items:center}
#promo.on{display:flex}
.piece-w{color:#fff;text-shadow:0 0 1.5px #000,0 0 1.5px #000,0 1px 2px rgba(0,0,0,.5)}
.piece-b{color:#141210;text-shadow:0 0 1px rgba(255,255,255,.35)}
</style></head><body><div class="wrap">
<h1>Play the agent</h1>
<div class="sub">The real <code>agent.py</code>, on a real clock. Click a piece, then its square.</div>
<div class="bar">
  <select id="colour"><option value="white">You are White</option><option value="black">You are Black</option></select>
  <select id="level">
    <option value="full">Full strength &mdash; 120s + 0.5s</option>
    <option value="30s">30s + 0.3s</option>
    <option value="10s">10s + 0.1s</option>
    <option value="2s">2s + 0.05s &mdash; weakest</option>
  </select>
  <button class="primary" id="new">New game</button>
  <button id="undo">Undo</button>
</div>
<div id="board"></div>
<div id="promo"><span>Promote to:</span>
  <button data-p="q">Queen</button><button data-p="r">Rook</button>
  <button data-p="b">Bishop</button><button data-p="n">Knight</button></div>
<div class="info"><span id="status">Loading&hellip;</span><span id="meta"></span></div>
</div><script>
const GLYPH={p:'\\u265F',n:'\\u265E',b:'\\u265D',r:'\\u265C',q:'\\u265B',k:'\\u265A'};
const FILES='abcdefgh';
let state=null, selected=null, pending=null, busy=false;

function squares(fen){
  const rows=fen.split(' ')[0].split('/'); const out={};
  rows.forEach((row,i)=>{ let file=0;
    for(const ch of row){
      if(/\\d/.test(ch)){ file+=+ch; continue; }
      out[FILES[file]+(8-i)]={t:ch.toLowerCase(), w:ch===ch.toUpperCase()}; file++;
    }});
  return out;
}
function render(){
  if(!state) return;
  const pieces=squares(state.fen);
  const flip=!state.humanIsWhite;
  const board=document.getElementById('board'); board.innerHTML='';
  const ranks=flip?[1,2,3,4,5,6,7,8]:[8,7,6,5,4,3,2,1];
  const files=flip?[...FILES].reverse():[...FILES];
  const last=state.lastMove;
  for(const rank of ranks) for(const file of files){
    const name=file+rank;
    const el=document.createElement('div');
    const dark=(FILES.indexOf(file)+rank)%2===0;
    el.className='sq '+(dark?'d':'l');
    if(last&&last.slice(0,2)===name) el.classList.add('from');
    if(last&&last.slice(2,4)===name) el.classList.add('to');
    if(selected===name) el.classList.add('sel');
    const p=pieces[name];
    if(p){
      el.textContent=GLYPH[p.t];
      el.className+=p.w?' piece-w':' piece-b';
      if(p.t==='k'&&state.check&&state.turnIsHuman) el.classList.add('chk');
    }
    if(selected&&(state.legal[selected]||[]).includes(name)){
      const dot=document.createElement('div'); dot.className='dot'; el.appendChild(dot);
    }
    el.onclick=()=>click(name);
    board.appendChild(el);
  }
  document.getElementById('status').textContent=busy?'Engine thinking\\u2026':state.status;
  const mins=Math.floor(state.clock/60), secs=Math.floor(state.clock%60);
  document.getElementById('meta').textContent=
    `engine clock ${mins}:${String(secs).padStart(2,'0')}`+
    (state.thinkMs?` \\u00b7 last move ${(state.thinkMs/1000).toFixed(1)}s`:'');
}
function click(name){
  if(busy||!state||!state.turnIsHuman) return;
  if(selected&&(state.legal[selected]||[]).includes(name)){
    const pieces=squares(state.fen); const p=pieces[selected];
    const rank=+name[1];
    if(p&&p.t==='p'&&(rank===8||rank===1)){
      pending=[selected,name]; document.getElementById('promo').classList.add('on');
      selected=null; render(); return;
    }
    send(selected+name); selected=null; return;
  }
  selected=(state.legal[name]&&state.legal[name].length)?name:null; render();
}
async function send(uci){
  busy=true; render();
  const r=await fetch('/move',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({uci})});
  state=await r.json(); busy=false; render();
}
async function post(path,body){
  busy=true; render();
  const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body||{})});
  state=await r.json(); busy=false; selected=null; render();
}
document.getElementById('promo').onclick=e=>{
  const piece=e.target.dataset.p; if(!piece||!pending) return;
  document.getElementById('promo').classList.remove('on');
  const uci=pending[0]+pending[1]+piece; pending=null; send(uci);
};
document.getElementById('new').onclick=()=>post('/new',{
  colour:document.getElementById('colour').value,
  level:document.getElementById('level').value});
document.getElementById('undo').onclick=()=>post('/undo');
fetch('/state').then(r=>r.json()).then(s=>{state=s;render();});
</script></body></html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:  # keep the console quiet
        return

    def _send(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/state"):
            with GAME.lock:
                self._send(GAME.state())
            return
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        with GAME.lock:
            if self.path.startswith("/new"):
                GAME.reset(payload.get("colour", "white"), payload.get("level", "full"))
                if GAME.board.turn != GAME.human:
                    GAME.engine_move()
            elif self.path.startswith("/undo"):
                for _ in range(2):
                    if GAME.board.move_stack:
                        GAME.board.pop()
                GAME.flagged = False
                if GAME.board.turn != GAME.human:
                    GAME.engine_move()
            elif self.path.startswith("/move"):
                try:
                    move = chess.Move.from_uci(payload["uci"])
                except ValueError:
                    move = None
                if move is not None and move in GAME.board.legal_moves:
                    GAME.board.push(move)
                    GAME.engine_move()
            self._send(GAME.state())


def main() -> None:
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as server:
        url = f"http://127.0.0.1:{PORT}"
        print(f"Playing at {url}   (Ctrl-C to stop)")
        with contextlib.suppress(Exception):
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
