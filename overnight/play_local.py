"""Play the real engine in a browser, locally, and pit it against Stockfish.

This runs `agent.py` itself - numba-compiled, full strength, on its own tournament clock -
rather than a port of it. Stockfish is the binary in this folder and never touches the
submission; it is an opponent and a yardstick, which the rules allow.

    python overnight/play_local.py
    python overnight/play_local.py --port 8123 --stockfish overnight/stockfish.exe

Then open the address it prints. First move is slow: the engine compiles at import, which
is the same 60 second budget it gets on the platform.

Modes in the page:
  - play the engine yourself, on its 120s + 0.5s tournament clock
  - watch the engine play Stockfish at any Elo from 1320 to 3190
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess
import chess.engine

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# The agent is a single shared workspace with a background ponder thread; one move at a
# time, always. The lock is what makes a browser with two tabs harmless rather than
# a source of corrupted searches.
AGENT_LOCK = threading.Lock()
AGENT = None
SF = None
SF_LOCK = threading.Lock()


def load_agent(path: Path):
    spec = importlib.util.spec_from_file_location("live_agent", str(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["live_agent"] = module
    started = time.monotonic()
    spec.loader.exec_module(module)
    took = time.monotonic() - started
    ready = getattr(module, "JIT_READY", False)
    print(f"  agent.py loaded in {took:.1f}s, JIT {'ready' if ready else 'OFF'}", flush=True)
    if not ready:
        reason = getattr(module, "JIT_ERROR", "")
        print(f"  warning: running without the JIT ({reason})", flush=True)
    return module


def reset_agent() -> None:
    """Forget the previous game. Mirrors what a fresh process gives it on the platform."""
    stop = getattr(AGENT, "_ponder_stop", None)
    if stop is not None:
        stop()
    for name in ("SEEN", "TT", "HISTORY", "COUNTER"):
        holder = getattr(AGENT, name, None)
        if isinstance(holder, dict):
            holder.clear()
    fens = getattr(AGENT, "HISTORY_FENS", None)
    if isinstance(fens, list):
        del fens[:]
    state = getattr(AGENT, "JIT_STATE", {})
    work = state.get("work") if isinstance(state, dict) else None
    if work is not None:
        for key in ("tt_key", "tt_move", "tt_score", "tt_meta", "killers", "hist_heur", "counter"):
            if key in work:
                work[key][:] = 0


def agent_move(fen: str, time_left_ms: int) -> dict:
    with AGENT_LOCK:
        started = time.monotonic()
        uci = AGENT.get_move(fen, int(time_left_ms))
        spent = (time.monotonic() - started) * 1000.0
    ponder = getattr(AGENT, "PONDER", None)
    return {
        "uci": uci,
        "ms": round(spent, 1),
        "ponder_hits": (ponder or {}).get("hits", 0),
        "ponder_tries": (ponder or {}).get("tries", 0),
    }


def stockfish_move(fen: str, elo: int, movetime_ms: int) -> dict:
    with SF_LOCK:
        SF.configure({"UCI_LimitStrength": True, "UCI_Elo": int(elo)})
        board = chess.Board(fen)
        started = time.monotonic()
        result = SF.play(board, chess.engine.Limit(time=movetime_ms / 1000.0))
        spent = (time.monotonic() - started) * 1000.0
    return {"uci": result.move.uci() if result.move else None, "ms": round(spent, 1)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the console readable
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/chess.js":
            lib = HERE / "chess.min.js"
            if lib.exists():
                self._send(200, lib.read_bytes(), "application/javascript")
            else:
                self._json({"error": "chess.min.js missing next to play_local.py"}, 404)
        elif self.path == "/api/state":
            self._json({
                "jit": bool(getattr(AGENT, "JIT_READY", False)),
                "stockfish": SF is not None,
            })
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, 400)
            return
        try:
            if self.path == "/api/newgame":
                reset_agent()
                self._json({"ok": True})
            elif self.path == "/api/agent":
                self._json(agent_move(payload["fen"], payload["time_left_ms"]))
            elif self.path == "/api/stockfish":
                if SF is None:
                    self._json({"error": "Stockfish was not started"}, 503)
                    return
                self._json(stockfish_move(payload["fen"], payload.get("elo", 2000),
                                          payload.get("movetime_ms", 300)))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:  # a failed move should not take the server down
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


PAGE = r"""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chessathon Engine - local</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Serif+4:opsz,wght@8..60,400&family=JetBrains+Mono:wght@400;700&display=swap">
<script src="/chess.js"></script>
<style>
:root{--ground:#E8E9E6;--panel:#F4F5F2;--panel-2:#DDDFD9;--ink:#1B1F1D;--ink-2:#535A55;
 --ink-3:#828A83;--line:#C6CAC2;--brass:#8A6A2F;--brass-hi:#B08A46;--sq-light:#C9CABE;
 --sq-dark:#68766F;--piece-w:#FBFBF8;--piece-b:#20262A;--bad:#8C4232;--sel:#B08A46;}
@media (prefers-color-scheme:dark){:root{--ground:#161A1D;--panel:#1D2226;--panel-2:#252B30;
 --ink:#E6E8E4;--ink-2:#A2AAA6;--ink-3:#727B78;--line:#2E353A;--brass:#C69A4E;
 --brass-hi:#DDB86A;--sq-light:#8E968C;--sq-dark:#4A5654;--piece-w:#F2F3EE;
 --piece-b:#14181B;--bad:#D08A76;--sel:#C69A4E;}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:"Source Serif 4",Georgia,serif;}
.wrap{max-width:1060px;margin:0 auto;padding:26px 20px 50px;}
h1{font-family:Archivo,sans-serif;font-size:26px;font-weight:700;
 letter-spacing:-.02em;margin:0 0 4px;}
.sub{color:var(--ink-2);font-size:14px;margin:0 0 22px;}
.main{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:26px;align-items:start;}
@media (max-width:860px){.main{grid-template-columns:minmax(0,1fr)}}
.board{display:grid;grid-template-columns:repeat(8,1fr);aspect-ratio:1;width:100%;
 border:1px solid var(--line);}
.sq{position:relative;display:flex;align-items:center;justify-content:center;
 font-size:clamp(24px,6vw,48px);line-height:1;cursor:pointer;user-select:none;}
.sq.l{background:var(--sq-light)}.sq.d{background:var(--sq-dark)}
.sq.sel{outline:3px solid var(--sel);outline-offset:-3px}
.sq.last{box-shadow:inset 0 0 0 3px rgba(176,138,70,.55)}
.sq.chk{background:var(--bad)!important}
.sq .dot{position:absolute;width:26%;height:26%;border-radius:50%;
 background:rgba(176,138,70,.7);pointer-events:none}
.pc{pointer-events:none}
.pc.w{color:var(--piece-w);text-shadow:-1px 0 0 var(--piece-b),
 1px 0 0 var(--piece-b),0 -1px 0 var(--piece-b),0 1px 0 var(--piece-b)}
.pc.b{color:var(--piece-b)}
.status{font-family:Archivo,sans-serif;font-weight:600;font-size:15px;margin:12px 0;min-height:22px}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}
button,select{font-family:Archivo,sans-serif;font-size:13px;font-weight:600;background:var(--panel);
 color:var(--ink);border:1px solid var(--line);padding:7px 12px;border-radius:3px;cursor:pointer}
button:hover:not(:disabled),select:hover{border-color:var(--brass)}
button:disabled{opacity:.45;cursor:not-allowed}
button.primary{background:var(--brass);color:var(--panel);border-color:var(--brass)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:4px;
 padding:13px 14px;margin-bottom:16px}
.card h3{font-family:Archivo,sans-serif;font-size:11px;letter-spacing:.11em;
 text-transform:uppercase;
 color:var(--ink-3);margin:0 0 10px;font-weight:600}
.readout{display:grid;grid-template-columns:auto 1fr;gap:5px 12px;
 font-family:"JetBrains Mono",monospace;
 font-size:12.5px;font-variant-numeric:tabular-nums}
.readout dt{color:var(--ink-3)}.readout dd{margin:0;text-align:right}
.hint{font-size:12.5px;color:var(--ink-3);margin:8px 0 0;line-height:1.45}
input[type=range]{width:100%;accent-color:var(--brass)}
.moves{font-family:"JetBrains Mono",monospace;font-size:12px;max-height:200px;overflow-y:auto;
 display:grid;grid-template-columns:2.2em 1fr 1fr;gap:2px 8px}
.moves .n{color:var(--ink-3)}
.warn{background:var(--panel);border-left:3px solid var(--bad);padding:10px 13px;font-size:13px;
 color:var(--ink-2);margin-bottom:16px}
</style>
<div class="wrap">
<h1>Chessathon Engine &mdash; local</h1>
<p class="sub">The real <code>agent.py</code>, numba-compiled and at full strength, on its own
120&nbsp;second tournament clock. Not a port.</p>
<div id="warn" class="warn" hidden></div>
<div class="main">
  <div>
    <div class="board" id="board"></div>
    <div class="status" id="status">Loading&hellip;</div>
    <div class="controls">
      <button class="primary" id="newgame">New game</button>
      <button id="flip">Play as Black</button>
      <select id="mode">
        <option value="human">You vs the engine</option>
        <option value="sf">Engine vs Stockfish</option>
      </select>
      <button id="stop" disabled>Stop</button>
    </div>
  </div>
  <div>
    <div class="card">
      <h3>Engine</h3>
      <dl class="readout">
        <dt>clock</dt><dd id="r-clock">2:00.0</dd>
        <dt>last move</dt><dd id="r-move">&mdash;</dd>
        <dt>spent</dt><dd id="r-spent">&mdash;</dd>
        <dt>ponder</dt><dd id="r-ponder">&mdash;</dd>
      </dl>
      <p class="hint">It budgets its own time from the clock, so an opening move takes about
      five and a half seconds. A flag loses, exactly as it would in a rated game.</p>
    </div>
    <div class="card">
      <h3>Stockfish opponent</h3>
      <div style="display:flex;justify-content:space-between;
        font-family:Archivo,sans-serif;font-size:13px;">
        <span>Elo</span><strong id="elolabel"
          style="font-family:'JetBrains Mono',monospace;color:var(--brass)">2000</strong>
      </div>
      <input type="range" id="elo" min="1320" max="3190" step="10" value="2000">
      <p class="hint">Stockfish 18&rsquo;s own <code>UCI_Elo</code> range. Pick a level,
      switch the mode
      to <em>Engine vs Stockfish</em> and press New game to watch them play.</p>
    </div>
    <div class="card"><h3>Moves</h3><div class="moves" id="moves"></div></div>
  </div>
</div>
</div>
<script>
const GLYPH={p:"♟",n:"♞",b:"♝",r:"♜",q:"♛",k:"♚"};
const BASE=120000, INC=500;
const game=new Chess();
let human="w", sel=null, targets=[], last=null, busy=false, running=false;
let clock=BASE;
const $=id=>document.getElementById(id);

function fmt(ms){if(ms<0)ms=0;const t=ms/1000,m=Math.floor(t/60),s=t-m*60;
  return m+":"+(s<10?"0":"")+s.toFixed(1);}
function paint(){$("r-clock").textContent=fmt(clock);}

function render(){
  const b=game.board();
  const rows=human==="w"?[0,1,2,3,4,5,6,7]:[7,6,5,4,3,2,1,0];
  const cols=human==="w"?[0,1,2,3,4,5,6,7]:[7,6,5,4,3,2,1,0];
  const el=$("board"); el.innerHTML="";
  let chk=null;
  if(game.in_check())for(let r=0;r<8;r++)for(let c=0;c<8;c++){const p=b[r][c];
    if(p&&p.type==="k"&&p.color===game.turn())chk="abcdefgh"[c]+(8-r);}
  for(const r of rows)for(const c of cols){
    const name="abcdefgh"[c]+(8-r), p=b[r][c];
    const d=document.createElement("div");
    d.className="sq "+((r+c)%2===0?"l":"d"); d.dataset.sq=name;
    if(sel===name)d.classList.add("sel");
    if(last&&(last.from===name||last.to===name))d.classList.add("last");
    if(chk===name)d.classList.add("chk");
    if(p){const s=document.createElement("span");s.className="pc "+p.color;
      s.textContent=GLYPH[p.type];d.appendChild(s);}
    if(targets.includes(name)){const m=document.createElement("span");
      m.className="dot";d.appendChild(m);}
    el.appendChild(d);
  }
  const h=game.history(), mv=$("moves"); mv.innerHTML="";
  for(let i=0;i<h.length;i+=2){
    const n=document.createElement("span");n.className="n";n.textContent=(i/2+1)+".";
    const w=document.createElement("span");w.textContent=h[i]||"";
    const bl=document.createElement("span");bl.textContent=h[i+1]||"";
    mv.append(n,w,bl);
  }
  mv.scrollTop=mv.scrollHeight;
}
function say(t){$("status").innerHTML=t;}
function overText(){
  if(game.in_checkmate())return "Checkmate &mdash; "+(game.turn()==="w"?"Black":"White")+" wins.";
  if(game.in_stalemate())return "Stalemate.";
  if(game.in_threefold_repetition())return "Draw by repetition.";
  if(game.insufficient_material())return "Draw &mdash; insufficient material.";
  if(game.in_draw())return "Draw by the fifty-move rule.";
  return null;
}
async function post(path,body){
  const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body||{})});
  return r.json();
}
async function engineMove(){
  const t0=performance.now();
  const r=await post("/api/agent",{fen:game.fen(),time_left_ms:Math.round(clock)});
  if(r.error){say("Engine error: "+r.error);return null;}
  clock-=(performance.now()-t0);
  if(clock<=0){clock=0;paint();say("The engine flagged.");return null;}
  clock+=INC; paint();
  $("r-spent").textContent=Math.round(r.ms)+" ms";
  $("r-ponder").textContent=r.ponder_tries?`${r.ponder_hits}/${r.ponder_tries}`:"—";
  return r.uci;
}
async function sfMove(){
  const r=await post("/api/stockfish",
    {fen:game.fen(),elo:parseInt($("elo").value,10),movetime_ms:400});
  if(r.error){say("Stockfish error: "+r.error);return null;}
  return r.uci;
}
function apply(uci){
  const mv=game.move({from:uci.slice(0,2),to:uci.slice(2,4),
    promotion:uci.length>4?uci[4]:undefined});
  if(!mv)return false;
  last={from:mv.from,to:mv.to}; $("r-move").textContent=mv.san; render(); return true;
}
async function engineTurn(){
  busy=true; say("The engine is thinking&hellip;");
  const uci=await engineMove();
  busy=false;
  if(!uci||!apply(uci))return;
  const o=overText(); say(o||("Your move &mdash; you are "+(human==="w"?"White":"Black")+"."));
}
async function watchLoop(){
  running=true; $("stop").disabled=false;
  while(running&&!game.game_over()){
    if(game.turn()==="w"){
      say("The engine is thinking&hellip;");
      const u=await engineMove(); if(!u||!running)break; if(!apply(u))break;
    }else{
      say("Stockfish ("+$("elo").value+") is thinking&hellip;");
      const u=await sfMove(); if(!u||!running)break; if(!apply(u))break;
    }
  }
  running=false; $("stop").disabled=true;
  say(overText()||"Stopped.");
}
$("board").addEventListener("click",e=>{
  if(busy||running||$("mode").value==="sf")return;
  const cell=e.target.closest(".sq"); if(!cell)return;
  const name=cell.dataset.sq;
  if(game.game_over()||game.turn()!==human)return;
  if(sel&&targets.includes(name)){
    const opts=game.moves({square:sel,verbose:true}).filter(m=>m.to===name);
    const promo=opts.some(m=>m.promotion)?"q":undefined;
    const mv=game.move({from:sel,to:name,promotion:promo});
    if(mv){last={from:mv.from,to:mv.to};sel=null;targets=[];render();
      const o=overText(); if(o){say(o);return;} engineTurn();}
    return;
  }
  const p=game.get(name);
  if(p&&p.color===human){sel=name;targets=game.moves({square:name,verbose:true}).map(m=>m.to);}
  else{sel=null;targets=[];}
  render();
});
async function newGame(){
  running=false; await post("/api/newgame",{});
  game.reset(); sel=null;targets=[];last=null;clock=BASE;paint();
  $("r-move").textContent="—"; $("r-spent").textContent="—"; $("r-ponder").textContent="—";
  render();
  if($("mode").value==="sf"){ watchLoop(); }
  else if(human==="b"){ engineTurn(); }
  else say("Your move &mdash; you are White.");
}
$("newgame").addEventListener("click",newGame);
$("stop").addEventListener("click",()=>{running=false;$("stop").disabled=true;say("Stopping&hellip;");});
$("flip").addEventListener("click",e=>{
  human=human==="w"?"b":"w";
  e.target.textContent=human==="w"?"Play as Black":"Play as White"; newGame();
});
$("elo").addEventListener("input",()=>{$("elolabel").textContent=$("elo").value;});
fetch("/api/state").then(r=>r.json()).then(s=>{
  const msgs=[];
  if(!s.jit)msgs.push("The engine is running <strong>without its JIT</strong>, "
    +"so it is much weaker than the submission.");
  if(!s.stockfish)msgs.push("Stockfish was not found, so <em>Engine vs Stockfish</em> "
    +"will not work. Pass <code>--stockfish</code>.");
  if(msgs.length){$("warn").innerHTML=msgs.join("<br>");$("warn").hidden=false;}
  say("Your move &mdash; you are White.");
});
paint(); render();
</script>
"""


def main() -> None:
    global AGENT, SF
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--agent", default=str(ROOT / "agent.py"))
    parser.add_argument("--stockfish", default=str(HERE / "stockfish-windows-x86-64-avx2.exe"))
    args = parser.parse_args()

    print("loading the engine (it compiles at import, same as on the platform)...", flush=True)
    AGENT = load_agent(Path(args.agent))

    if os.path.exists(args.stockfish):
        try:
            SF = chess.engine.SimpleEngine.popen_uci(args.stockfish)
            SF.configure({"Threads": 1, "Hash": 64})
            name = SF.id.get("name", "Stockfish")
            print(f"  {name} ready as the sparring opponent", flush=True)
        except Exception as exc:
            print(f"  Stockfish would not start ({exc}); that mode is off", flush=True)
    else:
        print(f"  no Stockfish at {args.stockfish}; that mode is off", flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  open  http://127.0.0.1:{args.port}\n  ctrl-c to stop\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if SF is not None:
            SF.quit()


if __name__ == "__main__":
    main()
