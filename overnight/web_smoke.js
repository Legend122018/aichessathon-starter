/* Boot the browser port's script headlessly and play a move through it.
 *
 * Static checks did not catch the bug that shipped: a listener bound to a control that
 * had been removed threw at the bottom of the script, so the board was never drawn.
 * Running the thing catches that class directly.
 *
 *   node overnight/web_smoke.js <inline.js> <chess.min.js>
 */
const fs = require("fs");
const vm = require("vm");

const [scriptPath, chessPath] = process.argv.slice(2);
const errors = [];

function makeEl(id) {
  const el = {
    id,
    _kids: [],
    _listeners: {},
    textContent: "",
    className: "",
    value: "",
    checked: false,
    disabled: false,
    tabIndex: 0,
    scrollTop: 0,
    scrollHeight: 0,
    dataset: {},
    style: {},
    options: [],
    classList: {
      _s: new Set(),
      add(c) { this._s.add(c); },
      remove(c) { this._s.delete(c); },
      toggle(c, on) { on ? this._s.add(c) : this._s.delete(c); },
      contains(c) { return this._s.has(c); },
    },
    set innerHTML(v) { if (v === "") this._kids.length = 0; },
    get innerHTML() { return ""; },
    appendChild(c) { this._kids.push(c); return c; },
    append(...cs) { this._kids.push(...cs); },
    setAttribute() {},
    addEventListener(type, fn) { (this._listeners[type] ||= []).push(fn); },
    closest() { return this; },
  };
  return el;
}

const byId = new Map();
const document = {
  getElementById(id) {
    if (!byId.has(id)) byId.set(id, makeEl(id));
    return byId.get(id);
  },
  createElement: () => makeEl("new"),
  head: makeEl("head"),
  body: makeEl("body"),
  addEventListener() {},
};

const sandbox = {
  document,
  window: {},
  performance: { now: () => Date.now() },
  setTimeout, clearTimeout, setInterval, clearInterval,
  console,
  Math, Date, JSON, Promise, Array, Object, String, Number, Set, Map, parseInt, parseFloat, isNaN,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

// chess.js first, exactly as the page loads it
vm.runInContext(fs.readFileSync(chessPath, "utf8"), sandbox);
if (typeof sandbox.Chess !== "function") { console.error("FAIL: chess.js did not define Chess"); process.exit(1); }

const src = fs.readFileSync(scriptPath, "utf8")
  + "\nglobalThis.__game = game; globalThis.__handle = handleSquare;";

try {
  vm.runInContext(src, sandbox);
} catch (e) {
  console.error("FAIL: boot threw:", e.message);
  process.exit(1);
}

const board = document.getElementById("board");
const squares = board ? board._kids.length : 0;
console.log("boot ok. squares rendered:", squares);
if (squares !== 64) { console.error("FAIL: expected 64 squares, the board was not drawn"); process.exit(1); }

console.log("clock painted:", JSON.stringify(document.getElementById("r-clock").textContent));
console.log("status:", JSON.stringify(document.getElementById("statustext").textContent));

// Play e2e4 through the real click handler, then let the engine reply.
sandbox.__handle("e2");
sandbox.__handle("e4");
const afterHuman = sandbox.__game.history();
console.log("after human click:", afterHuman);
if (afterHuman.length !== 1) { console.error("FAIL: the human move was not applied"); process.exit(1); }

setTimeout(() => {
  const h = sandbox.__game.history();
  console.log("history now:", h);
  console.log("engine clock:", document.getElementById("r-clock").textContent,
              "| budget:", document.getElementById("r-budget").textContent,
              "| depth:", document.getElementById("r-depth").textContent,
              "| eval:", document.getElementById("r-eval").textContent);
  if (h.length < 2) { console.error("FAIL: the engine never replied"); process.exit(1); }
  console.log("PASS: board drawn, move accepted, engine replied with", h[1]);
  process.exit(0);
}, 9000);
