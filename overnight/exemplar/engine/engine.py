"""Engine driver: owns the state arrays, runs iterative deepening, manages time.

The iterative-deepening loop lives in Python (one call per depth, so the
overhead is nil) because time-management decisions need to happen between
iterations.  Everything inside a single depth is compiled numba code, which
runs with the GIL released -- that is what lets a watchdog thread stop it.
"""
import threading
import time
import numpy as np

from . import board as B
from .board import (new_state, set_fen, move_to_uci, uci_to_move, gen_moves,
                    make_move, unmake_move, NO_MOVE, MT_PROMO)
from .layout import (MAX_PLY, MAX_MOVES, UN_N, PVSTRIDE, tt_alloc,
                     new_int_state, new_ctl_state,
                     OI_MBUF, OI_PV, OI_PVLEN, OI_ROOTMV, OI_ROOTSC, OI_ROOTNODE,
                     ON_NODES, ON_QNODES, ON_SELDEP, ON_STOP, ON_TTAGE, ON_ROOTN,
                     ON_BESTMV, ON_BESTSC, ON_DEPTH, ON_SPARAM, ON_NODELIM)
from .evaluate import install_params, evaluate
from . import search as S
from .search import (INF, MATE, MATE_IN_MAX, VALUE_NONE, SPI, SP_DEFAULTS,
                     root_init, root_sort, root_order_initial, search_root,
                     age_history, clear_heuristics, tt_probe, tt_move,
                     negamax, qsearch)


class SearchInfo:
    __slots__ = ("depth", "seldepth", "score", "nodes", "time_ms", "pv",
                 "bestmove", "mate", "stable", "best_frac", "completed")

    def __init__(self):
        self.depth = 0
        self.seldepth = 0
        self.score = 0
        self.nodes = 0
        self.time_ms = 0.0
        self.pv = []
        self.bestmove = None
        self.mate = None
        self.stable = 0
        self.best_frac = 0.0
        self.completed = 0


class Engine:
    def __init__(self, tt_bits=22, params=None):
        self.pos, self.sq, self.hist = new_state()
        self.undo = np.zeros(MAX_PLY * UN_N, dtype=np.uint64)
        self.tt, self.tt_size = tt_alloc(tt_bits)
        self.I = new_int_state()
        install_params(self.I, params)
        self.N = new_ctl_state()
        self.set_defaults()
        self._timer = None
        self.root_fen = None
        self.last_info = None       # SearchInfo from the most recent search() call

    # ---------------------------------------------------------------- config
    def set_defaults(self):
        for k, v in SP_DEFAULTS.items():
            self.N[ON_SPARAM + SPI[k]] = v

    def set_param(self, name, value):
        self.N[ON_SPARAM + SPI[name]] = value

    def get_param(self, name):
        return int(self.N[ON_SPARAM + SPI[name]])

    def new_game(self):
        self.tt[:] = 0
        self.N[ON_TTAGE] = 0
        clear_heuristics(self.I)

    # ------------------------------------------------------------- position
    def set_position(self, fen, moves=()):
        set_fen(self.pos, self.sq, self.hist, fen)
        self.root_fen = fen
        for u in moves:
            mv = uci_to_move(self.pos, self.sq, self.hist, u)
            if mv == NO_MOVE:
                raise ValueError(f"illegal move {u} in position {fen}")
            make_move(self.pos, self.sq, self.undo, 0, mv, self.hist)

    def push_uci(self, u):
        mv = uci_to_move(self.pos, self.sq, self.hist, u)
        if mv == NO_MOVE:
            return False
        make_move(self.pos, self.sq, self.undo, 0, mv, self.hist)
        return True

    def legal_moves(self):
        n = gen_moves(self.pos, self.sq, self.I, OI_MBUF, 0)
        return [move_to_uci(int(self.I[OI_MBUF + i])) for i in range(n)]

    def static_eval(self):
        return int(evaluate(self.pos, self.sq, self.I))

    # --------------------------------------------------------------- search
    def _pv_list(self):
        ln = int(self.I[OI_PVLEN])
        return [move_to_uci(int(self.I[OI_PV + i])) for i in range(ln)]

    def _arm_timer(self, seconds):
        self.N[ON_STOP] = 0
        if seconds is None:
            self._timer = None
            return
        t = threading.Timer(max(0.0, seconds), self._fire_stop)
        t.daemon = True
        t.start()
        self._timer = t

    def _fire_stop(self):
        self.N[ON_STOP] = 1

    def stop(self):
        """Cooperatively signal a search running in another thread to halt.

        Safe to call from any thread: it's the same flag write the internal
        watchdog Timer already performs from its own thread (`_fire_stop`),
        just triggered manually. Used for pondering, where the search has no
        time limit and must be stopped externally when the real move arrives.
        """
        self.N[ON_STOP] = 1

    def _disarm(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def search(self, max_depth=64, hard_ms=None, soft_ms=None, nodes=None,
               on_iter=None, quiet=True):
        """Iterative deepening.  hard_ms stops mid-iteration, soft_ms only
        between iterations (so a started iteration usually finishes)."""
        info = SearchInfo()
        t0 = time.perf_counter()

        self.N[ON_NODES] = 0
        self.N[ON_QNODES] = 0
        self.N[ON_SELDEP] = 0
        self.N[ON_BESTMV] = NO_MOVE
        self.N[ON_BESTSC] = 0
        self.N[ON_TTAGE] = (int(self.N[ON_TTAGE]) + 1) & 0x3F

        n = int(root_init(self.pos, self.sq, self.I, self.N))
        if n == 0:
            info.bestmove = None
            self.last_info = info
            return info
        if n == 1:
            # forced: still peek one ply so the score is meaningful, but cheap
            info.bestmove = move_to_uci(int(self.I[OI_ROOTMV]))
            info.pv = [info.bestmove]
            info.depth = 1
            self.last_info = info
            return info

        slot = tt_probe(self.tt, self.pos[B.IKEY])
        ttmv = int(tt_move(self.tt[self.tt_size + slot])) if slot >= 0 else NO_MOVE
        root_order_initial(self.pos, self.sq, self.I, self.N, ttmv)

        self._arm_timer(None if hard_ms is None else hard_ms / 1000.0)
        prev_score = 0
        prev_best = None
        stable = 0
        use_asp = int(self.N[ON_SPARAM + SPI["USE_ASPIRATION"]])
        asp_w = int(self.N[ON_SPARAM + SPI["ASP_WINDOW"]])

        try:
            for depth in range(1, max_depth + 1):
                if depth <= 4 or not use_asp:
                    alpha, beta, delta = -INF, INF, asp_w
                else:
                    delta = asp_w
                    alpha = max(prev_score - delta, -INF)
                    beta = min(prev_score + delta, INF)

                while True:
                    score = int(search_root(self.pos, self.sq, self.hist,
                                            self.undo, self.tt, self.I, self.N,
                                            depth, alpha, beta))
                    if self.N[ON_STOP]:
                        break
                    if score <= alpha:
                        beta = (alpha + beta) // 2
                        alpha = max(score - delta, -INF)
                        delta += delta // 2 + 3
                    elif score >= beta:
                        beta = min(score + delta, INF)
                        delta += delta // 2 + 3
                    else:
                        break
                    if nodes is not None and int(self.N[ON_NODES]) >= nodes:
                        self.N[ON_STOP] = 1
                        break

                if self.N[ON_STOP] and depth > 1:
                    break

                root_sort(self.I, self.N)
                prev_score = score
                bm = move_to_uci(int(self.I[OI_ROOTMV]))
                stable = stable + 1 if bm == prev_best else 0
                prev_best = bm

                info.depth = depth
                info.completed = depth
                info.seldepth = int(self.N[ON_SELDEP])
                info.score = score
                info.nodes = int(self.N[ON_NODES])
                info.time_ms = (time.perf_counter() - t0) * 1000.0
                info.pv = self._pv_list()
                info.bestmove = bm
                info.stable = stable
                tot = max(1, int(self.N[ON_NODES]))
                info.best_frac = int(self.I[OI_ROOTNODE]) / tot
                if abs(score) >= MATE_IN_MAX:
                    info.mate = (MATE - abs(score) + 1) // 2 * (1 if score > 0 else -1)
                else:
                    info.mate = None

                if on_iter is not None:
                    on_iter(info)
                if not quiet:
                    # Sliced because the competition caps a move's output at 4 KB and
                    # forfeits the game past it. This is unreachable as shipped -
                    # `quiet` defaults to True and the agent never passes it - but an
                    # unbounded print on the per-iteration path is one flipped default
                    # away from losing every game, and a PV is exactly the kind of
                    # string that grows without asking.
                    print(f"info depth {depth} seldepth {info.seldepth} "
                          f"score {score} nodes {info.nodes} "
                          f"time {info.time_ms:.0f} nps "
                          f"{info.nodes/max(info.time_ms,1)*1000:.0f} "
                          f"pv {' '.join(info.pv)}"[:512], flush=True)

                if nodes is not None and int(self.N[ON_NODES]) >= nodes:
                    break
                if info.mate is not None and abs(info.mate) <= depth // 2 + 1:
                    break
                if soft_ms is not None and info.time_ms >= soft_ms:
                    break
        finally:
            self._disarm()
            self.N[ON_STOP] = 0

        if info.bestmove is None:
            info.bestmove = move_to_uci(int(self.I[OI_ROOTMV]))
        info.nodes = int(self.N[ON_NODES])
        info.time_ms = (time.perf_counter() - t0) * 1000.0
        self.last_info = info
        return info
