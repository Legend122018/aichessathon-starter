"""Alpha-beta search: PVS + transposition table + the standard pruning family.

All mutable state is passed in (numba freezes module globals read-only), packed
into a small number of arrays -- see engine/layout.py.  Read-only lookup tables
stay as frozen globals, which costs nothing.

Every pruning technique is gated by a runtime flag in N[ON_SPARAM + SP_*], so
features can be toggled and measured one at a time without recompiling.
"""
import numpy as np
from numba import njit, uint64, uint8, int64, int32, boolean, void, types

from .tables import (
    U, ONE, FULL,
    WP, WN, WB, WR, WQ, WK, BP, BN, BB_, BR, BQ, BK, EMPTY,
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK,
    lsb, popcount, rook_attacks, bishop_attacks, queen_attacks,
)
from .board import (
    POS_N, IOCC_W, IOCC_A, ISIDE, IEP, IHM, IKEY, IPHASE, IHIDX, IPLY,
    MT_NORMAL, MT_PROMO, MT_EP, MT_CASTLE, NO_MOVE, PHASE_TOTAL,
    gen_moves, make_move, unmake_move, make_null, unmake_null,
    attackers_to, in_check,
)
from .evaluate import evaluate, non_pawn_material
from .layout import (
    MAX_PLY, MAX_MOVES, MAX_HIST, UN_N, PVSTRIDE,
    OI_EVALP, OI_MBUF, OI_MSCR, OI_SBUF, OI_SSCR, OI_KILL, OI_HISTORY,
    OI_CAPHIST, OI_COUNTER, OI_CONTHIST, OI_SSEVAL, OI_SSMOVE, OI_SSPIECE,
    OI_SSEXCL, OI_PV, OI_PVLEN, OI_QLIST, OI_CLIST, OI_ROOTMV, OI_ROOTSC,
    OI_ROOTNODE, OI_ROOTPV,
    ON_NODES, ON_QNODES, ON_SELDEP, ON_STOP, ON_TTAGE, ON_ROOTN, ON_BESTMV,
    ON_BESTSC, ON_DEPTH, ON_TBHIT, ON_PONDER, ON_NODELIM, ON_SPARAM,
)

INF = 32001
MATE = 32000
MATE_IN_MAX = MATE - MAX_PLY
VALUE_NONE = 32002
DRAW = 0
BOUND_NONE, BOUND_UPPER, BOUND_LOWER, BOUND_EXACT = 0, 1, 2, 3

SEE_VALUE = np.array([100, 325, 335, 500, 950, 10000, 0], dtype=np.int64)

# --------------------------------------------------------------------------
# runtime-tunable search parameters (indices into N[ON_SPARAM + i])
# --------------------------------------------------------------------------
SP_NAMES = [
    "USE_TT", "USE_NMP", "USE_LMR", "USE_FUTILITY", "USE_RFP", "USE_RAZOR",
    "USE_SEE_PRUNE", "USE_LMP", "USE_IIR", "USE_KILLERS", "USE_HISTORY",
    "USE_COUNTERMOVE", "USE_CHECK_EXT", "USE_SINGULAR", "USE_QSEE", "USE_DELTA",
    "USE_HIST_PRUNE", "USE_CONTHIST", "USE_ASPIRATION",
    "RFP_MARGIN", "RAZOR_MARGIN", "FUT_MARGIN", "FUT_BASE",
    "NMP_BASE", "NMP_DIV", "NMP_EVAL_DIV", "ASP_WINDOW", "DELTA_MARGIN",
    "SEE_QUIET", "SEE_CAP", "HIST_MAX", "SING_DEPTH", "SING_MARGIN",
    "HIST_PRUNE_MUL", "LMR_HIST_DIV", "RFP_DEPTH", "FUT_DEPTH",
]
SPI = {n: i for i, n in enumerate(SP_NAMES)}
SP_DEFAULTS = dict(
    USE_TT=1, USE_NMP=1, USE_LMR=1, USE_FUTILITY=1, USE_RFP=1, USE_RAZOR=1,
    USE_SEE_PRUNE=1, USE_LMP=1, USE_IIR=1, USE_KILLERS=1, USE_HISTORY=1,
    USE_COUNTERMOVE=1, USE_CHECK_EXT=1, USE_SINGULAR=1, USE_QSEE=1, USE_DELTA=1,
    USE_HIST_PRUNE=1, USE_CONTHIST=1, USE_ASPIRATION=1,
    RFP_MARGIN=75, RAZOR_MARGIN=340, FUT_MARGIN=110, FUT_BASE=90,
    NMP_BASE=4, NMP_DIV=4, NMP_EVAL_DIV=180, ASP_WINDOW=12, DELTA_MARGIN=130,
    SEE_QUIET=-55, SEE_CAP=-105, HIST_MAX=16384, SING_DEPTH=7, SING_MARGIN=2,
    HIST_PRUNE_MUL=-3200, LMR_HIST_DIV=6000, RFP_DEPTH=8, FUT_DEPTH=8,
)
for _n in SP_NAMES:
    globals()["SP_" + _n] = ON_SPARAM + SPI[_n]

SP_USE_TT = ON_SPARAM + SPI["USE_TT"]; SP_USE_NMP = ON_SPARAM + SPI["USE_NMP"]
SP_USE_LMR = ON_SPARAM + SPI["USE_LMR"]; SP_USE_FUT = ON_SPARAM + SPI["USE_FUTILITY"]
SP_USE_RFP = ON_SPARAM + SPI["USE_RFP"]; SP_USE_RAZ = ON_SPARAM + SPI["USE_RAZOR"]
SP_USE_SEEP = ON_SPARAM + SPI["USE_SEE_PRUNE"]; SP_USE_LMP = ON_SPARAM + SPI["USE_LMP"]
SP_USE_IIR = ON_SPARAM + SPI["USE_IIR"]; SP_USE_KILL = ON_SPARAM + SPI["USE_KILLERS"]
SP_USE_HIST = ON_SPARAM + SPI["USE_HISTORY"]; SP_USE_CM = ON_SPARAM + SPI["USE_COUNTERMOVE"]
SP_USE_CHKEXT = ON_SPARAM + SPI["USE_CHECK_EXT"]; SP_USE_SING = ON_SPARAM + SPI["USE_SINGULAR"]
SP_USE_QSEE = ON_SPARAM + SPI["USE_QSEE"]; SP_USE_DELTA = ON_SPARAM + SPI["USE_DELTA"]
SP_USE_HP = ON_SPARAM + SPI["USE_HIST_PRUNE"]; SP_USE_CONT = ON_SPARAM + SPI["USE_CONTHIST"]
SP_RFPM = ON_SPARAM + SPI["RFP_MARGIN"]; SP_RAZM = ON_SPARAM + SPI["RAZOR_MARGIN"]
SP_FUTM = ON_SPARAM + SPI["FUT_MARGIN"]; SP_FUTB = ON_SPARAM + SPI["FUT_BASE"]
SP_NMPB = ON_SPARAM + SPI["NMP_BASE"]; SP_NMPD = ON_SPARAM + SPI["NMP_DIV"]
SP_NMPE = ON_SPARAM + SPI["NMP_EVAL_DIV"]; SP_DELM = ON_SPARAM + SPI["DELTA_MARGIN"]
SP_SEEQ = ON_SPARAM + SPI["SEE_QUIET"]; SP_SEEC = ON_SPARAM + SPI["SEE_CAP"]
SP_HMAX = ON_SPARAM + SPI["HIST_MAX"]; SP_SINGD = ON_SPARAM + SPI["SING_DEPTH"]
SP_SINGM = ON_SPARAM + SPI["SING_MARGIN"]; SP_HPM = ON_SPARAM + SPI["HIST_PRUNE_MUL"]
SP_LMRHD = ON_SPARAM + SPI["LMR_HIST_DIV"]; SP_RFPD = ON_SPARAM + SPI["RFP_DEPTH"]
SP_FUTD = ON_SPARAM + SPI["FUT_DEPTH"]

# ---- frozen lookup tables ------------------------------------------------
LMR = np.zeros(64 * 64, dtype=np.int32)
for _d in range(64):
    for _m in range(64):
        LMR[_d * 64 + _m] = 0 if (_d == 0 or _m == 0) else \
            int(0.77 + np.log(_d) * np.log(_m) / 2.36)

LMP_TABLE = np.zeros(2 * 16, dtype=np.int32)
for _imp in range(2):
    for _d in range(16):
        LMP_TABLE[_imp * 16 + _d] = int((2.5 + 2.0 * _d * _d / 4.5) *
                                        (1.0 if _imp else 0.55)) + 2


# ==========================================================================
#  transposition table  (tt: keys in [0:size], values in [size:2*size])
# ==========================================================================
@njit(int64(uint64[:], uint64, int64), inline='always', cache=False, nogil=True)
def tt_index(tt, key, ttage):
    return int64(key & uint64((tt.shape[0] >> 1) - 4))


@njit(uint64(int64, int64, int64, int64, int64, int64),
      inline='always', cache=False, nogil=True)
def tt_pack(move, score, depth, bound, seval, age):
    return (uint64(move & 0xFFFF)
            | (uint64((score + 32768) & 0xFFFF) << U(16))
            | (uint64((depth + 16) & 0xFF) << U(32))
            | (uint64(bound & 3) << U(40))
            | (uint64(age & 0x3F) << U(42))
            | (uint64((seval + 32768) & 0xFFFF) << U(48)))


@njit(int64(uint64), inline='always', cache=False, nogil=True)
def tt_move(v):
    return int64(v & U(0xFFFF))


@njit(int64(uint64), inline='always', cache=False, nogil=True)
def tt_score(v):
    return int64((v >> U(16)) & U(0xFFFF)) - 32768


@njit(int64(uint64), inline='always', cache=False, nogil=True)
def tt_depth(v):
    return int64((v >> U(32)) & U(0xFF)) - 16


@njit(int64(uint64), inline='always', cache=False, nogil=True)
def tt_bound(v):
    return int64((v >> U(40)) & U(3))


@njit(int64(uint64), inline='always', cache=False, nogil=True)
def tt_age(v):
    return int64((v >> U(42)) & U(0x3F))


@njit(int64(uint64), inline='always', cache=False, nogil=True)
def tt_eval(v):
    return int64((v >> U(48)) & U(0xFFFF)) - 32768


@njit(int64(uint64[:], uint64), cache=False, nogil=True)
def tt_probe(tt, key):
    base = int64(key & uint64((tt.shape[0] >> 1) - 4))
    for i in range(4):
        if tt[base + i] == key:
            return base + i
    return -1


@njit(void(uint64[:], uint64, int64, int64, int64, int64, int64, int64),
      cache=False, nogil=True)
def tt_store(tt, key, move, score, depth, bound, seval, age):
    size = tt.shape[0] >> 1
    base = int64(key & uint64(size - 4))
    slot = base
    bestrep = int64(1 << 40)
    for i in range(4):
        k = tt[base + i]
        if k == key:
            old = tt[size + base + i]
            if move == NO_MOVE:
                move = tt_move(old)
            # keep a much deeper same-generation entry
            if bound != BOUND_EXACT and depth < tt_depth(old) - 3 \
                    and tt_age(old) == (age & 0x3F):
                return
            slot = base + i
            bestrep = int64(-(1 << 40))
            break
        if k == U(0):
            slot = base + i
            bestrep = int64(-(1 << 40))
            break
        v = tt[size + base + i]
        rep = tt_depth(v) - 6 * ((age - tt_age(v)) & 0x3F)
        if rep < bestrep:
            bestrep = rep
            slot = base + i
    tt[slot] = key
    tt[size + slot] = tt_pack(move, score, depth, bound, seval, age)


@njit(int64(int64, int64), inline='always', cache=False, nogil=True)
def score_to_tt(s, ply):
    if s >= MATE_IN_MAX:
        return s + ply
    if s <= -MATE_IN_MAX:
        return s - ply
    return s


@njit(int64(int64, int64), inline='always', cache=False, nogil=True)
def score_from_tt(s, ply):
    if s >= MATE_IN_MAX:
        return s - ply
    if s <= -MATE_IN_MAX:
        return s + ply
    return s


# ==========================================================================
#  static exchange evaluation
# ==========================================================================
@njit(boolean(uint64[:], uint8[:], int64, int64), cache=False, nogil=True)
def see_ge(pos, sq, mv, threshold):
    """True when the exchange sequence on this move nets >= threshold cp."""
    mt = (mv >> 14) & 3
    if mt != MT_NORMAL:
        return 0 >= threshold
    frm = mv & 63
    to = (mv >> 6) & 63
    cap = int64(sq[to])
    if cap == EMPTY:
        swap = -threshold
    else:
        swap = SEE_VALUE[cap % 6] - threshold
    if swap < 0:
        return False
    moving = int64(sq[frm])
    swap = SEE_VALUE[moving % 6] - swap
    if swap <= 0:
        return True

    occ = pos[IOCC_A] ^ (ONE << U(frm)) ^ (ONE << U(to))
    stm = moving // 6
    attackers = attackers_to(pos, to, occ)
    bishops = pos[WB] | pos[BB_] | pos[WQ] | pos[BQ]
    rooks = pos[WR] | pos[BR] | pos[WQ] | pos[BQ]
    res = int64(1)

    while True:
        stm = 1 - stm
        attackers &= occ
        stm_att = attackers & pos[IOCC_W + stm]
        if stm_att == U(0):
            break
        o = 6 * stm
        res ^= 1
        done = False
        for pt in range(6):
            b = stm_att & pos[pt + o]
            if b:
                if pt == KING:
                    # a king recapture is only allowed if nothing defends
                    if attackers & pos[IOCC_W + (1 - stm)]:
                        return res == 0
                    return res != 0
                swap = SEE_VALUE[pt] - swap
                if swap < res:
                    return res != 0
                occ ^= b & (~b + ONE)
                if pt == PAWN or pt == BISHOP or pt == QUEEN:
                    attackers |= bishop_attacks(to, occ) & bishops
                if pt == ROOK or pt == QUEEN:
                    attackers |= rook_attacks(to, occ) & rooks
                done = True
                break
        if not done:
            break
    return res != 0


# ==========================================================================
#  draws
# ==========================================================================
@njit(boolean(uint64[:], uint64[:], int64), cache=False, nogil=True)
def is_repetition(pos, hist, ply):
    hidx = int64(pos[IHIDX])
    hm = int64(pos[IHM])
    key = pos[IKEY]
    end = hidx - hm
    if end < 0:
        end = 0
    cnt = 0
    i = hidx - 2
    while i >= end:
        if hist[i] == key:
            if i > hidx - ply:
                return True          # repetition within the search tree
            cnt += 1
            if cnt >= 2:
                return True          # threefold counting game history
        i -= 2
    return False


@njit(boolean(uint64[:]), cache=False, nogil=True)
def insufficient_material(pos):
    if pos[WP] | pos[BP] | pos[WR] | pos[BR] | pos[WQ] | pos[BQ]:
        return False
    wn = popcount(pos[WN]); wb = popcount(pos[WB])
    bn = popcount(pos[BN]); bb = popcount(pos[BB_])
    if wn + wb <= 1 and bn + bb <= 1:
        return True
    if wb == 0 and bb == 0 and wn + bn <= 2 and wn <= 2 and bn <= 2:
        return True
    return False


# ==========================================================================
#  move ordering
# ==========================================================================
@njit(void(uint64[:], uint8[:], int32[:], int64[:], int64, int64, int64, int64),
      cache=False, nogil=True)
def score_moves(pos, sq, I, N, off, n, ttmv, ply):
    side = int64(pos[ISIDE])
    prev = int64(I[OI_SSMOVE + ply])
    prevpc = int64(I[OI_SSPIECE + ply])
    cm = int64(NO_MOVE)
    if N[SP_USE_CM] and prev != NO_MOVE:
        cm = int64(I[OI_COUNTER + prevpc * 64 + ((prev >> 6) & 63)])
    k0 = int64(I[OI_KILL + ply * 2]) if N[SP_USE_KILL] else int64(NO_MOVE)
    k1 = int64(I[OI_KILL + ply * 2 + 1]) if N[SP_USE_KILL] else int64(NO_MOVE)
    use_hist = N[SP_USE_HIST]
    use_cont = N[SP_USE_CONT]

    for i in range(n):
        mv = int64(I[off + i])
        if mv == ttmv:
            I[OI_MSCR - OI_MBUF + off + i] = int32(1 << 30)
            continue
        frm = mv & 63
        to = (mv >> 6) & 63
        mt = (mv >> 14) & 3
        pc = int64(sq[frm])
        victim = int64(sq[to])
        if mt == MT_PROMO:
            pr = (mv >> 12) & 3
            if pr == 3:
                s = (1 << 29) + (800 if victim != EMPTY else 0)
            else:
                s = -(1 << 22) + pr
            I[OI_MSCR - OI_MBUF + off + i] = int32(s)
        elif victim != EMPTY or mt == MT_EP:
            vv = SEE_VALUE[0] if mt == MT_EP else SEE_VALUE[victim % 6]
            mvv = 16 * vv - SEE_VALUE[pc % 6]
            cv = 6 if mt == MT_EP else victim % 6
            ch = int64(I[OI_CAPHIST + (pc * 64 + to) * 7 + cv])
            if see_ge(pos, sq, mv, -20):
                I[OI_MSCR - OI_MBUF + off + i] = int32((1 << 28) + mvv + ch // 8)
            else:
                I[OI_MSCR - OI_MBUF + off + i] = int32(-(1 << 23) + mvv + ch // 8)
        elif mv == k0:
            I[OI_MSCR - OI_MBUF + off + i] = int32((1 << 27) + 1)
        elif mv == k1:
            I[OI_MSCR - OI_MBUF + off + i] = int32(1 << 27)
        elif mv == cm:
            I[OI_MSCR - OI_MBUF + off + i] = int32(1 << 26)
        else:
            h = int64(I[OI_HISTORY + (side * 64 + frm) * 64 + to]) if use_hist else 0
            if use_cont and prev != NO_MOVE:
                h += int64(I[OI_CONTHIST +
                             ((prevpc * 64 + ((prev >> 6) & 63)) * 12 + pc) * 64 + to]) // 2
            I[OI_MSCR - OI_MBUF + off + i] = int32(h)


@njit(int64(int32[:], int64, int64, int64), inline='always', cache=False, nogil=True)
def pick_move(I, off, n, i):
    """Selection-sort step: bring the best remaining move to slot i."""
    soff = OI_MSCR - OI_MBUF + off
    best = i
    bs = I[soff + i]
    for j in range(i + 1, n):
        if I[soff + j] > bs:
            bs = I[soff + j]
            best = j
    if best != i:
        t = I[off + i]; I[off + i] = I[off + best]; I[off + best] = t
        t2 = I[soff + i]; I[soff + i] = I[soff + best]; I[soff + best] = t2
    return int64(I[off + i])


@njit(void(int32[:], int64, int64, int64, int64), inline='always', cache=False, nogil=True)
def hist_bonus(I, hmax, side, mv, bonus):
    idx = OI_HISTORY + (side * 64 + (mv & 63)) * 64 + ((mv >> 6) & 63)
    h = int64(I[idx])
    h += bonus - (h * (bonus if bonus > 0 else -bonus)) // hmax
    I[idx] = int32(h)


@njit(void(int32[:], int64, int64, int64, int64, int64),
      inline='always', cache=False, nogil=True)
def cont_bonus(I, hmax, ply, pc, mv, bonus):
    prev = int64(I[OI_SSMOVE + ply])
    if prev == NO_MOVE:
        return
    prevpc = int64(I[OI_SSPIECE + ply])
    idx = OI_CONTHIST + ((prevpc * 64 + ((prev >> 6) & 63)) * 12 + pc) * 64 + ((mv >> 6) & 63)
    h = int64(I[idx])
    h += bonus - (h * (bonus if bonus > 0 else -bonus)) // hmax
    I[idx] = int32(h)


# ==========================================================================
#  quiescence
# ==========================================================================
@njit(int64(uint64[:], uint8[:], uint64[:], uint64[:], uint64[:], int32[:],
            int64[:], int64, int64, int64), cache=False, nogil=True)
def qsearch(pos, sq, hist, undo, tt, I, N, alpha, beta, ply):
    N[ON_NODES] += 1
    N[ON_QNODES] += 1
    if (N[ON_NODES] & 1023) == 0 and N[ON_STOP]:
        return 0
    if ply > N[ON_SELDEP]:
        N[ON_SELDEP] = ply
    if ply >= MAX_PLY - 2:
        return evaluate(pos, sq, I)
    if int64(pos[IHM]) >= 100 or insufficient_material(pos) \
            or is_repetition(pos, hist, ply):
        return DRAW

    incheck = in_check(pos)
    key = pos[IKEY]
    ttmv = int64(NO_MOVE)
    tteval = int64(VALUE_NONE)
    is_pv = (beta - alpha) > 1

    if N[SP_USE_TT]:
        slot = tt_probe(tt, key)
        if slot >= 0:
            v = tt[(tt.shape[0] >> 1) + slot]
            ttmv = tt_move(v)
            tteval = tt_eval(v)
            s = score_from_tt(tt_score(v), ply)
            b = tt_bound(v)
            if not is_pv and (b == BOUND_EXACT
                              or (b == BOUND_LOWER and s >= beta)
                              or (b == BOUND_UPPER and s <= alpha)):
                return s

    if incheck:
        best = int64(-INF)
        stand = int64(-INF)
    else:
        stand = tteval if tteval != VALUE_NONE else evaluate(pos, sq, I)
        best = stand
        if best >= beta:
            return best
        if best > alpha:
            alpha = best

    off = OI_MBUF + ply * MAX_MOVES
    n = gen_moves(pos, sq, I, off, 0 if incheck else 1)
    if n == 0:
        return -MATE + ply if incheck else best
    score_moves(pos, sq, I, N, off, n, ttmv, ply)

    bestmove = int64(NO_MOVE)
    delta_m = N[SP_DELM]
    uoff = ply * UN_N
    for i in range(n):
        mv = pick_move(I, off, n, i)
        if not incheck:
            to = (mv >> 6) & 63
            mt = (mv >> 14) & 3
            victim = int64(sq[to])
            gain = int64(0)
            if victim != EMPTY:
                gain = SEE_VALUE[victim % 6]
            elif mt == MT_EP:
                gain = SEE_VALUE[0]
            if mt == MT_PROMO:
                gain += SEE_VALUE[QUEEN] - SEE_VALUE[PAWN]
            # delta pruning: even the best case cannot reach alpha
            if N[SP_USE_DELTA] and stand + gain + delta_m < alpha \
                    and alpha < MATE_IN_MAX and alpha > -MATE_IN_MAX:
                continue
            if N[SP_USE_QSEE] and not see_ge(pos, sq, mv, 0):
                continue

        I[OI_SSMOVE + ply + 1] = int32(mv)
        I[OI_SSPIECE + ply + 1] = int32(sq[mv & 63])
        make_move(pos, sq, undo, uoff, mv, hist)
        v = -qsearch(pos, sq, hist, undo, tt, I, N, -beta, -alpha, ply + 1)
        unmake_move(pos, sq, undo, uoff, mv)
        if N[ON_STOP]:
            return 0
        if v > best:
            best = v
            bestmove = mv
            if v > alpha:
                alpha = v
                if v >= beta:
                    break

    if N[SP_USE_TT]:
        bound = BOUND_LOWER if best >= beta else BOUND_UPPER
        tt_store(tt, key, bestmove, score_to_tt(best, ply), 0, bound,
                 stand if stand != -INF else VALUE_NONE, N[ON_TTAGE])
    return best


# ==========================================================================
#  main search
# ==========================================================================
@njit(int64(uint64[:], uint8[:], uint64[:], uint64[:], uint64[:], int32[:],
            int64[:], int64, int64, int64, int64, boolean), cache=False, nogil=True)
def negamax(pos, sq, hist, undo, tt, I, N, depth, alpha, beta, ply, cutnode):
    is_pv = (beta - alpha) > 1
    I[OI_PVLEN + ply] = 0

    if depth <= 0:
        return qsearch(pos, sq, hist, undo, tt, I, N, alpha, beta, ply)

    N[ON_NODES] += 1
    if (N[ON_NODES] & 1023) == 0 and N[ON_STOP]:
        return 0
    if ply > N[ON_SELDEP]:
        N[ON_SELDEP] = ply

    excl = int64(I[OI_SSEXCL + ply])

    if ply > 0:
        if is_repetition(pos, hist, ply) or int64(pos[IHM]) >= 100 \
                or insufficient_material(pos):
            return DRAW
        if ply >= MAX_PLY - 2:
            return evaluate(pos, sq, I)
        # mate distance pruning
        if alpha < -MATE + ply:
            alpha = -MATE + ply
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha

    key = pos[IKEY]
    ttmv = int64(NO_MOVE)
    ttscore = int64(VALUE_NONE)
    ttdepth = int64(-16)
    ttbound = int64(BOUND_NONE)
    tteval = int64(VALUE_NONE)
    ttsize = tt.shape[0] >> 1
    if N[SP_USE_TT] and excl == NO_MOVE:
        slot = tt_probe(tt, key)
        if slot >= 0:
            v = tt[ttsize + slot]
            ttmv = tt_move(v)
            ttdepth = tt_depth(v)
            ttbound = tt_bound(v)
            ttscore = score_from_tt(tt_score(v), ply)
            tteval = tt_eval(v)
            if not is_pv and ttdepth >= depth and ply > 0:
                if (ttbound == BOUND_EXACT
                        or (ttbound == BOUND_LOWER and ttscore >= beta)
                        or (ttbound == BOUND_UPPER and ttscore <= alpha)):
                    return ttscore

    incheck = in_check(pos)

    if incheck:
        seval = int64(VALUE_NONE)
        ev = int64(-INF)
    else:
        seval = tteval if tteval != VALUE_NONE else evaluate(pos, sq, I)
        ev = seval
        # a TT score bounded the right way beats raw static eval
        if ttscore != VALUE_NONE:
            if (ttbound == BOUND_EXACT
                    or (ttbound == BOUND_LOWER and ttscore > ev)
                    or (ttbound == BOUND_UPPER and ttscore < ev)):
                ev = ttscore
    I[OI_SSEVAL + ply] = int32(seval)

    improving = False
    if not incheck and ply >= 2 and int64(I[OI_SSEVAL + ply - 2]) != VALUE_NONE:
        improving = seval > int64(I[OI_SSEVAL + ply - 2])

    us = int64(pos[ISIDE])
    npm = non_pawn_material(pos, us)
    absb = beta if beta >= 0 else -beta

    # ---- node-level pruning ------------------------------------------------
    if not is_pv and not incheck and excl == NO_MOVE and absb < MATE_IN_MAX:
        if N[SP_USE_RFP] and depth <= N[SP_RFPD]:
            margin = N[SP_RFPM] * (depth - (1 if improving else 0))
            if ev - margin >= beta:
                return ev - margin // 2

        if N[SP_USE_RAZ] and depth <= 3 and ev + N[SP_RAZM] * depth < alpha:
            v = qsearch(pos, sq, hist, undo, tt, I, N, alpha, alpha + 1, ply)
            if v < alpha:
                return v

        if N[SP_USE_NMP] and depth >= 3 and ev >= beta and npm > 0 \
                and int64(I[OI_SSMOVE + ply]) != NO_MOVE:
            r = N[SP_NMPB] + depth // N[SP_NMPD]
            d = (ev - beta) // N[SP_NMPE]
            if d > 3:
                d = 3
            r += d
            nd = depth - r
            if nd < 1:
                nd = 1
            I[OI_SSMOVE + ply + 1] = int32(NO_MOVE)
            I[OI_SSPIECE + ply + 1] = 0
            make_null(pos, undo, ply * UN_N, hist)
            v = -negamax(pos, sq, hist, undo, tt, I, N, nd, -beta, -beta + 1,
                         ply + 1, not cutnode)
            unmake_null(pos, undo, ply * UN_N)
            if N[ON_STOP]:
                return 0
            if v >= beta:
                if v >= MATE_IN_MAX:
                    v = beta          # never return an unproven mate from null
                return v

    # internal iterative reduction
    if N[SP_USE_IIR] and ttmv == NO_MOVE and depth >= 4 and (is_pv or cutnode):
        depth -= 1

    off = OI_MBUF + ply * MAX_MOVES
    n = gen_moves(pos, sq, I, off, 0)
    if n == 0:
        return -MATE + ply if incheck else DRAW
    score_moves(pos, sq, I, N, off, n, ttmv, ply)

    best = int64(-INF)
    bestmove = int64(NO_MOVE)
    bound = int64(BOUND_UPPER)
    movecount = int64(0)
    nquiet = int64(0)
    qoff = OI_QLIST + ply * 64
    uoff = ply * UN_N
    hmax = N[SP_HMAX]
    lmp_lim = int64(1 << 30)
    if N[SP_USE_LMP] and depth < 16:
        lmp_lim = int64(LMP_TABLE[(1 if improving else 0) * 16 + depth])

    for i in range(n):
        mv = pick_move(I, off, n, i)
        if mv == excl:
            continue
        movecount += 1
        frm = mv & 63
        to = (mv >> 6) & 63
        mt = (mv >> 14) & 3
        pc = int64(sq[frm])
        victim = int64(sq[to])
        is_quiet = (victim == EMPTY and mt != MT_PROMO and mt != MT_EP)
        hscore = int64(0)
        if is_quiet:
            hscore = int64(I[OI_HISTORY + (us * 64 + frm) * 64 + to])

        # ---- shallow-depth move pruning ------------------------------------
        if ply > 0 and best > -MATE_IN_MAX and npm > 0 and not is_pv:
            if is_quiet:
                if N[SP_USE_LMP] and movecount >= lmp_lim:
                    continue
                if N[SP_USE_FUT] and not incheck and depth <= N[SP_FUTD] and \
                        ev + N[SP_FUTB] + N[SP_FUTM] * depth <= alpha:
                    continue
                if N[SP_USE_HP] and depth <= 4 and hscore < N[SP_HPM] * depth:
                    continue
                if N[SP_USE_SEEP] and depth <= 8 and \
                        not see_ge(pos, sq, mv, N[SP_SEEQ] * depth):
                    continue
            else:
                if N[SP_USE_SEEP] and depth <= 6 and \
                        not see_ge(pos, sq, mv, N[SP_SEEC] * depth):
                    continue

        # ---- singular extension --------------------------------------------
        ext = int64(0)
        ttsa = ttscore if ttscore >= 0 else -ttscore
        if N[SP_USE_SING] and ply > 0 and depth >= N[SP_SINGD] and mv == ttmv \
                and excl == NO_MOVE and ttdepth >= depth - 3 \
                and ttbound != BOUND_UPPER and ttsa < MATE_IN_MAX:
            sbeta = ttscore - N[SP_SINGM] * depth
            sdepth = (depth - 1) // 2
            # the research regenerates moves at this same ply: save the list
            for j in range(n):
                I[OI_SBUF + ply * MAX_MOVES + j] = I[off + j]
                I[OI_SSCR + ply * MAX_MOVES + j] = I[OI_MSCR - OI_MBUF + off + j]
            I[OI_SSEXCL + ply] = int32(mv)
            v = negamax(pos, sq, hist, undo, tt, I, N, sdepth, sbeta - 1, sbeta,
                        ply, cutnode)
            I[OI_SSEXCL + ply] = int32(NO_MOVE)
            for j in range(n):
                I[off + j] = I[OI_SBUF + ply * MAX_MOVES + j]
                I[OI_MSCR - OI_MBUF + off + j] = I[OI_SSCR + ply * MAX_MOVES + j]
            if N[ON_STOP]:
                return 0
            if v < sbeta:
                ext = 1
            elif sbeta >= beta:
                return sbeta                       # multi-cut
            elif ttscore >= beta:
                ext = -2
            elif cutnode:
                ext = -1

        I[OI_SSMOVE + ply + 1] = int32(mv)
        I[OI_SSPIECE + ply + 1] = int32(pc)
        make_move(pos, sq, undo, uoff, mv, hist)
        gives_check = in_check(pos)
        if N[SP_USE_CHKEXT] and gives_check and ext == 0 and depth <= 10:
            ext = 1

        newdepth = depth - 1 + ext
        if newdepth < 0:
            newdepth = 0

        if movecount == 1 and is_pv:
            v = -negamax(pos, sq, hist, undo, tt, I, N, newdepth, -beta, -alpha,
                         ply + 1, False)
        else:
            r = int64(0)
            if N[SP_USE_LMR] and depth >= 3 and movecount > (2 if is_pv else 1) \
                    and not gives_check:
                dd = depth if depth < 63 else 63
                mc = movecount if movecount < 63 else 63
                r = int64(LMR[dd * 64 + mc])
                if not improving:
                    r += 1
                if cutnode:
                    r += 2
                if is_pv:
                    r -= 1
                if incheck:
                    r -= 1
                if not is_quiet:
                    r -= 1
                r -= hscore // N[SP_LMRHD]
                if r < 0:
                    r = 0
                if r > newdepth - 1:
                    r = newdepth - 1
                if r < 0:
                    r = 0
            rd = newdepth - r
            if rd < 1:
                rd = 1
            v = -negamax(pos, sq, hist, undo, tt, I, N, rd, -alpha - 1, -alpha,
                         ply + 1, True)
            if v > alpha and rd < newdepth:
                v = -negamax(pos, sq, hist, undo, tt, I, N, newdepth,
                             -alpha - 1, -alpha, ply + 1, not cutnode)
            if is_pv and v > alpha and v < beta:
                v = -negamax(pos, sq, hist, undo, tt, I, N, newdepth, -beta,
                             -alpha, ply + 1, False)

        unmake_move(pos, sq, undo, uoff, mv)
        if N[ON_STOP]:
            return 0

        if is_quiet and nquiet < 63:
            I[qoff + nquiet] = int32(mv)
            nquiet += 1

        if v > best:
            best = v
            if v > alpha:
                bestmove = mv
                alpha = v
                bound = BOUND_EXACT
                if is_pv:
                    p0 = OI_PV + ply * PVSTRIDE
                    p1 = OI_PV + (ply + 1) * PVSTRIDE
                    I[p0] = int32(mv)
                    ln = int64(I[OI_PVLEN + ply + 1])
                    for j in range(ln):
                        I[p0 + 1 + j] = I[p1 + j]
                    I[OI_PVLEN + ply] = int32(ln + 1)
                if v >= beta:
                    bound = BOUND_LOWER
                    break

    # ---- history / killers -------------------------------------------------
    if bound == BOUND_LOWER and bestmove != NO_MOVE:
        bonus = depth * depth * 4 + depth * 8
        if bonus > 1600:
            bonus = 1600
        bfrm = bestmove & 63
        bto = (bestmove >> 6) & 63
        bmt = (bestmove >> 14) & 3
        bvic = int64(sq[bto])
        if bvic == EMPTY and bmt != MT_PROMO and bmt != MT_EP:
            if N[SP_USE_KILL] and int64(I[OI_KILL + ply * 2]) != bestmove:
                I[OI_KILL + ply * 2 + 1] = I[OI_KILL + ply * 2]
                I[OI_KILL + ply * 2] = int32(bestmove)
            if N[SP_USE_HIST]:
                hist_bonus(I, hmax, us, bestmove, bonus)
            if N[SP_USE_CONT]:
                cont_bonus(I, hmax, ply, int64(sq[bfrm]), bestmove, bonus)
            if N[SP_USE_CM]:
                prev = int64(I[OI_SSMOVE + ply])
                if prev != NO_MOVE:
                    I[OI_COUNTER + int64(I[OI_SSPIECE + ply]) * 64 +
                      ((prev >> 6) & 63)] = int32(bestmove)
            for j in range(nquiet):
                q = int64(I[qoff + j])
                if q != bestmove:
                    if N[SP_USE_HIST]:
                        hist_bonus(I, hmax, us, q, -bonus)
                    if N[SP_USE_CONT]:
                        cont_bonus(I, hmax, ply, int64(sq[q & 63]), q, -bonus)
        else:
            cpc = int64(sq[bfrm])
            cv = 6 if bmt == MT_EP else (bvic % 6 if bvic != EMPTY else 6)
            idx = OI_CAPHIST + (cpc * 64 + bto) * 7 + cv
            h = int64(I[idx])
            h += bonus - (h * bonus) // hmax
            I[idx] = int32(h)

    if N[SP_USE_TT] and excl == NO_MOVE and not N[ON_STOP]:
        tt_store(tt, key, bestmove, score_to_tt(best, ply), depth, bound,
                 seval, N[ON_TTAGE])
    return best


# ==========================================================================
#  root
# ==========================================================================
@njit(int64(uint64[:], uint8[:], int32[:], int64[:]), cache=False, nogil=True)
def root_init(pos, sq, I, N):
    """Generate and store the root move list.  Returns the move count."""
    n = gen_moves(pos, sq, I, OI_MBUF, 0)
    for i in range(n):
        I[OI_ROOTMV + i] = I[OI_MBUF + i]
        I[OI_ROOTSC + i] = int32(-INF)
        I[OI_ROOTNODE + i] = 0
    N[ON_ROOTN] = n
    for p in range(MAX_PLY + 8):
        I[OI_SSEXCL + p] = int32(NO_MOVE)
        I[OI_SSMOVE + p] = int32(NO_MOVE)
        I[OI_SSPIECE + p] = 0
        I[OI_SSEVAL + p] = int32(VALUE_NONE)
    return n


@njit(void(int32[:], int64[:]), cache=False, nogil=True)
def root_sort(I, N):
    """Order root moves by last iteration's score, best first."""
    n = int64(N[ON_ROOTN])
    for i in range(n):
        best = i
        bs = I[OI_ROOTSC + i]
        for j in range(i + 1, n):
            if I[OI_ROOTSC + j] > bs:
                bs = I[OI_ROOTSC + j]
                best = j
        if best != i:
            t = I[OI_ROOTMV + i]; I[OI_ROOTMV + i] = I[OI_ROOTMV + best]; I[OI_ROOTMV + best] = t
            t = I[OI_ROOTSC + i]; I[OI_ROOTSC + i] = I[OI_ROOTSC + best]; I[OI_ROOTSC + best] = t
            t = I[OI_ROOTNODE + i]; I[OI_ROOTNODE + i] = I[OI_ROOTNODE + best]; I[OI_ROOTNODE + best] = t


@njit(int64(uint64[:], uint8[:], int32[:], int64[:], int64), cache=False, nogil=True)
def root_order_initial(pos, sq, I, N, ttmv):
    """First-iteration ordering: reuse the normal move-ordering heuristics."""
    n = int64(N[ON_ROOTN])
    for i in range(n):
        I[OI_MBUF + i] = I[OI_ROOTMV + i]
    score_moves(pos, sq, I, N, OI_MBUF, n, ttmv, 0)
    for i in range(n):
        pick_move(I, OI_MBUF, n, i)
        I[OI_ROOTMV + i] = I[OI_MBUF + i]
    return n


@njit(int64(uint64[:], uint8[:], uint64[:], uint64[:], uint64[:], int32[:],
            int64[:], int64, int64, int64), cache=False, nogil=True)
def search_root(pos, sq, hist, undo, tt, I, N, depth, alpha, beta):
    n = int64(N[ON_ROOTN])
    best = int64(-INF)
    bestmove = int64(NO_MOVE)
    I[OI_PVLEN] = 0
    key = pos[IKEY]

    for i in range(n):
        mv = int64(I[OI_ROOTMV + i])
        node0 = N[ON_NODES]
        frm = mv & 63
        pc = int64(sq[frm])
        I[OI_SSMOVE + 1] = int32(mv)
        I[OI_SSPIECE + 1] = int32(pc)
        make_move(pos, sq, undo, 0, mv, hist)
        gives_check = in_check(pos)
        ext = int64(1) if (N[SP_USE_CHKEXT] and gives_check and depth <= 10) else int64(0)
        nd = depth - 1 + ext

        if i == 0:
            v = -negamax(pos, sq, hist, undo, tt, I, N, nd, -beta, -alpha, 1, False)
        else:
            r = int64(0)
            if N[SP_USE_LMR] and depth >= 3 and i >= 3 and not gives_check:
                dd = depth if depth < 63 else 63
                mc = (i + 1) if (i + 1) < 63 else 63
                r = int64(LMR[dd * 64 + mc]) - 1
                if r < 0:
                    r = 0
                if r > nd - 1:
                    r = nd - 1
                if r < 0:
                    r = 0
            rd = nd - r
            if rd < 1:
                rd = 1
            v = -negamax(pos, sq, hist, undo, tt, I, N, rd, -alpha - 1, -alpha, 1, True)
            if v > alpha and rd < nd:
                v = -negamax(pos, sq, hist, undo, tt, I, N, nd, -alpha - 1, -alpha, 1, False)
            if v > alpha:
                v = -negamax(pos, sq, hist, undo, tt, I, N, nd, -beta, -alpha, 1, False)

        unmake_move(pos, sq, undo, 0, mv)
        dn = N[ON_NODES] - node0
        if dn > 2000000000:
            dn = 2000000000
        I[OI_ROOTNODE + i] = int32(dn)

        if N[ON_STOP]:
            break

        if i == 0 or v > alpha:
            I[OI_ROOTSC + i] = int32(v)
        else:
            I[OI_ROOTSC + i] = int32(-INF)

        if v > best:
            best = v
            bestmove = mv
            if v > alpha:
                alpha = v
                p0 = OI_PV
                p1 = OI_PV + PVSTRIDE
                I[p0] = int32(mv)
                ln = int64(I[OI_PVLEN + 1])
                for j in range(ln):
                    I[p0 + 1 + j] = I[p1 + j]
                I[OI_PVLEN] = int32(ln + 1)
                if v >= beta:
                    break

    if bestmove != NO_MOVE and not N[ON_STOP]:
        N[ON_BESTMV] = bestmove
        N[ON_BESTSC] = best
        bnd = BOUND_LOWER if best >= beta else (
            BOUND_UPPER if best <= alpha and best < beta else BOUND_EXACT)
        tt_store(tt, key, bestmove, score_to_tt(best, 0), depth,
                 BOUND_EXACT if bnd == BOUND_EXACT else bnd,
                 VALUE_NONE, N[ON_TTAGE])
    return best


@njit(void(int32[:]), cache=False, nogil=True)
def age_history(I):
    """Halve history between moves: keep the shape, lose the stale magnitude."""
    for i in range(2 * 64 * 64):
        I[OI_HISTORY + i] = I[OI_HISTORY + i] // 2
    for i in range(12 * 64 * 7):
        I[OI_CAPHIST + i] = I[OI_CAPHIST + i] // 2
    for i in range(12 * 64 * 12 * 64):
        I[OI_CONTHIST + i] = I[OI_CONTHIST + i] // 2


@njit(void(int32[:]), cache=False, nogil=True)
def clear_heuristics(I):
    for i in range(MAX_PLY * 2):
        I[OI_KILL + i] = 0
    for i in range(2 * 64 * 64):
        I[OI_HISTORY + i] = 0
    for i in range(12 * 64 * 7):
        I[OI_CAPHIST + i] = 0
    for i in range(12 * 64):
        I[OI_COUNTER + i] = 0
    for i in range(12 * 64 * 12 * 64):
        I[OI_CONTHIST + i] = 0
