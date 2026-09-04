"""Bitboard position: representation, make/unmake, fully-legal move generation.

The position is three flat arrays so it can be handed to njit code with no
boxing:

  pos : uint64[POS_N]  bitboards + occupancies + scalar state (see index consts)
  sq  : uint8[64]      mailbox, piece code 0..11, 12 = empty
  hist: uint64[]       zobrist key history, for repetition detection

Move generation produces *only legal moves*.  Generating pseudo-legal moves and
filtering with make/unmake is simpler but measurably slower, and legal-only
generation removes a whole class of "engine played an illegal move" failures,
which are an instant loss in this competition.
"""
import numpy as np
from numba import njit, uint64, uint8, int64, int32, boolean, void

from .tables import (
    U, ONE, FULL, MASK64,
    WP, WN, WB, WR, WQ, WK, BP, BN, BB_, BR, BQ, BK, EMPTY,
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK,
    KNIGHT_ATT, KING_ATT, PAWN_ATT, BETWEEN, LINE,
    FILE_BB, RANK_BB, ZOB_PIECE, ZOB_CAST, ZOB_EP, ZOB_SIDE,
    lsb, popcount, rook_attacks, bishop_attacks, queen_attacks,
)

# ---------------------------------------------------------------- pos layout
IOCC_W, IOCC_B, IOCC_A = 12, 13, 14
ISIDE, ICAST, IEP, IHM, IKEY, IPKEY, IPHASE, IHIDX, IPLY = range(15, 24)
POS_N = 24

NO_EP = 64
MAX_PLY = 128
MAX_MOVES = 256
MAX_HIST = 1024

# castling right bits
CW_K, CW_Q, CB_K, CB_Q = 1, 2, 4, 8

# ---------------------------------------------------------------- move layout
# bits 0-5 from | 6-11 to | 12-13 promo(0=N,1=B,2=R,3=Q) | 14-15 type
MT_NORMAL, MT_PROMO, MT_EP, MT_CASTLE = 0, 1, 2, 3
NO_MOVE = 0

def mk_move(frm, to, mtype=MT_NORMAL, promo=0):
    return np.int32(frm | (to << 6) | (promo << 12) | (mtype << 14))

# game-phase weight per piece type
PHASE_W = np.array([0, 1, 1, 2, 4, 0], dtype=np.int64)
PHASE_TOTAL = 24

# squares whose occupancy change clears castling rights
_cm = np.full(64, 15, dtype=np.int64)
_cm[0] &= ~CW_Q; _cm[7] &= ~CW_K; _cm[4] &= ~(CW_K | CW_Q)
_cm[56] &= ~CB_Q; _cm[63] &= ~CB_K; _cm[60] &= ~(CB_K | CB_Q)
CASTLE_MASK = _cm

# pseudo attacks on an empty board, for sniper detection
PSEUDO_ROOK = np.zeros(64, dtype=np.uint64)
PSEUDO_BISHOP = np.zeros(64, dtype=np.uint64)
for _s in range(64):
    PSEUDO_ROOK[_s] = rook_attacks(_s, U(0))
    PSEUDO_BISHOP[_s] = bishop_attacks(_s, U(0))

# undo record fields
UN_CAST, UN_EP, UN_HM, UN_KEY, UN_CAP, UN_PKEY, UN_PHASE, UN_N = 0, 1, 2, 3, 4, 5, 6, 8


# ==========================================================================
#  attack queries
# ==========================================================================
@njit(uint64(uint64[:], int64, uint64), cache=False, nogil=True)
def attackers_to(pos, s, occ):
    """Every piece of either colour attacking square `s` under occupancy `occ`."""
    return ((PAWN_ATT[BLACK][s] & pos[WP]) |
            (PAWN_ATT[WHITE][s] & pos[BP]) |
            (KNIGHT_ATT[s] & (pos[WN] | pos[BN])) |
            (KING_ATT[s] & (pos[WK] | pos[BK])) |
            (rook_attacks(s, occ) & (pos[WR] | pos[BR] | pos[WQ] | pos[BQ])) |
            (bishop_attacks(s, occ) & (pos[WB] | pos[BB_] | pos[WQ] | pos[BQ])))


@njit(boolean(uint64[:], int64, int64, uint64), cache=False, nogil=True)
def attacked_by(pos, s, by, occ):
    """Is square `s` attacked by colour `by` under occupancy `occ`?"""
    o = 6 * by
    if PAWN_ATT[1 - by][s] & pos[WP + o]:
        return True
    if KNIGHT_ATT[s] & pos[WN + o]:
        return True
    if KING_ATT[s] & pos[WK + o]:
        return True
    if rook_attacks(s, occ) & (pos[WR + o] | pos[WQ + o]):
        return True
    if bishop_attacks(s, occ) & (pos[WB + o] | pos[WQ + o]):
        return True
    return False


@njit(uint64(uint64[:], int64, int64), cache=False, nogil=True)
def checkers_of(pos, us, ksq):
    them = 1 - us
    o = 6 * them
    occ = pos[IOCC_A]
    return ((PAWN_ATT[us][ksq] & pos[WP + o]) |
            (KNIGHT_ATT[ksq] & pos[WN + o]) |
            (rook_attacks(ksq, occ) & (pos[WR + o] | pos[WQ + o])) |
            (bishop_attacks(ksq, occ) & (pos[WB + o] | pos[WQ + o])))


@njit(uint64(uint64[:], int64, int64), cache=False, nogil=True)
def pinned_of(pos, us, ksq):
    """Our pieces that are absolutely pinned against our king."""
    them = 1 - us
    o = 6 * them
    ours = pos[IOCC_W + us]
    occ = pos[IOCC_A]
    snipers = ((PSEUDO_ROOK[ksq] & (pos[WR + o] | pos[WQ + o])) |
               (PSEUDO_BISHOP[ksq] & (pos[WB + o] | pos[WQ + o])))
    pinned = U(0)
    while snipers:
        s = lsb(snipers)
        snipers &= snipers - ONE
        b = BETWEEN[ksq][s] & occ
        if b and (b & (b - ONE)) == U(0):      # exactly one blocker
            pinned |= b & ours
    return pinned


# ==========================================================================
#  move generation  (legal only)
# ==========================================================================
@njit(int64(uint64[:], uint8[:], int32[:], int64, int64), cache=False, nogil=True)
def gen_moves(pos, sq, buf, off, mode):
    """mode 0 = all legal moves, 1 = captures + queen promotions only.

    Returns the number of moves written to buf[off:].
    """
    n = int64(0)
    us = int64(pos[ISIDE])
    them = 1 - us
    o = 6 * us
    ours = pos[IOCC_W + us]
    theirs = pos[IOCC_W + them]
    occ = pos[IOCC_A]
    ksq = lsb(pos[WK + o])
    checkers = checkers_of(pos, us, ksq)
    pinned = pinned_of(pos, us, ksq)
    nchk = popcount(checkers)

    # ---- king moves (always generated) ------------------------------------
    occ_nk = occ ^ (ONE << U(ksq))
    katt = KING_ATT[ksq] & ~ours
    if mode == 1:
        katt &= theirs
    while katt:
        t = lsb(katt)
        katt &= katt - ONE
        if not attacked_by(pos, t, them, occ_nk):
            buf[off + n] = int32(ksq | (t << 6))
            n += 1

    if nchk > 1:
        return n                                  # double check: king only

    # ---- target squares ----------------------------------------------------
    if nchk == 1:
        csq = lsb(checkers)
        target_all = BETWEEN[ksq][csq] | checkers   # block or capture
    else:
        target_all = ~ours
    # promotions are searched even in "captures only" mode: they are tactical
    if mode == 1:
        target = target_all & (theirs | checkers)
    else:
        target = target_all

    # ---- pawns -------------------------------------------------------------
    pawns = pos[WP + o]
    if us == WHITE:
        push1_sh = 8; rank3 = RANK_BB[2]; promo_rank = RANK_BB[7]
    else:
        push1_sh = -8; rank3 = RANK_BB[5]; promo_rank = RANK_BB[0]
    empty = ~occ

    p = pawns
    while p:
        f = lsb(p)
        p &= p - ONE
        legal_mask = FULL
        if (ONE << U(f)) & pinned:
            legal_mask = LINE[ksq][f]

        # captures
        caps = PAWN_ATT[us][f] & theirs & target & legal_mask
        while caps:
            t = lsb(caps)
            caps &= caps - ONE
            if (ONE << U(t)) & promo_rank:
                buf[off + n] = int32(f | (t << 6) | (3 << 12) | (MT_PROMO << 14)); n += 1
                if mode == 0:
                    buf[off + n] = int32(f | (t << 6) | (2 << 12) | (MT_PROMO << 14)); n += 1
                    buf[off + n] = int32(f | (t << 6) | (1 << 12) | (MT_PROMO << 14)); n += 1
                    buf[off + n] = int32(f | (t << 6) | (0 << 12) | (MT_PROMO << 14)); n += 1
            else:
                buf[off + n] = int32(f | (t << 6)); n += 1

        # quiet pushes
        t1 = f + push1_sh
        if 0 <= t1 < 64 and (empty & (ONE << U(t1))):
            if (ONE << U(t1)) & promo_rank:
                if (ONE << U(t1)) & target_all & legal_mask:
                    buf[off + n] = int32(f | (t1 << 6) | (3 << 12) | (MT_PROMO << 14)); n += 1
                    if mode == 0:
                        buf[off + n] = int32(f | (t1 << 6) | (2 << 12) | (MT_PROMO << 14)); n += 1
                        buf[off + n] = int32(f | (t1 << 6) | (1 << 12) | (MT_PROMO << 14)); n += 1
                        buf[off + n] = int32(f | (t1 << 6) | (0 << 12) | (MT_PROMO << 14)); n += 1
            elif mode == 0:
                if (ONE << U(t1)) & target_all & legal_mask:
                    buf[off + n] = int32(f | (t1 << 6)); n += 1
                t2 = t1 + push1_sh
                if ((ONE << U(t1)) & rank3) and (empty & (ONE << U(t2))) \
                        and ((ONE << U(t2)) & target_all & legal_mask):
                    buf[off + n] = int32(f | (t2 << 6)); n += 1

    # ---- en passant (rare; full legality check) ----------------------------
    ep = int64(pos[IEP])
    if ep != NO_EP:
        capt_sq = ep - push1_sh
        src = PAWN_ATT[them][ep] & pos[WP + o]
        while src:
            f = lsb(src)
            src &= src - ONE
            occ2 = (occ ^ (ONE << U(f)) ^ (ONE << U(capt_sq))) | (ONE << U(ep))
            eo = 6 * them
            if not ((rook_attacks(ksq, occ2) & (pos[WR + eo] | pos[WQ + eo])) or
                    (bishop_attacks(ksq, occ2) & (pos[WB + eo] | pos[WQ + eo]))):
                # also confirm it actually resolves an existing check
                if nchk == 0 or (checkers & (ONE << U(capt_sq))) or \
                        ((ONE << U(ep)) & BETWEEN[ksq][lsb(checkers)]):
                    buf[off + n] = int32(f | (ep << 6) | (MT_EP << 14)); n += 1

    # ---- knights -----------------------------------------------------------
    b = pos[WN + o] & ~pinned                     # a pinned knight can never move
    while b:
        f = lsb(b)
        b &= b - ONE
        a = KNIGHT_ATT[f] & target
        while a:
            t = lsb(a)
            a &= a - ONE
            buf[off + n] = int32(f | (t << 6)); n += 1

    # ---- bishops / queens (diagonal) --------------------------------------
    b = pos[WB + o] | pos[WQ + o]
    while b:
        f = lsb(b)
        b &= b - ONE
        a = bishop_attacks(f, occ) & target
        if (ONE << U(f)) & pinned:
            a &= LINE[ksq][f]
        while a:
            t = lsb(a)
            a &= a - ONE
            buf[off + n] = int32(f | (t << 6)); n += 1

    # ---- rooks / queens (orthogonal) --------------------------------------
    b = pos[WR + o] | pos[WQ + o]
    while b:
        f = lsb(b)
        b &= b - ONE
        a = rook_attacks(f, occ) & target
        if (ONE << U(f)) & pinned:
            a &= LINE[ksq][f]
        while a:
            t = lsb(a)
            a &= a - ONE
            buf[off + n] = int32(f | (t << 6)); n += 1

    # ---- castling ----------------------------------------------------------
    if mode == 0 and nchk == 0:
        cr = int64(pos[ICAST])
        if us == WHITE:
            if (cr & CW_K) and not (occ & U(0x60)) \
                    and not attacked_by(pos, 5, BLACK, occ) \
                    and not attacked_by(pos, 6, BLACK, occ):
                buf[off + n] = int32(4 | (6 << 6) | (MT_CASTLE << 14)); n += 1
            if (cr & CW_Q) and not (occ & U(0x0E)) \
                    and not attacked_by(pos, 3, BLACK, occ) \
                    and not attacked_by(pos, 2, BLACK, occ):
                buf[off + n] = int32(4 | (2 << 6) | (MT_CASTLE << 14)); n += 1
        else:
            if (cr & CB_K) and not (occ & U(0x6000000000000000)) \
                    and not attacked_by(pos, 61, WHITE, occ) \
                    and not attacked_by(pos, 62, WHITE, occ):
                buf[off + n] = int32(60 | (62 << 6) | (MT_CASTLE << 14)); n += 1
            if (cr & CB_Q) and not (occ & U(0x0E00000000000000)) \
                    and not attacked_by(pos, 59, WHITE, occ) \
                    and not attacked_by(pos, 58, WHITE, occ):
                buf[off + n] = int32(60 | (58 << 6) | (MT_CASTLE << 14)); n += 1
    return n


# ==========================================================================
#  make / unmake
# ==========================================================================
@njit(void(uint64[:], uint8[:], uint64[:], int64, int64, uint64[:]),
      cache=False, nogil=True)
def make_move(pos, sq, undo, uoff, mv, hist):
    frm = mv & 63
    to = (mv >> 6) & 63
    mt = (mv >> 14) & 3
    us = int64(pos[ISIDE])
    them = 1 - us
    o = 6 * us
    pc = int64(sq[frm])
    cap = int64(sq[to])
    key = pos[IKEY]
    pkey = pos[IPKEY]

    undo[uoff + UN_CAST] = pos[ICAST]
    undo[uoff + UN_EP] = pos[IEP]
    undo[uoff + UN_HM] = pos[IHM]
    undo[uoff + UN_KEY] = key
    undo[uoff + UN_PKEY] = pkey
    undo[uoff + UN_PHASE] = pos[IPHASE]

    # clear old ep from key
    old_ep = int64(pos[IEP])
    if old_ep != NO_EP:
        key ^= ZOB_EP[old_ep & 7]

    hm = int64(pos[IHM]) + 1
    new_ep = NO_EP

    if mt == MT_EP:
        capsq = to - 8 if us == WHITE else to + 8
        cappc = BP if us == WHITE else WP
        pos[cappc] ^= ONE << U(capsq)
        pos[IOCC_W + them] ^= ONE << U(capsq)
        sq[capsq] = EMPTY
        key ^= ZOB_PIECE[cappc][capsq]
        pkey ^= ZOB_PIECE[cappc][capsq]
        undo[uoff + UN_CAP] = EMPTY
        hm = 0
    else:
        undo[uoff + UN_CAP] = cap
        if cap != EMPTY:
            pos[cap] ^= ONE << U(to)
            pos[IOCC_W + them] ^= ONE << U(to)
            key ^= ZOB_PIECE[cap][to]
            if cap == WP or cap == BP:
                pkey ^= ZOB_PIECE[cap][to]
            else:
                pos[IPHASE] -= U(PHASE_W[cap % 6])
            hm = 0

    # move the piece
    pos[pc] ^= (ONE << U(frm)) | (ONE << U(to))
    pos[IOCC_W + us] ^= (ONE << U(frm)) | (ONE << U(to))
    sq[frm] = EMPTY
    sq[to] = pc
    key ^= ZOB_PIECE[pc][frm] ^ ZOB_PIECE[pc][to]

    if pc == WP or pc == BP:
        pkey ^= ZOB_PIECE[pc][frm] ^ ZOB_PIECE[pc][to]
        hm = 0
        if mt == MT_PROMO:
            newpc = (KNIGHT + ((mv >> 12) & 3)) + o
            pos[pc] ^= ONE << U(to)
            pos[newpc] ^= ONE << U(to)
            sq[to] = newpc
            key ^= ZOB_PIECE[pc][to] ^ ZOB_PIECE[newpc][to]
            pkey ^= ZOB_PIECE[pc][to]
            pos[IPHASE] += U(PHASE_W[newpc % 6])
        elif (to - frm) == 16 or (to - frm) == -16:
            mid = (to + frm) >> 1
            # only set ep if an enemy pawn can actually take (keeps keys tight)
            if PAWN_ATT[us][mid] & pos[WP + 6 * them]:
                new_ep = mid
    elif mt == MT_CASTLE:
        if to == 6:    rf, rt = 7, 5
        elif to == 2:  rf, rt = 0, 3
        elif to == 62: rf, rt = 63, 61
        else:          rf, rt = 56, 59
        rpc = WR + o
        pos[rpc] ^= (ONE << U(rf)) | (ONE << U(rt))
        pos[IOCC_W + us] ^= (ONE << U(rf)) | (ONE << U(rt))
        sq[rf] = EMPTY
        sq[rt] = rpc
        key ^= ZOB_PIECE[rpc][rf] ^ ZOB_PIECE[rpc][rt]

    # castling rights
    cr = int64(pos[ICAST])
    ncr = cr & CASTLE_MASK[frm] & CASTLE_MASK[to]
    if ncr != cr:
        key ^= ZOB_CAST[cr] ^ ZOB_CAST[ncr]
        pos[ICAST] = U(ncr)

    if new_ep != NO_EP:
        key ^= ZOB_EP[new_ep & 7]
    pos[IEP] = U(new_ep)
    pos[IHM] = U(hm)
    key ^= ZOB_SIDE
    pos[ISIDE] = U(them)
    pos[IKEY] = key
    pos[IPKEY] = pkey
    pos[IOCC_A] = pos[IOCC_W] | pos[IOCC_B]
    hidx = int64(pos[IHIDX]) + 1
    pos[IHIDX] = U(hidx)
    hist[hidx] = key
    pos[IPLY] += ONE


@njit(void(uint64[:], uint8[:], uint64[:], int64, int64), cache=False, nogil=True)
def unmake_move(pos, sq, undo, uoff, mv):
    frm = mv & 63
    to = (mv >> 6) & 63
    mt = (mv >> 14) & 3
    them = int64(pos[ISIDE])
    us = 1 - them
    o = 6 * us
    pc = int64(sq[to])

    if mt == MT_PROMO:
        pos[pc] ^= ONE << U(to)
        pc = WP + o
        pos[pc] ^= ONE << U(to)
        pos[IPHASE] = undo[uoff + UN_PHASE]

    pos[pc] ^= (ONE << U(frm)) | (ONE << U(to))
    pos[IOCC_W + us] ^= (ONE << U(frm)) | (ONE << U(to))
    sq[frm] = pc
    sq[to] = EMPTY

    if mt == MT_EP:
        capsq = to - 8 if us == WHITE else to + 8
        cappc = BP if us == WHITE else WP
        pos[cappc] ^= ONE << U(capsq)
        pos[IOCC_W + them] ^= ONE << U(capsq)
        sq[capsq] = cappc
    else:
        cap = int64(undo[uoff + UN_CAP])
        if cap != EMPTY:
            pos[cap] ^= ONE << U(to)
            pos[IOCC_W + them] ^= ONE << U(to)
            sq[to] = cap
            pos[IPHASE] = undo[uoff + UN_PHASE]

    if mt == MT_CASTLE:
        if to == 6:    rf, rt = 7, 5
        elif to == 2:  rf, rt = 0, 3
        elif to == 62: rf, rt = 63, 61
        else:          rf, rt = 56, 59
        rpc = WR + o
        pos[rpc] ^= (ONE << U(rf)) | (ONE << U(rt))
        pos[IOCC_W + us] ^= (ONE << U(rf)) | (ONE << U(rt))
        sq[rf] = rpc
        sq[rt] = EMPTY

    pos[ICAST] = undo[uoff + UN_CAST]
    pos[IEP] = undo[uoff + UN_EP]
    pos[IHM] = undo[uoff + UN_HM]
    pos[IKEY] = undo[uoff + UN_KEY]
    pos[IPKEY] = undo[uoff + UN_PKEY]
    pos[ISIDE] = U(us)
    pos[IOCC_A] = pos[IOCC_W] | pos[IOCC_B]
    pos[IHIDX] -= ONE
    pos[IPLY] -= ONE


@njit(void(uint64[:], uint64[:], int64, uint64[:]), cache=False, nogil=True)
def make_null(pos, undo, uoff, hist):
    undo[uoff + UN_EP] = pos[IEP]
    undo[uoff + UN_KEY] = pos[IKEY]
    undo[uoff + UN_HM] = pos[IHM]
    key = pos[IKEY]
    ep = int64(pos[IEP])
    if ep != NO_EP:
        key ^= ZOB_EP[ep & 7]
    pos[IEP] = U(NO_EP)
    key ^= ZOB_SIDE
    pos[IKEY] = key
    pos[ISIDE] = U(1 - int64(pos[ISIDE]))
    # a null move is an irreversible boundary for repetition scanning
    pos[IHM] = U(0)
    hidx = int64(pos[IHIDX]) + 1
    pos[IHIDX] = U(hidx)
    hist[hidx] = key
    pos[IPLY] += ONE


@njit(void(uint64[:], uint64[:], int64), cache=False, nogil=True)
def unmake_null(pos, undo, uoff):
    pos[IEP] = undo[uoff + UN_EP]
    pos[IKEY] = undo[uoff + UN_KEY]
    pos[IHM] = undo[uoff + UN_HM]
    pos[ISIDE] = U(1 - int64(pos[ISIDE]))
    pos[IHIDX] -= ONE
    pos[IPLY] -= ONE


@njit(boolean(uint64[:]), cache=False, nogil=True)
def in_check(pos):
    us = int64(pos[ISIDE])
    return checkers_of(pos, us, lsb(pos[WK + 6 * us])) != U(0)


# ==========================================================================
#  python-side construction helpers (cold path only)
# ==========================================================================
_FEN_PIECE = {'P': WP, 'N': WN, 'B': WB, 'R': WR, 'Q': WQ, 'K': WK,
              'p': BP, 'n': BN, 'b': BB_, 'r': BR, 'q': BQ, 'k': BK}
PIECE_CHAR = "PNBRQKpnbrqk."


def new_state():
    return (np.zeros(POS_N, dtype=np.uint64),
            np.full(64, EMPTY, dtype=np.uint8),
            np.zeros(MAX_HIST, dtype=np.uint64))


def set_fen(pos, sq, hist, fen):
    """Populate pos/sq/hist from a FEN.  Cold path — clarity over speed."""
    pos[:] = 0
    sq[:] = EMPTY
    parts = fen.split()
    board, stm = parts[0], parts[1]
    castle = parts[2] if len(parts) > 2 else '-'
    epf = parts[3] if len(parts) > 3 else '-'
    hm = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0

    r, f = 7, 0
    for ch in board:
        if ch == '/':
            r -= 1; f = 0
        elif ch.isdigit():
            f += int(ch)
        else:
            s = r * 8 + f
            pc = _FEN_PIECE[ch]
            sq[s] = pc
            pos[pc] |= U(1) << U(s)
            f += 1

    for c in range(2):
        v = U(0)
        for p in range(6):
            v |= pos[p + 6 * c]
        pos[IOCC_W + c] = v
    pos[IOCC_A] = pos[IOCC_W] | pos[IOCC_B]
    pos[ISIDE] = U(WHITE if stm == 'w' else BLACK)

    cr = 0
    if 'K' in castle: cr |= CW_K
    if 'Q' in castle: cr |= CW_Q
    if 'k' in castle: cr |= CB_K
    if 'q' in castle: cr |= CB_Q
    pos[ICAST] = U(cr)

    ep = NO_EP
    if epf != '-' and len(epf) == 2:
        cand = (ord(epf[0]) - 97) + (int(epf[1]) - 1) * 8
        us = int(pos[ISIDE])
        if PAWN_ATT[1 - us][cand] & pos[WP + 6 * us]:
            ep = cand
    pos[IEP] = U(ep)
    pos[IHM] = U(hm)

    ph = 0
    for p in (KNIGHT, BISHOP, ROOK, QUEEN):
        ph += int(PHASE_W[p]) * (int(popcount(pos[p])) + int(popcount(pos[p + 6])))
    pos[IPHASE] = U(min(ph, PHASE_TOTAL))

    pos[IKEY] = _compute_key(pos, sq)
    pos[IPKEY] = _compute_pawn_key(pos, sq)
    pos[IHIDX] = U(0)
    pos[IPLY] = U(0)
    hist[0] = pos[IKEY]
    return pos


def _compute_key(pos, sq):
    k = U(0)
    for s in range(64):
        p = int(sq[s])
        if p != EMPTY:
            k ^= ZOB_PIECE[p][s]
    k ^= ZOB_CAST[int(pos[ICAST])]
    if int(pos[IEP]) != NO_EP:
        k ^= ZOB_EP[int(pos[IEP]) & 7]
    if int(pos[ISIDE]) == BLACK:
        k ^= ZOB_SIDE
    return k


def _compute_pawn_key(pos, sq):
    k = U(0)
    for s in range(64):
        p = int(sq[s])
        if p == WP or p == BP:
            k ^= ZOB_PIECE[p][s]
    return k


def move_to_uci(mv):
    mv = int(mv)
    frm, to = mv & 63, (mv >> 6) & 63
    s = chr(97 + (frm & 7)) + str((frm >> 3) + 1) + chr(97 + (to & 7)) + str((to >> 3) + 1)
    if ((mv >> 14) & 3) == MT_PROMO:
        s += "nbrq"[(mv >> 12) & 3]
    return s


def uci_to_move(pos, sq, hist, uci):
    """Match a UCI string against the legal move list (cold path)."""
    buf = np.zeros(MAX_MOVES, dtype=np.int32)
    n = gen_moves(pos, sq, buf, 0, 0)
    for i in range(n):
        if move_to_uci(buf[i]) == uci:
            return int(buf[i])
    return NO_MOVE
