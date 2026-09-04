"""Precomputed attack / geometry / zobrist tables.

Everything here is built once at import time and then referenced as a frozen
global constant from the njit hot path.  Build cost is milliseconds.

Bitboard convention: bit i == square i, a1=0, b1=1, ..., h8=63.
File = sq & 7, rank = sq >> 3.  Bitboards are uint64 throughout; every shift
amount is explicitly cast to uint64 because numba unifies uint64 with int64 as
float64 (the single most common source of silent corruption in numba bitboards).
"""
import numpy as np
from numba import njit, uint64, int64, int32, void

from .magics import (ROOK_MAGIC, ROOK_SHIFT, BISHOP_MAGIC, BISHOP_SHIFT,
                     ROOK_TABLE_SIZE, BISHOP_TABLE_SIZE)

U = np.uint64
ONE = U(1)
FULL = U(0xFFFFFFFFFFFFFFFF)

# --------------------------------------------------------------------------
# piece codes
# --------------------------------------------------------------------------
WP, WN, WB, WR, WQ, WK, BP, BN, BB_, BR, BQ, BK, EMPTY = range(13)
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)
WHITE, BLACK = 0, 1

# --------------------------------------------------------------------------
# geometry helpers (plain python, import-time only)
# --------------------------------------------------------------------------
MASK64 = (1 << 64) - 1

def _sq(f, r):
    return r * 8 + f

def _slide(s, occ, deltas):
    att = 0
    f0, r0 = s & 7, s >> 3
    for df, dr in deltas:
        f, r = f0 + df, r0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            b = 1 << _sq(f, r)
            att |= b
            if occ & b:
                break
            f += df; r += dr
    return att

def _mask(s, deltas):
    m = 0
    f0, r0 = s & 7, s >> 3
    for df, dr in deltas:
        f, r = f0 + df, r0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            nf, nr = f + df, r + dr
            if not (0 <= nf < 8 and 0 <= nr < 8):
                break
            m |= 1 << _sq(f, r)
            f, r = nf, nr
    return m

R_DELTAS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
B_DELTAS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

def _subsets(m):
    out, s = [], 0
    while True:
        out.append(s)
        s = (s - m) & m
        if s == 0:
            return out

# --------------------------------------------------------------------------
# leaper attacks
# --------------------------------------------------------------------------
KNIGHT_ATT = np.zeros(64, dtype=np.uint64)
KING_ATT   = np.zeros(64, dtype=np.uint64)
PAWN_ATT   = np.zeros((2, 64), dtype=np.uint64)     # [colour][from]

for s in range(64):
    f, r = s & 7, s >> 3
    n = 0
    for df, dr in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)):
        nf, nr = f + df, r + dr
        if 0 <= nf < 8 and 0 <= nr < 8:
            n |= 1 << _sq(nf, nr)
    KNIGHT_ATT[s] = U(n)
    k = 0
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == dr == 0:
                continue
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                k |= 1 << _sq(nf, nr)
    KING_ATT[s] = U(k)
    for c, dr in ((WHITE, 1), (BLACK, -1)):
        p = 0
        for df in (-1, 1):
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                p |= 1 << _sq(nf, nr)
        PAWN_ATT[c][s] = U(p)

# --------------------------------------------------------------------------
# magic slider tables
# --------------------------------------------------------------------------
ROOK_MASK   = np.array([_mask(s, R_DELTAS) for s in range(64)], dtype=np.uint64)
BISHOP_MASK = np.array([_mask(s, B_DELTAS) for s in range(64)], dtype=np.uint64)
R_MAGIC     = np.array(ROOK_MAGIC,   dtype=np.uint64)
B_MAGIC     = np.array(BISHOP_MAGIC, dtype=np.uint64)
R_SHIFT     = np.array(ROOK_SHIFT,   dtype=np.uint64)
B_SHIFT     = np.array(BISHOP_SHIFT, dtype=np.uint64)

R_OFFSET = np.zeros(64, dtype=np.int64)
B_OFFSET = np.zeros(64, dtype=np.int64)
_o = 0
for s in range(64):
    R_OFFSET[s] = _o
    _o += 1 << (64 - ROOK_SHIFT[s])
_o = 0
for s in range(64):
    B_OFFSET[s] = _o
    _o += 1 << (64 - BISHOP_SHIFT[s])

ROOK_TABLE   = np.zeros(ROOK_TABLE_SIZE,   dtype=np.uint64)
BISHOP_TABLE = np.zeros(BISHOP_TABLE_SIZE, dtype=np.uint64)

for s in range(64):
    for deltas, msk, mag, sh, off, tbl in (
            (R_DELTAS, ROOK_MASK, R_MAGIC, R_SHIFT, R_OFFSET, ROOK_TABLE),
            (B_DELTAS, BISHOP_MASK, B_MAGIC, B_SHIFT, B_OFFSET, BISHOP_TABLE)):
        m = int(msk[s])
        for occ in _subsets(m):
            idx = ((occ * int(mag[s])) & MASK64) >> int(sh[s])
            tbl[off[s] + idx] = U(_slide(s, occ, deltas))

# --------------------------------------------------------------------------
# between / line (pin + check-block geometry)
# --------------------------------------------------------------------------
BETWEEN = np.zeros((64, 64), dtype=np.uint64)   # exclusive, 0 if not aligned
LINE    = np.zeros((64, 64), dtype=np.uint64)   # whole line through a,b incl. both

for a in range(64):
    for b in range(64):
        if a == b:
            continue
        for deltas in (R_DELTAS, B_DELTAS):
            if _slide(a, 0, deltas) & (1 << b):
                BETWEEN[a][b] = U(_slide(a, 1 << b, deltas) & _slide(b, 1 << a, deltas))
                LINE[a][b] = U((_slide(a, 0, deltas) & _slide(b, 0, deltas))
                               | (1 << a) | (1 << b))
                break

# --------------------------------------------------------------------------
# file / rank / adjacency helpers (used by eval and pawn logic)
# --------------------------------------------------------------------------
FILE_BB = np.array([U(0x0101010101010101 << f) for f in range(8)], dtype=np.uint64)
RANK_BB = np.array([U(0xFF << (8 * r)) for r in range(8)], dtype=np.uint64)
ADJ_FILES = np.zeros(8, dtype=np.uint64)
for f in range(8):
    v = 0
    if f > 0: v |= int(FILE_BB[f - 1])
    if f < 7: v |= int(FILE_BB[f + 1])
    ADJ_FILES[f] = U(v)

# forward span of a square for a colour (all squares strictly ahead, same file)
FORWARD = np.zeros((2, 64), dtype=np.uint64)
# passed-pawn mask: forward span on own + adjacent files
PASSED = np.zeros((2, 64), dtype=np.uint64)
# squares behind a pawn on adjacent files (for backward-pawn detection)
NEIGHBOURS = np.zeros((2, 64), dtype=np.uint64)
for c in (WHITE, BLACK):
    for s in range(64):
        f, r = s & 7, s >> 3
        fwd = 0
        rr = range(r + 1, 8) if c == WHITE else range(0, r)
        for r2 in rr:
            fwd |= int(RANK_BB[r2])
        FORWARD[c][s] = U(fwd & int(FILE_BB[f]))
        PASSED[c][s]  = U(fwd & (int(FILE_BB[f]) | int(ADJ_FILES[f])))
        back = 0
        rr2 = range(0, r + 1) if c == WHITE else range(r, 8)
        for r2 in rr2:
            back |= int(RANK_BB[r2])
        NEIGHBOURS[c][s] = U(back & int(ADJ_FILES[f]))

# distance tables
SQ_DIST = np.zeros((64, 64), dtype=np.int32)      # Chebyshev
MAN_DIST = np.zeros((64, 64), dtype=np.int32)     # Manhattan
CENTRE_MAN_DIST = np.zeros(64, dtype=np.int32)
for a in range(64):
    for b in range(64):
        df, dr = abs((a & 7) - (b & 7)), abs((a >> 3) - (b >> 3))
        SQ_DIST[a][b] = max(df, dr)
        MAN_DIST[a][b] = df + dr
    CENTRE_MAN_DIST[a] = min(MAN_DIST[a][27], MAN_DIST[a][28],
                             MAN_DIST[a][35], MAN_DIST[a][36])

# king ring / king zone for safety eval
KING_RING = np.zeros(64, dtype=np.uint64)
for s in range(64):
    f = min(max(s & 7, 1), 6)
    r = min(max(s >> 3, 1), 6)
    c = _sq(f, r)
    KING_RING[s] = U(int(KING_ATT[c]) | (1 << c))

# --------------------------------------------------------------------------
# zobrist
# --------------------------------------------------------------------------
_rng = np.random.Generator(np.random.PCG64(0x9E3779B97F4A7C15))
ZOB_PIECE = _rng.integers(0, 1 << 64, size=(12, 64), dtype=np.uint64)
ZOB_CAST  = _rng.integers(0, 1 << 64, size=16, dtype=np.uint64)
ZOB_EP    = _rng.integers(0, 1 << 64, size=8, dtype=np.uint64)
ZOB_SIDE  = U(_rng.integers(0, 1 << 64, dtype=np.uint64))

# --------------------------------------------------------------------------
# njit primitives
# --------------------------------------------------------------------------
_DEBRUIJN = U(0x03f79d71b4cb0a89)
_DBIDX = np.zeros(64, dtype=np.int64)
for i in range(64):
    _DBIDX[(((1 << i) * int(_DEBRUIJN)) & MASK64) >> 58] = i


@njit(int64(uint64), inline='always', cache=False, nogil=True)
def lsb(b):
    """Index of least significant set bit.  b must be non-zero."""
    return _DBIDX[(((b & (~b + U(1)))) * _DEBRUIJN) >> U(58)]


@njit(int64(uint64), inline='always', cache=False, nogil=True)
def popcount(b):
    c = int64(0)
    while b:
        b &= b - U(1)
        c += 1
    return c


@njit(uint64(int64, uint64), inline='always', cache=False, nogil=True)
def rook_attacks(s, occ):
    return ROOK_TABLE[R_OFFSET[s] + int64(((occ & ROOK_MASK[s]) * R_MAGIC[s]) >> R_SHIFT[s])]


@njit(uint64(int64, uint64), inline='always', cache=False, nogil=True)
def bishop_attacks(s, occ):
    return BISHOP_TABLE[B_OFFSET[s] + int64(((occ & BISHOP_MASK[s]) * B_MAGIC[s]) >> B_SHIFT[s])]


@njit(uint64(int64, uint64), inline='always', cache=False, nogil=True)
def queen_attacks(s, occ):
    return rook_attacks(s, occ) | bishop_attacks(s, occ)
