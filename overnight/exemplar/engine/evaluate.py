"""Tapered hand-crafted evaluation.

Every weight lives in one flat int32 array (EVAL_PARAMS) so the Texel tuner can
mutate it in place without recompiling: numba freezes a *global array* by its
data pointer, so in-place writes are visible to already-compiled code.

Score is returned from the side-to-move's point of view, in centipawns.
"""
import numpy as np
from numba import njit, uint64, uint8, int64, int32, types

from .tables import (
    U, ONE, FULL,
    WP, WN, WB, WR, WQ, WK, BP, BN, BB_, BR, BQ, BK, EMPTY,
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK,
    KNIGHT_ATT, KING_ATT, PAWN_ATT, FILE_BB, RANK_BB, ADJ_FILES,
    PASSED, FORWARD, NEIGHBOURS, KING_RING, SQ_DIST, CENTRE_MAN_DIST,
    lsb, popcount, rook_attacks, bishop_attacks, queen_attacks,
)
from .board import IOCC_W, IOCC_B, IOCC_A, ISIDE, IPHASE, PHASE_TOTAL
from .layout import OI_EVALP, NPARAM as LAYOUT_NPARAM

# --------------------------------------------------------------------------
# parameter vector layout
# --------------------------------------------------------------------------
_SPEC = [
    ("MAT",        6 * 2),        # material mg,eg per piece type
    ("PST",        6 * 64 * 2),   # piece-square, white pov, mg/eg interleaved
    ("MOB_N",      9 * 2), ("MOB_B", 14 * 2), ("MOB_R", 15 * 2), ("MOB_Q", 28 * 2),
    ("PASSED",     8 * 2),        # by relative rank
    ("PASSED_BLOCK", 8 * 2),      # bonus if path to promotion is clear
    ("ISOLATED",   2), ("DOUBLED", 2), ("BACKWARD", 2), ("CONNECTED", 8 * 2),
    ("BISHOP_PAIR", 2), ("ROOK_OPEN", 2), ("ROOK_SEMI", 2), ("ROOK_7TH", 2),
    ("KNIGHT_OUTPOST", 2), ("BISHOP_OUTPOST", 2),
    ("KS_WEIGHT",  6), ("KS_SCALE", 2), ("KS_SHELTER", 4 * 2),
    ("THREAT_PAWN", 2), ("THREAT_MINOR", 2), ("THREAT_ROOK", 2),
    ("TEMPO",      1), ("KING_PP_DIST", 2),
]
OFF = {}
_o = 0
for _n, _s in _SPEC:
    OFF[_n] = _o
    _o += _s
NPARAM = _o

O_MAT = OI_EVALP + OFF["MAT"]; O_PST = OI_EVALP + OFF["PST"]
O_MOBN = OI_EVALP + OFF["MOB_N"]; O_MOBB = OI_EVALP + OFF["MOB_B"]; O_MOBR = OI_EVALP + OFF["MOB_R"]; O_MOBQ = OI_EVALP + OFF["MOB_Q"]
O_PASSED = OI_EVALP + OFF["PASSED"]; O_PASSED_BLOCK = OI_EVALP + OFF["PASSED_BLOCK"]
O_ISO = OI_EVALP + OFF["ISOLATED"]; O_DOUB = OI_EVALP + OFF["DOUBLED"]; O_BACK = OI_EVALP + OFF["BACKWARD"]
O_CONN = OI_EVALP + OFF["CONNECTED"]
O_PAIR = OI_EVALP + OFF["BISHOP_PAIR"]; O_ROPEN = OI_EVALP + OFF["ROOK_OPEN"]; O_RSEMI = OI_EVALP + OFF["ROOK_SEMI"]
O_R7 = OI_EVALP + OFF["ROOK_7TH"]; O_NOUT = OI_EVALP + OFF["KNIGHT_OUTPOST"]; O_BOUT = OI_EVALP + OFF["BISHOP_OUTPOST"]
O_KSW = OI_EVALP + OFF["KS_WEIGHT"]; O_KSS = OI_EVALP + OFF["KS_SCALE"]; O_KSH = OI_EVALP + OFF["KS_SHELTER"]
O_THP = OI_EVALP + OFF["THREAT_PAWN"]; O_THM = OI_EVALP + OFF["THREAT_MINOR"]; O_THR = OI_EVALP + OFF["THREAT_ROOK"]
O_TEMPO = OI_EVALP + OFF["TEMPO"]; O_KPP = OI_EVALP + OFF["KING_PP_DIST"]

assert NPARAM == LAYOUT_NPARAM, f"layout.NPARAM must be {NPARAM}"
EVAL_PARAMS = np.zeros(NPARAM, dtype=np.int32)


# --------------------------------------------------------------------------
# principled initial values (Texel tuning refines them later)
# --------------------------------------------------------------------------
def _init_params(p):
    def st(name, i, mg, eg=None):
        p[OFF[name] + 2 * i] = mg
        p[OFF[name] + 2 * i + 1] = mg if eg is None else eg

    for i, (mg, eg) in enumerate([(82, 94), (337, 281), (365, 297),
                                  (477, 512), (1025, 936), (0, 0)]):
        st("MAT", i, mg, eg)

    # --- piece-square tables, generated from simple positional principles ---
    def centre(sq):
        f, r = sq & 7, sq >> 3
        return -(abs(2 * f - 7) + abs(2 * r - 7)) // 2      # 0 centre .. -7 corner

    for pt in range(6):
        for s in range(64):
            f, r = s & 7, s >> 3
            mg = eg = 0
            if pt == PAWN:
                if 1 <= r <= 6:
                    mg = [0, 0, 2, 6, 14, 28, 50, 0][r] + (3 - abs(2 * f - 7) // 2) * (r >= 2)
                    eg = [0, 0, 6, 14, 30, 60, 100, 0][r]
                    if r == 1 and 2 <= f <= 5:
                        mg -= 4                              # undeveloped centre pawns
            elif pt == KNIGHT:
                mg = 4 * centre(s) + 22 + (4 if 2 <= r <= 5 else 0)
                eg = 4 * centre(s) + 20
            elif pt == BISHOP:
                mg = 3 * centre(s) + 14 + (6 if r >= 1 else 0)
                eg = 3 * centre(s) + 12
            elif pt == ROOK:
                mg = (2 if 2 <= f <= 5 else 0) + (14 if r == 6 else 0)
                eg = 3
            elif pt == QUEEN:
                mg = 2 * centre(s) + (0 if r == 0 else 3)
                eg = 3 * centre(s)
            elif pt == KING:
                mg = ([12, 16, 6, -6, -8, -6, 6, 10][f] if r == 0 else
                      [-8, -10, -14, -18, -20, -22, -26, -30][min(r, 7)])
                eg = 5 * centre(s) + 12
            idx = OFF["PST"] + (pt * 64 + s) * 2
            p[idx] = mg
            p[idx + 1] = eg

    # --- mobility: concave, most value in the first few squares -------------
    for name, n, mgs, egs in (("MOB_N", 9, 6.0, 6.5), ("MOB_B", 14, 5.5, 6.0),
                              ("MOB_R", 15, 3.0, 6.0), ("MOB_Q", 28, 1.8, 3.2)):
        mid = (n - 1) / 2.0
        for i in range(n):
            st(name, i, int(round(mgs * (i - mid) * 0.9)), int(round(egs * (i - mid) * 0.9)))

    for r, (mg, eg) in enumerate([(0, 0), (2, 8), (5, 16), (12, 36),
                                  (25, 68), (50, 118), (85, 190), (0, 0)]):
        st("PASSED", r, mg, eg)
        st("PASSED_BLOCK", r, mg // 4, eg // 3)
    for r in range(8):
        st("CONNECTED", r, 3 + r * 2, 2 + r * 3)

    st("ISOLATED", 0, -12, -16)
    st("DOUBLED", 0, -8, -22)
    st("BACKWARD", 0, -8, -12)
    st("BISHOP_PAIR", 0, 28, 52)
    st("ROOK_OPEN", 0, 28, 12)
    st("ROOK_SEMI", 0, 12, 6)
    st("ROOK_7TH", 0, 12, 22)
    st("KNIGHT_OUTPOST", 0, 26, 12)
    st("BISHOP_OUTPOST", 0, 16, 6)
    for i, w in enumerate([0, 20, 18, 32, 62, 0]):       # per attacking piece type
        p[OFF["KS_WEIGHT"] + i] = w
    p[OFF["KS_SCALE"]] = 22       # divisor for the quadratic danger term
    p[OFF["KS_SCALE"] + 1] = 3    # eg damping
    for i, (mg, eg) in enumerate([(-6, 0), (-22, 0), (-38, 0), (-52, 0)]):
        st("KS_SHELTER", i, mg, eg)
    st("THREAT_PAWN", 0, 34, 42)
    st("THREAT_MINOR", 0, 28, 32)
    st("THREAT_ROOK", 0, 42, 24)
    p[OFF["TEMPO"]] = 14
    st("KING_PP_DIST", 0, 0, 8)


_init_params(EVAL_PARAMS)

MOB_OFFSETS = np.array([0, O_MOBN, O_MOBB, O_MOBR, O_MOBQ, 0], dtype=np.int64)

# side-relative rank of a square
REL_RANK = np.zeros((2, 64), dtype=np.int64)
for _c in range(2):
    for _s in range(64):
        REL_RANK[_c][_s] = (_s >> 3) if _c == WHITE else 7 - (_s >> 3)

# non-pawn material for null-move / endgame guards, in "phase" units
NPM_VALUE = np.array([0, 337, 365, 477, 1025, 0], dtype=np.int64)


# ==========================================================================
@njit(types.UniTuple(int64, 2)(uint64[:], uint8[:], int64, int32[:]),
      cache=False, nogil=True)
def _pawn_eval_side(pos, sq, c, P):
    """Pawn-structure terms for colour c, returned as (mg, eg)."""
    own = pos[WP + 6 * c]
    opp = pos[WP + 6 * (1 - c)]
    mg = int64(0); eg = int64(0)
    b = own
    while b:
        s = lsb(b)
        b &= b - ONE
        f = s & 7
        rr = REL_RANK[c][s]
        # doubled
        if own & FORWARD[c][s]:
            mg += P[O_DOUB]; eg += P[O_DOUB + 1]
        neighbours = own & ADJ_FILES[f]
        if neighbours == U(0):
            mg += P[O_ISO]; eg += P[O_ISO + 1]
        else:
            # connected / phalanx: supported from behind or side by side
            if (PAWN_ATT[1 - c][s] & own) or (own & ADJ_FILES[f] & RANK_BB[s >> 3]):
                mg += P[O_CONN + 2 * rr]; eg += P[O_CONN + 2 * rr + 1]
            elif (own & NEIGHBOURS[c][s]) == U(0):
                mg += P[O_BACK]; eg += P[O_BACK + 1]
        # passed
        if (opp & PASSED[c][s]) == U(0):
            mg += P[O_PASSED + 2 * rr]; eg += P[O_PASSED + 2 * rr + 1]
            if (pos[IOCC_A] & FORWARD[c][s]) == U(0):
                mg += P[O_PASSED_BLOCK + 2 * rr]
                eg += P[O_PASSED_BLOCK + 2 * rr + 1]
    return mg, eg


@njit(int64(uint64[:], uint8[:], int32[:]), cache=False, nogil=True)
def evaluate(pos, sq, P):
    """Static evaluation in centipawns, from the side to move's perspective."""
    occ = pos[IOCC_A]
    mg = int64(0); eg = int64(0)

    wpawn_att = (((pos[WP] & U(0x7f7f7f7f7f7f7f7f)) << U(9)) |
                 ((pos[WP] & U(0xfefefefefefefefe)) << U(7)))
    bpawn_att = (((pos[BP] & U(0x7f7f7f7f7f7f7f7f)) >> U(7)) |
                 ((pos[BP] & U(0xfefefefefefefefe)) >> U(9)))

    for c in range(2):
        sgn = int64(1) if c == WHITE else int64(-1)
        o = 6 * c
        them = 1 - c
        eo = 6 * them
        ours = pos[IOCC_W + c]
        theirs = pos[IOCC_W + them]
        own_pawns = pos[WP + o]
        opp_pawns = pos[WP + eo]
        opp_pawn_att = bpawn_att if c == WHITE else wpawn_att
        own_pawn_att = wpawn_att if c == WHITE else bpawn_att
        eksq = lsb(pos[WK + eo])
        ekring = KING_RING[eksq]
        mob_area = ~(own_pawns | pos[WK + o] | opp_pawn_att)

        ks_attackers = int64(0)
        ks_weight = int64(0)
        cmg = int64(0); ceg = int64(0)

        for pt in range(6):
            b = pos[pt + o]
            cnt = popcount(b)
            cmg += cnt * P[O_MAT + 2 * pt]
            ceg += cnt * P[O_MAT + 2 * pt + 1]
            while b:
                s = lsb(b)
                b &= b - ONE
                ps = s if c == WHITE else (s ^ 56)
                pidx = O_PST + (pt * 64 + ps) * 2
                cmg += P[pidx]; ceg += P[pidx + 1]

                if pt == PAWN or pt == KING:
                    continue
                if pt == KNIGHT:
                    att = KNIGHT_ATT[s]
                elif pt == BISHOP:
                    att = bishop_attacks(s, occ ^ pos[WQ + o])
                elif pt == ROOK:
                    att = rook_attacks(s, occ ^ pos[WQ + o] ^ pos[WR + o])
                else:
                    att = queen_attacks(s, occ)

                m = popcount(att & mob_area)
                moff = MOB_OFFSETS[pt]
                cmg += P[moff + 2 * m]; ceg += P[moff + 2 * m + 1]

                ka = att & ekring
                if ka:
                    ks_attackers += 1
                    ks_weight += P[O_KSW + pt] * popcount(ka)

                # rook files / 7th rank
                if pt == ROOK:
                    ff = FILE_BB[s & 7]
                    if (own_pawns & ff) == U(0):
                        if (opp_pawns & ff) == U(0):
                            cmg += P[O_ROPEN]; ceg += P[O_ROPEN + 1]
                        else:
                            cmg += P[O_RSEMI]; ceg += P[O_RSEMI + 1]
                    if REL_RANK[c][s] == 6:
                        cmg += P[O_R7]; ceg += P[O_R7 + 1]
                # outposts: protected by a pawn, unreachable by enemy pawns
                elif (pt == KNIGHT or pt == BISHOP) and REL_RANK[c][s] >= 3:
                    if (own_pawn_att & (ONE << U(s))) and \
                            (opp_pawns & PASSED[c][s] & ADJ_FILES[s & 7]) == U(0):
                        if pt == KNIGHT:
                            cmg += P[O_NOUT]; ceg += P[O_NOUT + 1]
                        else:
                            cmg += P[O_BOUT]; ceg += P[O_BOUT + 1]

        # bishop pair
        if popcount(pos[WB + o]) >= 2:
            cmg += P[O_PAIR]; ceg += P[O_PAIR + 1]

        # king danger, quadratic in accumulated attack weight
        if ks_attackers >= 2:
            d = ks_weight
            cmg += (d * d) // (int64(P[O_KSS]) * 16)
            ceg += (d * d) // (int64(P[O_KSS]) * 16 * max(int64(1), int64(P[O_KSS + 1])))

        # pawn shelter in front of our own king
        ksq = lsb(pos[WK + o])
        kf = ksq & 7
        shelter = int64(0)
        for df in range(-1, 2):
            nf = kf + df
            if 0 <= nf < 8:
                # squares on file nf strictly ahead of the king's rank
                if (own_pawns & FORWARD[c][ksq - kf + nf]) == U(0):
                    shelter += 1
        si = min(int64(3), shelter)
        cmg += P[O_KSH + 2 * si]; ceg += P[O_KSH + 2 * si + 1]

        # threats: our attacks landing on undefended-or-more-valuable enemy pieces
        weak = theirs & ~opp_pawn_att
        t = own_pawn_att & weak & (pos[WN + eo] | pos[WB + eo] | pos[WR + eo] | pos[WQ + eo])
        n = popcount(t)
        cmg += n * P[O_THP]; ceg += n * P[O_THP + 1]

        # pawn structure
        pmg, peg = _pawn_eval_side(pos, sq, c, P)
        cmg += pmg
        ceg += peg

        mg += sgn * cmg
        eg += sgn * ceg

    # king proximity to passed pawns matters in the endgame
    ph = int64(pos[IPHASE])
    if ph > PHASE_TOTAL:
        ph = PHASE_TOTAL
    num = mg * ph + eg * (PHASE_TOTAL - ph)
    # truncate toward zero, not floor: floor division makes eval(pos) != -eval(mirror)
    if num >= 0:
        score = num // PHASE_TOTAL
    else:
        score = -((-num) // PHASE_TOTAL)

    if pos[ISIDE] == U(BLACK):
        score = -score
    return score + P[O_TEMPO]


@njit(int64(uint64[:], int64), cache=False, nogil=True)
def non_pawn_material(pos, c):
    o = 6 * c
    v = int64(0)
    for pt in range(1, 5):
        v += NPM_VALUE[pt] * popcount(pos[pt + o])
    return v


def install_params(I, params=None):
    """Copy the evaluation weights into the shared int32 state block."""
    src = EVAL_PARAMS if params is None else params
    I[OI_EVALP:OI_EVALP + NPARAM] = src
    return I
