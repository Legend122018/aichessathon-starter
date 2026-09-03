"""AI Chessathon submission.

A negamax search with alpha-beta, iterative deepening, a transposition table that
survives between moves, MVV-LVA / killer / history move ordering, null-move pruning,
late move reductions and a quiescence search, over a tapered material + piece-square
evaluation.

The platform imports this module once per game and calls get_move for each of our
turns. Everything above get_move is built at import time, inside the 60 s budget.
"""

from __future__ import annotations

import random
import time
from collections.abc import Hashable, Iterable
from typing import Any

import chess

# --------------------------------------------------------------------------------------
# Evaluation tables
#
# Material values and piece-square tables from PeSTO, tapered between a midgame and an
# endgame set. Tables are written with rank 8 first, so white reads them at `square ^ 56`
# and black reads them at `square`.
# --------------------------------------------------------------------------------------

MG_VALUE = (0, 0, 0, 0, 0, 0, 0)
EG_VALUE = (0, 0, 0, 0, 0, 0, 0)

# Phase weights per piece type; a full board is 24.
PHASE_WEIGHT = (0, 0, 1, 1, 2, 4, 0)
TOTAL_PHASE = 24

MG_PAWN = (
      100,   100,   100,   100,   100,   100,   100,   100,
      148,   143,   142,   162,   152,    88,    41,    86,
      124,   115,   141,   156,   163,   177,   121,   148,
       96,   124,   121,   133,   135,   130,   131,    99,
       83,    92,   104,   112,   116,    99,    95,    84,
       83,    93,    97,    71,    80,    89,   126,    82,
       60,    91,    82,    50,    54,   117,   115,    68,
      100,   100,   100,   100,   100,   100,   100,   100,
)  # fmt: skip
EG_PAWN = (
      100,   100,   100,   100,   100,   100,   100,   100,
      163,   186,   177,   169,   204,   197,   203,   172,
      102,   118,   104,    80,    79,   100,   117,    85,
      101,   104,   104,    96,    92,   106,   102,    93,
      100,   104,    92,    87,    91,   100,   106,   100,
       99,   108,    98,   101,   102,   115,    98,    92,
      109,   115,   128,   135,   124,   123,   119,   106,
      100,   100,   100,   100,   100,   100,   100,   100,
)  # fmt: skip
MG_KNIGHT = (
      253,   285,   304,   431,   439,   195,   362,   386,
      399,   369,   487,   464,   409,   501,   359,   456,
      396,   492,   484,   523,   564,   537,   516,   420,
      455,   445,   468,   489,   481,   498,   474,   466,
      432,   410,   447,   445,   453,   463,   454,   440,
      391,   424,   435,   439,   450,   444,   450,   410,
      362,   388,   407,   407,   419,   434,   410,   392,
      318,   378,   361,   407,   403,   395,   388,   356,
)  # fmt: skip
EG_KNIGHT = (
      105,   205,   241,   184,   199,   254,   184,    74,
      196,   226,   197,   231,   239,   204,   229,   209,
      215,   226,   249,   243,   218,   238,   212,   220,
      231,   261,   255,   260,   266,   255,   254,   233,
      234,   248,   266,   268,   275,   258,   241,   227,
      241,   259,   254,   266,   272,   255,   255,   226,
      219,   257,   245,   264,   259,   245,   234,   224,
      246,   214,   261,   241,   252,   244,   228,   180,
)  # fmt: skip
MG_BISHOP = (
      346,   328,   346,   320,   286,   238,   365,   385,
      377,   418,   447,   412,   400,   466,   407,   435,
      412,   469,   443,   526,   476,   510,   493,   450,
      433,   438,   437,   472,   485,   468,   451,   444,
      456,   426,   445,   453,   462,   446,   416,   451,
      429,   454,   447,   441,   438,   447,   461,   422,
      455,   449,   436,   418,   433,   431,   447,   426,
      422,   425,   398,   396,   407,   394,   402,   395,
)  # fmt: skip
EG_BISHOP = (
      278,   270,   283,   269,   270,   295,   269,   273,
      270,   266,   256,   268,   269,   247,   264,   233,
      269,   254,   279,   255,   262,   257,   244,   255,
      253,   269,   272,   278,   254,   250,   276,   245,
      232,   265,   285,   272,   272,   274,   262,   232,
      265,   267,   271,   279,   278,   273,   252,   262,
      250,   251,   264,   268,   269,   266,   264,   267,
      278,   264,   249,   262,   258,   268,   252,   273,
)  # fmt: skip
MG_ROOK = (
      567,   572,   602,   650,   601,   628,   597,   608,
      579,   574,   622,   658,   655,   652,   612,   627,
      549,   586,   607,   652,   649,   637,   645,   576,
      538,   540,   560,   580,   589,   604,   572,   566,
      520,   526,   541,   537,   549,   543,   532,   527,
      505,   533,   522,   527,   549,   544,   544,   533,
      497,   513,   527,   530,   535,   551,   520,   495,
      520,   528,   535,   548,   557,   550,   527,   528,
)  # fmt: skip
EG_ROOK = (
      453,   450,   443,   426,   435,   436,   443,   444,
      447,   450,   439,   430,   427,   421,   435,   425,
      446,   433,   427,   410,   405,   415,   409,   434,
      452,   457,   444,   440,   427,   417,   426,   427,
      456,   444,   454,   452,   442,   441,   433,   450,
      450,   449,   457,   448,   438,   445,   438,   441,
      451,   447,   435,   445,   443,   433,   434,   445,
      448,   450,   447,   440,   441,   440,   454,   426,
)  # fmt: skip
MG_QUEEN = (
     1047,  1118,  1189,  1232,  1182,  1234,  1160,  1086,
     1098,  1038,  1107,  1083,  1113,  1161,  1147,  1119,
     1065,  1082,  1103,  1137,  1141,  1187,  1167,  1125,
     1073,  1078,  1078,  1094,  1092,  1098,  1078,  1114,
     1086,  1101,  1081,  1079,  1091,  1075,  1089,  1085,
     1072,  1089,  1089,  1088,  1073,  1090,  1095,  1095,
     1089,  1095,  1103,  1095,  1093,  1101,  1104,  1094,
     1085,  1087,  1073,  1077,  1072,  1077,  1108,  1072,
)  # fmt: skip
EG_QUEEN = (
      770,   750,   719,   736,   779,   706,   727,   768,
      782,   836,   823,   858,   846,   820,   783,   799,
      805,   792,   825,   831,   845,   819,   816,   803,
      775,   805,   849,   866,   869,   875,   860,   805,
      737,   754,   825,   853,   822,   835,   793,   795,
      751,   752,   785,   788,   813,   784,   768,   755,
      724,   717,   734,   748,   733,   737,   671,   703,
      740,   713,   714,   731,   710,   666,   613,   717,
)  # fmt: skip
MG_KING = (
       94,   189,   164,   160,   102,   241,   183,   159,
      121,   120,    19,    23,    59,    47,   100,    94,
        6,     9,   -62,  -124,  -140,   -53,   -14,    26,
     -113,   -45,  -101,  -175,  -172,  -118,   -36,   -74,
     -105,   -82,  -138,   -57,  -122,  -123,   -81,   -78,
      -39,   -12,   -63,   -86,  -100,   -79,   -30,   -55,
       31,    30,   -14,   -54,   -67,   -19,     6,    35,
       17,    77,    56,   -35,    13,   -21,    58,    57,
)  # fmt: skip
EG_KING = (
      -66,   -39,    -9,   -27,    -7,   -64,   -61,   -97,
      -26,     5,    46,    34,    22,    34,    -4,   -31,
       -2,    40,    62,    57,    63,    52,    36,   -24,
        9,    30,    48,    58,    53,    48,    17,   -16,
      -36,    24,    34,    19,    33,    28,    20,   -26,
      -19,   -11,    14,    15,    22,    16,    -2,   -20,
      -32,   -18,    -3,     2,    11,     4,    -8,   -33,
      -16,   -47,   -46,   -39,   -51,   -24,   -44,   -43,
)  # fmt: skip

MG_PST = (MG_PAWN, MG_PAWN, MG_KNIGHT, MG_BISHOP, MG_ROOK, MG_QUEEN, MG_KING)
EG_PST = (EG_PAWN, EG_PAWN, EG_KNIGHT, EG_BISHOP, EG_ROOK, EG_QUEEN, EG_KING)


# Per-piece lookup: (black_table, white_table), each 64 entries.
ColourTables = tuple[tuple[int, ...], tuple[int, ...]]


def _build_tables() -> tuple[tuple[ColourTables, ...], tuple[ColourTables, ...]]:
    """Fold material value into the piece-square tables, one table per colour.

    Indexed [piece_type][colour][square], so evaluation is a single lookup per piece.
    """
    mg: list[ColourTables] = []
    eg: list[ColourTables] = []
    for piece in range(7):
        mg_black = tuple(MG_VALUE[piece] + MG_PST[piece][sq] for sq in range(64))
        mg_white = tuple(MG_VALUE[piece] + MG_PST[piece][sq ^ 56] for sq in range(64))
        eg_black = tuple(EG_VALUE[piece] + EG_PST[piece][sq] for sq in range(64))
        eg_white = tuple(EG_VALUE[piece] + EG_PST[piece][sq ^ 56] for sq in range(64))
        mg.append((mg_black, mg_white))
        eg.append((eg_black, eg_white))
    return tuple(mg), tuple(eg)


MG_TABLE, EG_TABLE = _build_tables()

BISHOP_PAIR_MG = 56
BISHOP_PAIR_EG = 44
ROOK_OPEN_FILE = 54
ROOK_SEMI_OPEN_FILE = 35
TEMPO = 12

FILE_MASKS = tuple(chess.BB_FILES[chess.square_file(sq)] for sq in range(64))

# Manhattan distance from each square to the centre, for the mop-up term below.
CENTER_DISTANCE = (
    6, 5, 4, 3, 3, 4, 5, 6,
    5, 4, 3, 2, 2, 3, 4, 5,
    4, 3, 2, 1, 1, 2, 3, 4,
    3, 2, 1, 0, 0, 1, 2, 3,
    3, 2, 1, 0, 0, 1, 2, 3,
    4, 3, 2, 1, 1, 2, 3, 4,
    5, 4, 3, 2, 2, 3, 4, 5,
    6, 5, 4, 3, 3, 4, 5, 6,
)  # fmt: skip
MOP_UP_THRESHOLD = 450

# --------------------------------------------------------------------------------------
# Pawn structure
#
# Structure is expensive relative to everything else in the evaluation, but it changes
# only when a pawn moves or is captured, so it is cached on the pair of pawn bitboards.
# Hit rates are high enough that the terms are close to free.
# --------------------------------------------------------------------------------------

PASSED_MG = (0, -5, -30, -23, -4, 15, 27, 0)
PASSED_EG = (0, 8, 26, 43, 52, 89, 110, 0)
ISOLATED_MG, ISOLATED_EG = 16, 7
DOUBLED_MG, DOUBLED_EG = 7, 25
SHELTER_PENALTY = 35


def _pawn_masks() -> tuple[tuple[int, ...], ...]:
    """Adjacent-file, front-span and passed-pawn masks, per square and colour."""
    adjacent: list[int] = []
    for file_index in range(8):
        mask = 0
        if file_index > 0:
            mask |= chess.BB_FILES[file_index - 1]
        if file_index < 7:
            mask |= chess.BB_FILES[file_index + 1]
        adjacent.append(mask)

    front_white: list[int] = []
    front_black: list[int] = []
    passed_white: list[int] = []
    passed_black: list[int] = []
    for square in range(64):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        ahead_white = 0
        for rank in range(rank_index + 1, 8):
            ahead_white |= chess.BB_RANKS[rank]
        ahead_black = 0
        for rank in range(rank_index):
            ahead_black |= chess.BB_RANKS[rank]
        own_file = chess.BB_FILES[file_index]
        three = own_file | adjacent[file_index]
        front_white.append(ahead_white & own_file)
        front_black.append(ahead_black & own_file)
        passed_white.append(ahead_white & three)
        passed_black.append(ahead_black & three)
    return (
        tuple(adjacent),
        tuple(front_black),
        tuple(front_white),
        tuple(passed_black),
        tuple(passed_white),
    )


ADJACENT_FILES, FRONT_BLACK, FRONT_WHITE, PASSED_BLACK, PASSED_WHITE = _pawn_masks()
FRONT_SPAN = (FRONT_BLACK, FRONT_WHITE)
PASSED_MASK = (PASSED_BLACK, PASSED_WHITE)

PAWN_CACHE: dict[tuple[int, int], tuple[int, int]] = {}
PAWN_CACHE_MAX = 200_000


def pawn_structure(white_pawns: int, black_pawns: int) -> tuple[int, int]:
    """White-relative (midgame, endgame) score for passed, isolated and doubled pawns."""
    key = (white_pawns, black_pawns)
    cached = PAWN_CACHE.get(key)
    if cached is not None:
        return cached

    mg = 0
    eg = 0
    for square in chess.scan_forward(white_pawns):
        rank = square >> 3
        if not PASSED_WHITE[square] & black_pawns:
            mg += PASSED_MG[rank]
            eg += PASSED_EG[rank]
        if not ADJACENT_FILES[square & 7] & white_pawns:
            mg -= ISOLATED_MG
            eg -= ISOLATED_EG
        if FRONT_WHITE[square] & white_pawns:
            mg -= DOUBLED_MG
            eg -= DOUBLED_EG
    for square in chess.scan_forward(black_pawns):
        rank = 7 - (square >> 3)
        if not PASSED_BLACK[square] & white_pawns:
            mg -= PASSED_MG[rank]
            eg -= PASSED_EG[rank]
        if not ADJACENT_FILES[square & 7] & black_pawns:
            mg += ISOLATED_MG
            eg += ISOLATED_EG
        if FRONT_BLACK[square] & black_pawns:
            mg += DOUBLED_MG
            eg += DOUBLED_EG

    if len(PAWN_CACHE) < PAWN_CACHE_MAX:
        PAWN_CACHE[key] = (mg, eg)
    return mg, eg


def shelter(king_square: int, own_pawns: int) -> int:
    """Midgame penalty for a king sitting behind open files."""
    file_index = king_square & 7
    penalty = 0
    if not chess.BB_FILES[file_index] & own_pawns:
        penalty += SHELTER_PENALTY
    if file_index > 0 and not chess.BB_FILES[file_index - 1] & own_pawns:
        penalty += SHELTER_PENALTY
    if file_index < 7 and not chess.BB_FILES[file_index + 1] & own_pawns:
        penalty += SHELTER_PENALTY
    return penalty

# --------------------------------------------------------------------------------------
# Search constants and state
# --------------------------------------------------------------------------------------

MATE = 30000
MATE_THRESHOLD = MATE - 1000
MAX_PLY = 64

EXACT, LOWER, UPPER = 0, 1, 2

# depth, score, flag, best move
TTEntry = tuple[int, int, int, chess.Move | None]

# Kept on the module so it survives between our moves in one game. The platform starts a
# fresh process per game, so this never leaks across games.
TT: dict[Hashable, TTEntry] = {}
TT_MAX = 500_000

KILLERS: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY + 8)]
HISTORY: dict[tuple[bool, int, int], int] = {}
# Counter moves: the reply that most recently refuted this exact opponent move.
COUNTER: dict[tuple[int, int], chess.Move] = {}

# Position keys we have already been asked to move in, so a repetition is visible to the
# search even though each FEN arrives without its game history.
SEEN: dict[Hashable, int] = {}

MVV_LVA_VICTIM = (0, 100, 320, 330, 500, 900, 0)

# Static exchange evaluation uses a king value large enough that a king capture always
# dominates, without overflowing the arithmetic.
SEE_VALUE = (0, 100, 320, 330, 500, 900, 20000)
ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def cheapest_attacker(
    board: chess.Board, attackers: chess.Bitboard, colour: chess.Color
) -> tuple[int, int]:
    """Square and type of the least valuable attacker in `attackers`, or (-1, 0)."""
    for piece in ORDER:
        subset = attackers & board.pieces_mask(piece, colour)
        if subset:
            return chess.lsb(subset), piece
    return -1, 0


def see(board: chess.Board, move: chess.Move) -> int:
    """What a capture is worth once both sides trade optimally on the target square.

    Negative means the capture loses material even though it wins a piece up front, which
    is the case alpha-beta wastes the most time on. Occupancy is stripped as pieces are
    swapped off so that x-rays behind them are counted.
    """
    target = move.to_square
    en_passant = board.is_en_passant(move)
    if en_passant:
        captured = chess.PAWN
    else:
        found = board.piece_type_at(target)
        captured = found if found is not None else 0
    attacker = board.piece_type_at(move.from_square)
    if attacker is None:
        return 0

    gains = [SEE_VALUE[captured]]
    occupied = board.occupied & ~chess.BB_SQUARES[move.from_square]
    if en_passant:
        behind = target - 8 if board.turn == chess.WHITE else target + 8
        occupied &= ~chess.BB_SQUARES[behind]
    if move.promotion is not None:
        gains[0] += SEE_VALUE[move.promotion] - SEE_VALUE[chess.PAWN]
        attacker = move.promotion

    colour = not board.turn
    index = 0
    while True:
        attackers = board.attackers_mask(colour, target, occupied) & occupied
        square, piece = cheapest_attacker(board, attackers, colour)
        if square < 0:
            break
        index += 1
        gains.append(SEE_VALUE[attacker] - gains[index - 1])
        if max(-gains[index - 1], gains[index]) < 0:
            break
        attacker = piece
        occupied &= ~chess.BB_SQUARES[square]
        colour = not colour

    while index:
        gains[index - 1] = -max(-gains[index - 1], gains[index])
        index -= 1
    return gains[0]


# Opening book. Standard main lines, written out as they appear in any openings
# reference, so the early moves cost no clock and no early blunder can happen where the
# search is shallowest. Every line is verified for legality when the book is built.
BOOK_LINES = (
    # 1. e4 e5, Ruy Lopez
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Na5 Bc2 c5 d4",
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Nxe4 d4 b5 Bb3 d5 dxe5 Be6 c3 Be7 Re1",
    "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 d4 Nd6 Bxc6 dxc6 dxe5 Nf5 Qxd8+ Kxd8",
    # 1. e4 e5, Italian
    "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O O-O Re1 a6 Bb3 Ba7 h3",
    "e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Bc5 c3 d6 O-O O-O Re1 a6 Bb3",
    # 1. e4 e5, Scotch and Four Knights
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5 c4 Ba6",
    "e4 e5 Nf3 Nc6 Nc3 Nf6 Bb5 Bb4 O-O O-O d3 d6 Bg5 Bxc3 bxc3 Qe7",
    # 1. e4 c5, Open Sicilian
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e5 Nb3 Be7 O-O O-O",
    "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6 Bg5 a6 Na3 b5",
    "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7 Be2 a6 O-O Nf6 Be3 Bb4",
    # 1. e4 e6 French, 1. e4 c6 Caro-Kann
    "e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3 Ne7 Qg4 O-O",
    "e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7 Bd3 c5 c3 Nc6 Ne2 cxd4 cxd4 f6",
    "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7 Bd3 Bxd3 Qxd3",
    "e4 c6 d4 d5 e5 Bf5 Nf3 e6 Be2 c5 Be3 Nd7 c4 dxc4 Na3",
    # 1. e4 d5, 1. e4 Nf6, 1. e4 d6
    "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5 Bd2 e6 Qe2",
    "e4 Nf6 e5 Nd5 d4 d6 Nf3 g6 Bc4 Nb6 Bb3 Bg7 Qe2 Nc6 O-O O-O",
    "e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7 Be2 O-O O-O c6 a4 Qc7",
    # 1. d4 as Black
    "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5 O-O Nc6 a3 Bxc3 bxc3 dxc4 Bxc4",
    "d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 O-O e3 h6 Bh4 b6 cxd5 Nxd5",
    "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bh4 b6 cxd5 Nxd5 Bxe7 Qxe7",
    "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4 O-O Nbd7",
    "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 c5 Rb1 O-O Be2",
    "d4 Nf6 Nf3 e6 c4 d5 Nc3 Be7 Bf4 O-O e3 c5 dxc5 Bxc5",
    # Flank openings
    "Nf3 d5 d4 Nf6 c4 e6 Nc3 Be7 Bg5 O-O e3 h6 Bh4 b6",
    "c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Nb6 O-O Be7",
    "g3 d5 Nf3 Nf6 Bg2 e6 O-O Be7 d4 O-O c4 c6 Nbd2 Nbd7",
)


def _build_book() -> dict[str, tuple[str, ...]]:
    """Turn the lines above into a position lookup, discarding anything illegal."""
    lines: dict[str, list[str]] = {}
    for line in BOOK_LINES:
        board = chess.Board()
        for token in line.split():
            try:
                move = board.parse_san(token)
            except ValueError:
                break
            key = " ".join(board.fen().split()[:4])
            uci = move.uci()
            bucket = lines.setdefault(key, [])
            if uci not in bucket:
                bucket.append(uci)
            board.push(move)
    return {key: tuple(value) for key, value in lines.items()}


BOOK = _build_book()
# The book knows several deep lines after 1. e4 and 1. d4 and only shallow ones after
# anything else, so as White it opens with one of those two rather than wandering out of
# book on move three.
BOOK[" ".join(chess.Board().fen().split()[:4])] = ("e2e4", "d2d4")

ASPIRATION_WINDOW = 40
NULL_MOVE_R = 2

HISTORY_CAP = 1 << 18
FUTILITY_MARGIN = (0, 150, 320, 520)
DELTA_MARGIN = 200


class TimeUp(Exception):
    """Raised inside the search when the move budget is spent."""


class Searcher:
    """One search, holding the clock and the node counter for it."""

    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.nodes = 0
        self.path: list[Hashable] = []
        # Tapered material + piece-square accumulators, white-relative, kept in step
        # with the board by push/pop rather than rebuilt at every leaf.
        self.mg = 0
        self.eg = 0
        self.phase = 0
        self.stack: list[tuple[int, int, int]] = []

    # ----------------------------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------------------------

    def seed(self, board: chess.Board) -> None:
        """Rebuild the accumulators from scratch. Runs once per search, not per leaf."""
        white = board.occupied_co[chess.WHITE]
        black = board.occupied_co[chess.BLACK]
        mg = 0
        eg = 0
        phase = 0
        for piece, bb in (
            (chess.PAWN, board.pawns),
            (chess.KNIGHT, board.knights),
            (chess.BISHOP, board.bishops),
            (chess.ROOK, board.rooks),
            (chess.QUEEN, board.queens),
            (chess.KING, board.kings),
        ):
            mg_black, mg_white = MG_TABLE[piece]
            eg_black, eg_white = EG_TABLE[piece]
            weight = PHASE_WEIGHT[piece]
            for square in chess.scan_forward(bb & white):
                mg += mg_white[square]
                eg += eg_white[square]
                phase += weight
            for square in chess.scan_forward(bb & black):
                mg -= mg_black[square]
                eg -= eg_black[square]
                phase += weight
        self.mg = mg
        self.eg = eg
        self.phase = phase

    def _add(self, piece: int, colour: chess.Color, square: int) -> None:
        if colour:
            self.mg += MG_TABLE[piece][1][square]
            self.eg += EG_TABLE[piece][1][square]
        else:
            self.mg -= MG_TABLE[piece][0][square]
            self.eg -= EG_TABLE[piece][0][square]
        self.phase += PHASE_WEIGHT[piece]

    def _remove(self, piece: int, colour: chess.Color, square: int) -> None:
        if colour:
            self.mg -= MG_TABLE[piece][1][square]
            self.eg -= EG_TABLE[piece][1][square]
        else:
            self.mg += MG_TABLE[piece][0][square]
            self.eg += EG_TABLE[piece][0][square]
        self.phase -= PHASE_WEIGHT[piece]

    def push(self, board: chess.Board, move: chess.Move) -> None:
        """Make a move on the board and update the accumulators to match."""
        self.stack.append((self.mg, self.eg, self.phase))
        colour = board.turn
        piece = board.piece_type_at(move.from_square)
        if piece is None:
            board.push(move)
            return
        self._remove(piece, colour, move.from_square)
        if piece == chess.PAWN and board.is_en_passant(move):
            behind = move.to_square - 8 if colour else move.to_square + 8
            self._remove(chess.PAWN, not colour, behind)
        else:
            victim = board.piece_type_at(move.to_square)
            if victim is not None:
                self._remove(victim, not colour, move.to_square)
        self._add(move.promotion or piece, colour, move.to_square)
        if piece == chess.KING and abs(move.to_square - move.from_square) == 2:
            if move.to_square > move.from_square:
                rook_from, rook_to = move.to_square + 1, move.to_square - 1
            else:
                rook_from, rook_to = move.to_square - 2, move.to_square + 1
            self._remove(chess.ROOK, colour, rook_from)
            self._add(chess.ROOK, colour, rook_to)
        board.push(move)

    def push_null(self, board: chess.Board) -> None:
        self.stack.append((self.mg, self.eg, self.phase))
        board.push(chess.Move.null())

    def pop(self, board: chess.Board) -> None:
        board.pop()
        self.mg, self.eg, self.phase = self.stack.pop()

    def evaluate(self, board: chess.Board) -> int:
        """Static score in centipawns, from the side to move's point of view."""
        mg = self.mg
        eg = self.eg
        white = board.occupied_co[chess.WHITE]
        black = board.occupied_co[chess.BLACK]

        if chess.popcount(board.bishops & white) > 1:
            mg += BISHOP_PAIR_MG
            eg += BISHOP_PAIR_EG
        if chess.popcount(board.bishops & black) > 1:
            mg -= BISHOP_PAIR_MG
            eg -= BISHOP_PAIR_EG

        white_pawns = board.pawns & white
        black_pawns = board.pawns & black

        pawn_mg, pawn_eg = pawn_structure(white_pawns, black_pawns)
        mg += pawn_mg
        eg += pawn_eg

        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        if white_king is not None:
            mg -= shelter(white_king, white_pawns)
        if black_king is not None:
            mg += shelter(black_king, black_pawns)

        for square in chess.scan_forward(board.rooks & white):
            file_bb = FILE_MASKS[square]
            if not file_bb & white_pawns:
                mg += ROOK_OPEN_FILE if not file_bb & black_pawns else ROOK_SEMI_OPEN_FILE
        for square in chess.scan_forward(board.rooks & black):
            file_bb = FILE_MASKS[square]
            if not file_bb & black_pawns:
                mg -= ROOK_OPEN_FILE if not file_bb & white_pawns else ROOK_SEMI_OPEN_FILE

        phase = min(self.phase, TOTAL_PHASE)
        score = (mg * phase + eg * (TOTAL_PHASE - phase)) // TOTAL_PHASE

        # Mop-up. With a decisive lead and no enemy pawns left, material tells us nothing
        # about progress, so every move looks equal and the search shuffles until the
        # referee claims a repetition. Drive their king to the edge and walk ours in.
        if score > MOP_UP_THRESHOLD or score < -MOP_UP_THRESHOLD:
            strong = chess.WHITE if score > 0 else chess.BLACK
            weak_king = board.king(not strong)
            strong_king = board.king(strong)
            if (
                weak_king is not None
                and strong_king is not None
                and not board.pawns & board.occupied_co[not strong]
            ):
                spread = chess.square_manhattan_distance(strong_king, weak_king)
                bonus = 47 * CENTER_DISTANCE[weak_king] + 16 * (14 - spread)
                score += bonus if strong == chess.WHITE else -bonus

        # Fade the score as the fifty-move counter climbs, so shuffling is never free.
        score = score * (200 - board.halfmove_clock) // 200

        if board.turn == chess.BLACK:
            score = -score
        return score + TEMPO

    # ----------------------------------------------------------------------------------
    # Move ordering
    # ----------------------------------------------------------------------------------

    def order(
        self,
        board: chess.Board,
        moves: Iterable[chess.Move],
        tt_move: chess.Move | None,
        ply: int,
        counter: chess.Move | None = None,
    ) -> list[chess.Move]:
        """Sort moves so that alpha-beta gets its cutoffs early."""
        killers = KILLERS[ply]
        turn = board.turn
        scored: list[tuple[int, chess.Move]] = []
        for move in moves:
            if move == tt_move:
                scored.append((1 << 24, move))
                continue
            victim = board.piece_type_at(move.to_square)
            if victim is not None:
                attacker = board.piece_type_at(move.from_square) or chess.PAWN
                score = (1 << 20) + MVV_LVA_VICTIM[victim] * 16 - attacker
                # A capture that loses the exchange belongs behind the quiet moves, not
                # in front of them. Only the potentially losing ones pay for the check.
                if SEE_VALUE[attacker] > SEE_VALUE[victim] and see(board, move) < 0:
                    score -= 1 << 21
            elif move.promotion is not None:
                score = (1 << 20) + MVV_LVA_VICTIM[move.promotion] * 16
            elif board.is_en_passant(move):
                score = (1 << 20) + MVV_LVA_VICTIM[chess.PAWN] * 16 - chess.PAWN
            elif move == killers[0] or move == killers[1]:
                score = 1 << 19
            elif move == counter:
                score = (1 << 19) - 1
            else:
                score = HISTORY.get((turn, move.from_square, move.to_square), 0)
            scored.append((score, move))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [move for _, move in scored]

    # ----------------------------------------------------------------------------------
    # Quiescence
    # ----------------------------------------------------------------------------------

    def quiesce(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        """Search only forcing moves, so evaluation never lands mid-exchange."""
        self.nodes += 1
        if not self.nodes & 1023 and time.monotonic() > self.deadline:
            raise TimeUp

        in_check = board.is_check()
        if not in_check:
            # Stalemate is scored by the evaluation as a crushing win unless we look for
            # it here, which is exactly how a won ending gets thrown away. The popcount
            # guard keeps the legality check off the hot path in real positions.
            if chess.popcount(board.occupied_co[board.turn]) <= 4 and not any(
                board.generate_legal_moves()
            ):
                return 0
            stand_pat = self.evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
        else:
            stand_pat = -MATE + ply

        if ply >= MAX_PLY:
            return alpha

        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
        else:
            moves = list(board.generate_legal_captures())
            for move in board.generate_legal_moves(board.pawns, ~board.occupied):
                if move.promotion == chess.QUEEN:
                    moves.append(move)

        best = stand_pat
        for move in self.order(board, moves, None, ply):
            if not in_check:
                # Delta pruning: a capture that cannot lift us near alpha is not worth it.
                victim = board.piece_type_at(move.to_square)
                gain = MVV_LVA_VICTIM[victim] if victim is not None else 100
                if move.promotion is not None:
                    gain += 800
                if stand_pat + gain + DELTA_MARGIN < alpha:
                    continue
                # Skip captures that lose material once the square is traded out. Worth
                # computing only when the attacker is the more valuable piece.
                attacker = board.piece_type_at(move.from_square)
                if (
                    attacker is not None
                    and victim is not None
                    and SEE_VALUE[attacker] > SEE_VALUE[victim]
                    and see(board, move) < 0
                ):
                    continue
            self.push(board, move)
            score = -self.quiesce(board, -beta, -alpha, ply + 1)
            self.pop(board)
            if score > best:
                best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
        return best

    # ----------------------------------------------------------------------------------
    # Main search
    # ----------------------------------------------------------------------------------

    def negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        can_null: bool = True,
        previous: chess.Move | None = None,
    ) -> int:
        self.nodes += 1
        if not self.nodes & 1023 and time.monotonic() > self.deadline:
            raise TimeUp

        alpha_original = alpha
        key = board._transposition_key()

        # A position we have already stood in, or one repeated inside this search line,
        # is a draw we should score as one.
        if ply and (key in self.path or SEEN.get(key, 0) >= 1):
            return 0
        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return 0

        tt_move: chess.Move | None = None
        entry = TT.get(key)
        if entry is not None:
            tt_depth, tt_score, tt_flag, tt_move = entry
            if tt_depth >= depth and ply:
                score = tt_score
                if score > MATE_THRESHOLD:
                    score -= ply
                elif score < -MATE_THRESHOLD:
                    score += ply
                if tt_flag == EXACT:
                    return score
                if tt_flag == LOWER and score > alpha:
                    alpha = score
                elif tt_flag == UPPER and score < beta:
                    beta = score
                if alpha >= beta:
                    return score

        in_check = board.is_check()
        if in_check:
            depth += 1  # Check extension: never evaluate a position that is on fire.

        if depth <= 0:
            return self.quiesce(board, alpha, beta, ply)

        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else 0

        is_pv = beta - alpha > 1
        static = self.evaluate(board) if not in_check else 0

        # Reverse futility: far enough above beta with a shallow depth left, take the cut.
        if not is_pv and not in_check and depth <= 3 and static - 120 * depth >= beta:
            return static


        # Null move pruning. Skipped in check, in pawn-only endings (zugzwang), and when
        # we are already in a null-move branch.
        if (
            can_null
            and not is_pv
            and not in_check
            and depth >= 3
            and static >= beta
            and board.occupied_co[board.turn] & ~(board.pawns | board.kings)
        ):
            self.push_null(board)
            self.path.append(key)
            score = -self.negamax(board, depth - 1 - NULL_MOVE_R, -beta, -beta + 1,
                                  ply + 1, can_null=False)
            self.path.pop()
            self.pop(board)
            if score >= beta:
                return beta

        best_score = -MATE - 1
        best_move: chess.Move | None = None
        self.path.append(key)
        try:
            counter = None
            if previous is not None:
                counter = COUNTER.get((previous.from_square, previous.to_square))
            for index, move in enumerate(self.order(board, moves, tt_move, ply, counter)):
                is_quiet = board.piece_type_at(move.to_square) is None and not move.promotion

                # Futility pruning on quiet moves near the leaves.
                if (
                    is_quiet
                    and not is_pv
                    and not in_check
                    and depth <= 3
                    and index > 0
                    and static + FUTILITY_MARGIN[depth] <= alpha
                ):
                    continue

                self.push(board, move)
                if index == 0:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1,
                                          previous=move)
                else:
                    # Conservative reductions. A log-scaled table is standard, but at the
                    # seven or eight plies reachable in Python it reduces branches that
                    # were never deep enough to spare them.
                    reduction = 0
                    if is_quiet and depth >= 3 and index >= 4 and not board.is_check():
                        reduction = 1 if index < 8 else 2
                        if is_pv and reduction:
                            reduction -= 1
                    score = -self.negamax(board, depth - 1 - reduction, -alpha - 1,
                                          -alpha, ply + 1, previous=move)
                    if reduction and score > alpha:
                        score = -self.negamax(board, depth - 1, -alpha - 1, -alpha,
                                              ply + 1, previous=move)
                    if alpha < score < beta:
                        score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1,
                                              previous=move)
                self.pop(board)

                if score > best_score:
                    best_score = score
                    best_move = move
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    if is_quiet:
                        killers = KILLERS[ply]
                        if move != killers[0]:
                            killers[1] = killers[0]
                            killers[0] = move
                        hist_key = (board.turn, move.from_square, move.to_square)
                        HISTORY[hist_key] = min(
                            HISTORY.get(hist_key, 0) + depth * depth, HISTORY_CAP
                        )
                        if previous is not None:
                            COUNTER[(previous.from_square, previous.to_square)] = move
                    break
        finally:
            self.path.pop()

        stored = best_score
        if stored > MATE_THRESHOLD:
            stored += ply
        elif stored < -MATE_THRESHOLD:
            stored -= ply
        if best_score <= alpha_original:
            flag = UPPER
        elif best_score >= beta:
            flag = LOWER
        else:
            flag = EXACT
        # Depth-preferred replacement. Refusing new entries once the table is full is
        # worse than forgetting, because what it keeps is whatever arrived first.
        existing = TT.get(key)
        if existing is None or depth >= existing[0] or flag == EXACT:
            TT[key] = (depth, stored, flag, best_move)

        return best_score

    # ----------------------------------------------------------------------------------
    # Root
    # ----------------------------------------------------------------------------------

    def search_root(self, board: chess.Board, depth: int, moves: list[chess.Move],
                    previous_best: chess.Move | None,
                    alpha: int = -MATE - 1, beta: int = MATE + 1) -> tuple[int, chess.Move]:
        """One pass at a fixed depth. Raises TimeUp if the budget runs out."""
        best_move = previous_best or moves[0]
        best_score = -MATE - 1
        key = board._transposition_key()
        self.path.append(key)
        try:
            for index, move in enumerate(self.order(board, moves, previous_best, 0)):
                self.push(board, move)
                if index == 0:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
                else:
                    score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                    if score > alpha:
                        score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
                self.pop(board)
                if score > best_score:
                    best_score = score
                    best_move = move
                if score > alpha:
                    alpha = score
        finally:
            self.path.pop()
        return best_score, best_move


# --------------------------------------------------------------------------------------
# Time management
# --------------------------------------------------------------------------------------

INCREMENT_MS = 500
SAFETY_MS = 300
PANIC_MS = 250


def budget_ms(board: chess.Board, time_left_ms: int) -> tuple[float, float]:
    """Return (soft, hard) budgets in milliseconds for this move.

    Soft is the point past which we do not start another iteration; hard is the point at
    which the search is abandoned and the best move so far is played. A flag is a loss,
    so both sit well inside the clock we were handed.
    """
    usable = max(0.0, time_left_ms - SAFETY_MS)
    pieces = chess.popcount(board.occupied)
    expected_moves = 22 if pieces > 20 else (28 if pieces > 10 else 18)
    soft = usable / expected_moves + INCREMENT_MS * 0.7
    soft = min(soft, usable * 0.25)
    hard = min(soft * 3.0, usable * 0.45)
    return max(soft, 10.0), max(hard, 20.0)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def _python_get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in `fen`."""
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if not moves:
        return "0000"

    fallback = moves[0].uci()
    try:
        key = board._transposition_key()
        SEEN[key] = SEEN.get(key, 0) + 1

        if len(moves) == 1:
            return fallback

        opening = BOOK.get(" ".join(fen.split()[:4]))
        if opening:
            playable = [
                move for move in (chess.Move.from_uci(uci) for uci in opening)
                if move in moves
            ]
            if playable:
                return random.choice(playable).uci()

        # Under a second on the clock, do not think: a flag loses the game outright.
        if time_left_ms <= PANIC_MS:
            searcher = Searcher(time.monotonic() + 0.05)
            ordered = searcher.order(board, moves, None, 0)
            return ordered[0].uci()

        start = time.monotonic()
        soft, hard = budget_ms(board, time_left_ms)
        searcher = Searcher(start + hard / 1000.0)
        searcher.seed(board)

        best_move = moves[0]
        previous_best = best_move
        score = 0
        for depth in range(1, MAX_PLY):
            # Aspiration windows: re-searching a narrow window around the last score is
            # cheaper than a full one, and the occasional re-search costs less than the
            # nodes it saves everywhere else.
            window = ASPIRATION_WINDOW
            if depth < 4:
                alpha, beta = -MATE - 1, MATE + 1
            else:
                alpha, beta = score - window, score + window
            try:
                while True:
                    value, move = searcher.search_root(
                        board, depth, moves, best_move, alpha, beta
                    )
                    if alpha < value < beta:
                        score, best_move = value, move
                        break
                    if value <= alpha:
                        alpha = max(-MATE - 1, alpha - window * 3)
                    else:
                        beta = min(MATE + 1, beta + window * 3)
                        best_move = move
                    window *= 2
            except TimeUp:
                break
            elapsed_ms = (time.monotonic() - start) * 1000.0
            # Only start another iteration if there is a realistic chance of finishing it,
            # but give an unsettled position more rope: a best move that is still changing
            # between iterations is exactly where an extra ply pays for itself.
            unstable = best_move != previous_best
            previous_best = best_move
            if elapsed_ms > soft * (1.0 if unstable else 0.6):
                break

        if len(TT) > TT_MAX:
            TT.clear()
            COUNTER.clear()
        return best_move.uci()
    except Exception as exc:  # never lose a game to an unhandled error
        # The output cap is 4096 bytes per move and passing it loses the game.
        # A numba TypingError repr alone runs to several kilobytes, so this is
        # truncated rather than trusted.
        print(f"search failed, playing fallback: {exc!r}"[:200])
        return fallback


# ======================================================================================
# JIT engine
#
# The same search as above, rewritten so numba can compile it to machine code: plain
# numpy arrays and scalars, no objects, no python-chess in the hot path. It searches
# roughly thirty times as many positions per second, which is worth about three extra
# plies of depth.
#
# numba has no clock in nopython mode, so this search runs to a node budget and the
# driver converts the remaining time into one. That is stricter than a deadline check,
# not looser: an iteration cannot overrun its allowance.
#
# Everything here is optional. If numba is missing, incompatible with the installed
# numpy, or still compiling, the driver falls back to the pure-Python engine above.
# ======================================================================================

JIT_READY = False
JIT_ERROR = ""

try:
    import numpy as _np
    from numba import njit as _njit

    njit = _njit
    HAVE_NUMBA = True
except Exception as _exc:  # numba absent or mismatched with numpy
    HAVE_NUMBA = False
    JIT_ERROR = repr(_exc)

if HAVE_NUMBA:
    JIT_MG_TABLE = _np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 100, 100, 100, 100, 100, 100, 148, 143,
        142, 162, 152, 88, 41, 86, 124, 115, 141, 156, 163, 177, 121, 148, 96, 124, 121, 133,
        135, 130, 131, 99, 83, 92, 104, 112, 116, 99, 95, 84, 83, 93, 97, 71, 80, 89, 126, 82,
        60, 91, 82, 50, 54, 117, 115, 68, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 60, 91, 82, 50, 54, 117, 115, 68, 83, 93, 97, 71, 80, 89,
        126, 82, 83, 92, 104, 112, 116, 99, 95, 84, 96, 124, 121, 133, 135, 130, 131, 99, 124,
        115, 141, 156, 163, 177, 121, 148, 148, 143, 142, 162, 152, 88, 41, 86, 100, 100, 100,
        100, 100, 100, 100, 100, 253, 285, 304, 431, 439, 195, 362, 386, 399, 369, 487, 464,
        409, 501, 359, 456, 396, 492, 484, 523, 564, 537, 516, 420, 455, 445, 468, 489, 481,
        498, 474, 466, 432, 410, 447, 445, 453, 463, 454, 440, 391, 424, 435, 439, 450, 444,
        450, 410, 362, 388, 407, 407, 419, 434, 410, 392, 318, 378, 361, 407, 403, 395, 388,
        356, 318, 378, 361, 407, 403, 395, 388, 356, 362, 388, 407, 407, 419, 434, 410, 392,
        391, 424, 435, 439, 450, 444, 450, 410, 432, 410, 447, 445, 453, 463, 454, 440, 455,
        445, 468, 489, 481, 498, 474, 466, 396, 492, 484, 523, 564, 537, 516, 420, 399, 369,
        487, 464, 409, 501, 359, 456, 253, 285, 304, 431, 439, 195, 362, 386, 346, 328, 346,
        320, 286, 238, 365, 385, 377, 418, 447, 412, 400, 466, 407, 435, 412, 469, 443, 526,
        476, 510, 493, 450, 433, 438, 437, 472, 485, 468, 451, 444, 456, 426, 445, 453, 462,
        446, 416, 451, 429, 454, 447, 441, 438, 447, 461, 422, 455, 449, 436, 418, 433, 431,
        447, 426, 422, 425, 398, 396, 407, 394, 402, 395, 422, 425, 398, 396, 407, 394, 402,
        395, 455, 449, 436, 418, 433, 431, 447, 426, 429, 454, 447, 441, 438, 447, 461, 422,
        456, 426, 445, 453, 462, 446, 416, 451, 433, 438, 437, 472, 485, 468, 451, 444, 412,
        469, 443, 526, 476, 510, 493, 450, 377, 418, 447, 412, 400, 466, 407, 435, 346, 328,
        346, 320, 286, 238, 365, 385, 567, 572, 602, 650, 601, 628, 597, 608, 579, 574, 622,
        658, 655, 652, 612, 627, 549, 586, 607, 652, 649, 637, 645, 576, 538, 540, 560, 580,
        589, 604, 572, 566, 520, 526, 541, 537, 549, 543, 532, 527, 505, 533, 522, 527, 549,
        544, 544, 533, 497, 513, 527, 530, 535, 551, 520, 495, 520, 528, 535, 548, 557, 550,
        527, 528, 520, 528, 535, 548, 557, 550, 527, 528, 497, 513, 527, 530, 535, 551, 520,
        495, 505, 533, 522, 527, 549, 544, 544, 533, 520, 526, 541, 537, 549, 543, 532, 527,
        538, 540, 560, 580, 589, 604, 572, 566, 549, 586, 607, 652, 649, 637, 645, 576, 579,
        574, 622, 658, 655, 652, 612, 627, 567, 572, 602, 650, 601, 628, 597, 608, 1047, 1118,
        1189, 1232, 1182, 1234, 1160, 1086, 1098, 1038, 1107, 1083, 1113, 1161, 1147, 1119,
        1065, 1082, 1103, 1137, 1141, 1187, 1167, 1125, 1073, 1078, 1078, 1094, 1092, 1098,
        1078, 1114, 1086, 1101, 1081, 1079, 1091, 1075, 1089, 1085, 1072, 1089, 1089, 1088,
        1073, 1090, 1095, 1095, 1089, 1095, 1103, 1095, 1093, 1101, 1104, 1094, 1085, 1087,
        1073, 1077, 1072, 1077, 1108, 1072, 1085, 1087, 1073, 1077, 1072, 1077, 1108, 1072,
        1089, 1095, 1103, 1095, 1093, 1101, 1104, 1094, 1072, 1089, 1089, 1088, 1073, 1090,
        1095, 1095, 1086, 1101, 1081, 1079, 1091, 1075, 1089, 1085, 1073, 1078, 1078, 1094,
        1092, 1098, 1078, 1114, 1065, 1082, 1103, 1137, 1141, 1187, 1167, 1125, 1098, 1038,
        1107, 1083, 1113, 1161, 1147, 1119, 1047, 1118, 1189, 1232, 1182, 1234, 1160, 1086, 94,
        189, 164, 160, 102, 241, 183, 159, 121, 120, 19, 23, 59, 47, 100, 94, 6, 9, -62, -124,
        -140, -53, -14, 26, -113, -45, -101, -175, -172, -118, -36, -74, -105, -82, -138, -57,
        -122, -123, -81, -78, -39, -12, -63, -86, -100, -79, -30, -55, 31, 30, -14, -54, -67,
        -19, 6, 35, 17, 77, 56, -35, 13, -21, 58, 57, 17, 77, 56, -35, 13, -21, 58, 57, 31, 30,
        -14, -54, -67, -19, 6, 35, -39, -12, -63, -86, -100, -79, -30, -55, -105, -82, -138,
        -57, -122, -123, -81, -78, -113, -45, -101, -175, -172, -118, -36, -74, 6, 9, -62,
        -124, -140, -53, -14, 26, 121, 120, 19, 23, 59, 47, 100, 94, 94, 189, 164, 160, 102,
        241, 183, 159
    ], dtype=_np.int32)
    JIT_EG_TABLE = _np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 100, 100, 100, 100, 100, 100, 163, 186,
        177, 169, 204, 197, 203, 172, 102, 118, 104, 80, 79, 100, 117, 85, 101, 104, 104, 96,
        92, 106, 102, 93, 100, 104, 92, 87, 91, 100, 106, 100, 99, 108, 98, 101, 102, 115, 98,
        92, 109, 115, 128, 135, 124, 123, 119, 106, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100, 109, 115, 128, 135, 124, 123, 119, 106, 99,
        108, 98, 101, 102, 115, 98, 92, 100, 104, 92, 87, 91, 100, 106, 100, 101, 104, 104, 96,
        92, 106, 102, 93, 102, 118, 104, 80, 79, 100, 117, 85, 163, 186, 177, 169, 204, 197,
        203, 172, 100, 100, 100, 100, 100, 100, 100, 100, 105, 205, 241, 184, 199, 254, 184,
        74, 196, 226, 197, 231, 239, 204, 229, 209, 215, 226, 249, 243, 218, 238, 212, 220,
        231, 261, 255, 260, 266, 255, 254, 233, 234, 248, 266, 268, 275, 258, 241, 227, 241,
        259, 254, 266, 272, 255, 255, 226, 219, 257, 245, 264, 259, 245, 234, 224, 246, 214,
        261, 241, 252, 244, 228, 180, 246, 214, 261, 241, 252, 244, 228, 180, 219, 257, 245,
        264, 259, 245, 234, 224, 241, 259, 254, 266, 272, 255, 255, 226, 234, 248, 266, 268,
        275, 258, 241, 227, 231, 261, 255, 260, 266, 255, 254, 233, 215, 226, 249, 243, 218,
        238, 212, 220, 196, 226, 197, 231, 239, 204, 229, 209, 105, 205, 241, 184, 199, 254,
        184, 74, 278, 270, 283, 269, 270, 295, 269, 273, 270, 266, 256, 268, 269, 247, 264,
        233, 269, 254, 279, 255, 262, 257, 244, 255, 253, 269, 272, 278, 254, 250, 276, 245,
        232, 265, 285, 272, 272, 274, 262, 232, 265, 267, 271, 279, 278, 273, 252, 262, 250,
        251, 264, 268, 269, 266, 264, 267, 278, 264, 249, 262, 258, 268, 252, 273, 278, 264,
        249, 262, 258, 268, 252, 273, 250, 251, 264, 268, 269, 266, 264, 267, 265, 267, 271,
        279, 278, 273, 252, 262, 232, 265, 285, 272, 272, 274, 262, 232, 253, 269, 272, 278,
        254, 250, 276, 245, 269, 254, 279, 255, 262, 257, 244, 255, 270, 266, 256, 268, 269,
        247, 264, 233, 278, 270, 283, 269, 270, 295, 269, 273, 453, 450, 443, 426, 435, 436,
        443, 444, 447, 450, 439, 430, 427, 421, 435, 425, 446, 433, 427, 410, 405, 415, 409,
        434, 452, 457, 444, 440, 427, 417, 426, 427, 456, 444, 454, 452, 442, 441, 433, 450,
        450, 449, 457, 448, 438, 445, 438, 441, 451, 447, 435, 445, 443, 433, 434, 445, 448,
        450, 447, 440, 441, 440, 454, 426, 448, 450, 447, 440, 441, 440, 454, 426, 451, 447,
        435, 445, 443, 433, 434, 445, 450, 449, 457, 448, 438, 445, 438, 441, 456, 444, 454,
        452, 442, 441, 433, 450, 452, 457, 444, 440, 427, 417, 426, 427, 446, 433, 427, 410,
        405, 415, 409, 434, 447, 450, 439, 430, 427, 421, 435, 425, 453, 450, 443, 426, 435,
        436, 443, 444, 770, 750, 719, 736, 779, 706, 727, 768, 782, 836, 823, 858, 846, 820,
        783, 799, 805, 792, 825, 831, 845, 819, 816, 803, 775, 805, 849, 866, 869, 875, 860,
        805, 737, 754, 825, 853, 822, 835, 793, 795, 751, 752, 785, 788, 813, 784, 768, 755,
        724, 717, 734, 748, 733, 737, 671, 703, 740, 713, 714, 731, 710, 666, 613, 717, 740,
        713, 714, 731, 710, 666, 613, 717, 724, 717, 734, 748, 733, 737, 671, 703, 751, 752,
        785, 788, 813, 784, 768, 755, 737, 754, 825, 853, 822, 835, 793, 795, 775, 805, 849,
        866, 869, 875, 860, 805, 805, 792, 825, 831, 845, 819, 816, 803, 782, 836, 823, 858,
        846, 820, 783, 799, 770, 750, 719, 736, 779, 706, 727, 768, -66, -39, -9, -27, -7, -64,
        -61, -97, -26, 5, 46, 34, 22, 34, -4, -31, -2, 40, 62, 57, 63, 52, 36, -24, 9, 30, 48,
        58, 53, 48, 17, -16, -36, 24, 34, 19, 33, 28, 20, -26, -19, -11, 14, 15, 22, 16, -2,
        -20, -32, -18, -3, 2, 11, 4, -8, -33, -16, -47, -46, -39, -51, -24, -44, -43, -16, -47,
        -46, -39, -51, -24, -44, -43, -32, -18, -3, 2, 11, 4, -8, -33, -19, -11, 14, 15, 22,
        16, -2, -20, -36, 24, 34, 19, 33, 28, 20, -26, 9, 30, 48, 58, 53, 48, 17, -16, -2, 40,
        62, 57, 63, 52, 36, -24, -26, 5, 46, 34, 22, 34, -4, -31, -66, -39, -9, -27, -7, -64,
        -61, -97
    ], dtype=_np.int32)
    JIT_PHASE_WEIGHT = _np.array([
        0, 0, 1, 1, 2, 4, 0
    ], dtype=_np.int32)
    JIT_CENTER_DISTANCE = _np.array([
        6, 5, 4, 3, 3, 4, 5, 6, 5, 4, 3, 2, 2, 3, 4, 5, 4, 3, 2, 1, 1, 2, 3, 4, 3, 2, 1, 0, 0,
        1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 1, 2, 3, 4, 5, 4, 3, 2, 2, 3, 4, 5, 6, 5,
        4, 3, 3, 4, 5, 6
    ], dtype=_np.int32)
    JIT_PASSED_MG = _np.array([
        0, -5, -30, -23, -4, 15, 27, 0
    ], dtype=_np.int32)
    JIT_PASSED_EG = _np.array([
        0, 8, 26, 43, 52, 89, 110, 0
    ], dtype=_np.int32)
    JIT_BISHOP_PAIR_MG = 56
    JIT_BISHOP_PAIR_EG = 44
    JIT_ROOK_OPEN = 54
    JIT_ROOK_SEMI_OPEN = 35
    JIT_TEMPO = 12
    JIT_TOTAL_PHASE = 24
    JIT_MOP_UP = 450
    JIT_ISOLATED_MG = 16
    JIT_ISOLATED_EG = 7
    JIT_DOUBLED_MG = 7
    JIT_DOUBLED_EG = 25
    JIT_SHELTER = 35




    # Step tables. numba freezes module-level arrays as compile-time constants, which is
    # exactly what we want for these.
    KNIGHT_STEPS = _np.array([33, 31, 18, 14, -33, -31, -18, -14], dtype=_np.int64)
    BISHOP_STEPS = _np.array([15, 17, -15, -17], dtype=_np.int64)
    ROOK_STEPS = _np.array([1, 16, -1, -16], dtype=_np.int64)
    KING_STEPS = _np.array([1, 16, -1, -16, 15, 17, -15, -17], dtype=_np.int64)

    TURN, CASTLING, EP, HALFMOVE, KING_W, KING_B, HASH = 0, 1, 2, 3, 4, 5, 6
    STATE_SLOTS = 8
    HIST_STRIDE = 6

    # Zobrist keys, fixed seed so a position always hashes the same way. The hash is kept up
    # to date inside jit_make_move rather than recomputed, which was costing a scan of the board
    # at every node.
    _rng = _np.random.default_rng(0x5EED)
    Z_PIECE = _rng.integers(-(2**62), 2**62, size=16 * 128, dtype=_np.int64)
    Z_CASTLE = _rng.integers(-(2**62), 2**62, size=16, dtype=_np.int64)
    Z_EP = _rng.integers(-(2**62), 2**62, size=128, dtype=_np.int64)
    Z_SIDE = _np.int64(_rng.integers(-(2**62), 2**62, dtype=_np.int64))

    WK, WQ, BK, BQ = 1, 2, 4, 8

    MOVE_EP = 1 << 24
    MOVE_CASTLE = 1 << 25
    MOVE_DOUBLE = 1 << 26

    JIT_MAX_PLY = 96
    JIT_MOVE_SLOTS = 256

    JIT = {"nopython": True, "cache": False, "fastmath": False}


    @njit(inline="always")
    def jit_on_board(square: int) -> bool:
        return (square & 0x88) == 0


    @njit(inline="always")
    def jit_encode(from_sq: int, to_sq: int, promo: int, captured: int, flags: int) -> int:
        return from_sq | (to_sq << 8) | (promo << 16) | (captured << 19) | flags


    @njit(inline="always")
    def jit_move_from(move: int) -> int:
        return move & 0xFF


    @njit(inline="always")
    def jit_move_to(move: int) -> int:
        return (move >> 8) & 0xFF


    @njit(inline="always")
    def jit_move_promo(move: int) -> int:
        return (move >> 16) & 7


    @njit(inline="always")
    def jit_move_captured(move: int) -> int:
        return (move >> 19) & 31


    @njit(cache=False)
    def jit_is_attacked(board: Any, square: Any, by_colour: Any) -> Any:
        """Is `square` attacked by any piece of `by_colour`?"""
        if by_colour == 0:
            target = 1  # white pawn
            a = square - 15
            b = square - 17
        else:
            target = 9  # black pawn
            a = square + 15
            b = square + 17
        if jit_on_board(a) and board[a] == target:
            return True
        if jit_on_board(b) and board[b] == target:
            return True

        knight = 2 | (by_colour << 3)
        king = 6 | (by_colour << 3)
        for i in range(8):
            sq = square + KNIGHT_STEPS[i]
            if jit_on_board(sq) and board[sq] == knight:
                return True
            sq = square + KING_STEPS[i]
            if jit_on_board(sq) and board[sq] == king:
                return True

        for i in range(4):
            step = BISHOP_STEPS[i]
            sq = square + step
            while jit_on_board(sq):
                piece = board[sq]
                if piece != 0:
                    kind = piece & 7
                    if (piece >> 3) == by_colour and (kind == 3 or kind == 5):
                        return True
                    break
                sq += step
            step = ROOK_STEPS[i]
            sq = square + step
            while jit_on_board(sq):
                piece = board[sq]
                if piece != 0:
                    kind = piece & 7
                    if (piece >> 3) == by_colour and (kind == 4 or kind == 5):
                        return True
                    break
                sq += step
        return False


    @njit(cache=False)
    def jit_in_check(board: Any, st: Any, colour: Any) -> Any:
        king_sq = st[KING_W] if colour == 0 else st[KING_B]
        return jit_is_attacked(board, king_sq, 1 - colour)


    @njit(cache=False)
    def jit_generate(board: Any, st: Any, moves: Any, base: Any, captures_only: Any) -> Any:
        """Pseudo-legal moves into moves[base:], returning how many were written."""
        count = 0
        us = st[TURN]
        them = 1 - us
        forward = 16 if us == 0 else -16
        start_rank = 1 if us == 0 else 6
        last_rank = 7 if us == 0 else 0
        ep = st[EP]

        for sq in range(128):
            if (sq & 0x88) != 0:
                continue
            piece = board[sq]
            if piece == 0 or (piece >> 3) != us:
                continue
            kind = piece & 7

            if kind == 1:
                one = sq + forward
                if jit_on_board(one) and board[one] == 0:
                    if (one >> 4) == last_rank:
                        for promo in (5, 4, 3, 2):
                            moves[base + count] = jit_encode(sq, one, promo, 0, 0)
                            count += 1
                    elif not captures_only:
                        moves[base + count] = jit_encode(sq, one, 0, 0, 0)
                        count += 1
                        two = one + forward
                        if (sq >> 4) == start_rank and board[two] == 0:
                            moves[base + count] = jit_encode(sq, two, 0, 0, MOVE_DOUBLE)
                            count += 1
                for side in (-1, 1):
                    target = sq + forward + side
                    if not jit_on_board(target):
                        continue
                    victim = board[target]
                    if victim != 0 and (victim >> 3) == them:
                        if (target >> 4) == last_rank:
                            for promo in (5, 4, 3, 2):
                                moves[base + count] = jit_encode(sq, target, promo, victim, 0)
                                count += 1
                        else:
                            moves[base + count] = jit_encode(sq, target, 0, victim, 0)
                            count += 1
                    elif victim == 0 and target == ep:
                        moves[base + count] = jit_encode(sq, target, 0, 1 | (them << 3), MOVE_EP)
                        count += 1
                continue

            if kind == 2 or kind == 6:
                for i in range(8):
                    step = KNIGHT_STEPS[i] if kind == 2 else KING_STEPS[i]
                    target = sq + step
                    if not jit_on_board(target):
                        continue
                    victim = board[target]
                    if victim != 0 and (victim >> 3) == us:
                        continue
                    if captures_only and victim == 0:
                        continue
                    moves[base + count] = jit_encode(sq, target, 0, victim, 0)
                    count += 1
                continue

            directions = 4 if kind != 5 else 8
            for i in range(directions):
                if kind == 3:
                    step = BISHOP_STEPS[i]
                elif kind == 4:
                    step = ROOK_STEPS[i]
                else:
                    step = KING_STEPS[i]
                target = sq + step
                while jit_on_board(target):
                    victim = board[target]
                    if victim != 0:
                        if (victim >> 3) != us:
                            moves[base + count] = jit_encode(sq, target, 0, victim, 0)
                            count += 1
                        break
                    if not captures_only:
                        moves[base + count] = jit_encode(sq, target, 0, 0, 0)
                        count += 1
                    target += step

        if not captures_only:
            rights = st[CASTLING]
            if us == 0:
                king_sq = st[KING_W]
                if ((rights & WK) != 0 and board[0x05] == 0 and board[0x06] == 0
                        and not jit_is_attacked(board, 0x04, 1)
                        and not jit_is_attacked(board, 0x05, 1)
                        and not jit_is_attacked(board, 0x06, 1)):
                    moves[base + count] = jit_encode(king_sq, 0x06, 0, 0, MOVE_CASTLE)
                    count += 1
                if ((rights & WQ) != 0 and board[0x03] == 0 and board[0x02] == 0
                        and board[0x01] == 0
                        and not jit_is_attacked(board, 0x04, 1)
                        and not jit_is_attacked(board, 0x03, 1)
                        and not jit_is_attacked(board, 0x02, 1)):
                    moves[base + count] = jit_encode(king_sq, 0x02, 0, 0, MOVE_CASTLE)
                    count += 1
            else:
                king_sq = st[KING_B]
                if ((rights & BK) != 0 and board[0x75] == 0 and board[0x76] == 0
                        and not jit_is_attacked(board, 0x74, 0)
                        and not jit_is_attacked(board, 0x75, 0)
                        and not jit_is_attacked(board, 0x76, 0)):
                    moves[base + count] = jit_encode(king_sq, 0x76, 0, 0, MOVE_CASTLE)
                    count += 1
                if ((rights & BQ) != 0 and board[0x73] == 0 and board[0x72] == 0
                        and board[0x71] == 0
                        and not jit_is_attacked(board, 0x74, 0)
                        and not jit_is_attacked(board, 0x73, 0)
                        and not jit_is_attacked(board, 0x72, 0)):
                    moves[base + count] = jit_encode(king_sq, 0x72, 0, 0, MOVE_CASTLE)
                    count += 1
        return count


    @njit(cache=False)
    def jit_full_hash(board: Any, st: Any) -> Any:
        key = _np.int64(0)
        for sq in range(128):
            if (sq & 0x88) != 0:
                continue
            piece = board[sq]
            if piece != 0:
                key ^= Z_PIECE[piece * 128 + sq]
        key ^= Z_CASTLE[st[CASTLING]]
        if st[EP] >= 0:
            key ^= Z_EP[st[EP]]
        if st[TURN] == 1:
            key ^= Z_SIDE
        return key


    @njit(cache=False)
    def jit_make_move(board: Any, st: Any, hist: Any, move: Any, ply: Any) -> Any:
        """Play `move`; returns False and restores everything if it leaves us in check."""
        slot = ply * HIST_STRIDE
        hist[slot] = st[CASTLING]
        hist[slot + 1] = st[EP]
        hist[slot + 2] = st[HALFMOVE]
        hist[slot + 3] = st[KING_W]
        hist[slot + 4] = st[KING_B]
        hist[slot + 5] = st[HASH]

        key = st[HASH]
        key ^= Z_CASTLE[st[CASTLING]]
        if st[EP] >= 0:
            key ^= Z_EP[st[EP]]

        us = st[TURN]
        them = 1 - us
        from_sq = jit_move_from(move)
        to_sq = jit_move_to(move)
        piece = board[from_sq]
        promo = jit_move_promo(move)
        captured = jit_move_captured(move)
        kind = piece & 7

        board[from_sq] = 0
        key ^= Z_PIECE[piece * 128 + from_sq]
        if (move & MOVE_EP) != 0:
            cap_sq = to_sq - 16 if us == 0 else to_sq + 16
            key ^= Z_PIECE[board[cap_sq] * 128 + cap_sq]
            board[cap_sq] = 0
        elif captured != 0:
            key ^= Z_PIECE[captured * 128 + to_sq]
        placed = (promo | (us << 3)) if promo != 0 else piece
        board[to_sq] = placed
        key ^= Z_PIECE[placed * 128 + to_sq]

        if kind == 6:
            if us == 0:
                st[KING_W] = to_sq
            else:
                st[KING_B] = to_sq
            if (move & MOVE_CASTLE) != 0:
                if (to_sq & 7) == 6:
                    rook_from = to_sq + 1
                    rook_to = to_sq - 1
                else:
                    rook_from = to_sq - 2
                    rook_to = to_sq + 1
                rook = board[rook_from]
                board[rook_to] = rook
                board[rook_from] = 0
                key ^= Z_PIECE[rook * 128 + rook_from] ^ Z_PIECE[rook * 128 + rook_to]

        rights = st[CASTLING]
        if rights != 0:
            if from_sq == 0x04:
                rights &= ~(WK | WQ)
            if from_sq == 0x74:
                rights &= ~(BK | BQ)
            if from_sq == 0x07 or to_sq == 0x07:
                rights &= ~WK
            if from_sq == 0x00 or to_sq == 0x00:
                rights &= ~WQ
            if from_sq == 0x77 or to_sq == 0x77:
                rights &= ~BK
            if from_sq == 0x70 or to_sq == 0x70:
                rights &= ~BQ
            st[CASTLING] = rights

        if (move & MOVE_DOUBLE) != 0:
            st[EP] = from_sq + 16 if us == 0 else from_sq - 16
        else:
            st[EP] = -1
        st[HALFMOVE] = 0 if (kind == 1 or captured != 0) else st[HALFMOVE] + 1
        st[TURN] = them

        key ^= Z_CASTLE[st[CASTLING]]
        if st[EP] >= 0:
            key ^= Z_EP[st[EP]]
        key ^= Z_SIDE
        st[HASH] = key

        king_sq = st[KING_W] if us == 0 else st[KING_B]
        if jit_is_attacked(board, king_sq, them):
            jit_unmake_move(board, st, hist, move, ply)
            return False
        return True


    @njit(cache=False)
    def jit_unmake_move(board: Any, st: Any, hist: Any, move: Any, ply: Any) -> Any:
        slot = ply * HIST_STRIDE
        st[CASTLING] = hist[slot]
        st[EP] = hist[slot + 1]
        st[HALFMOVE] = hist[slot + 2]
        st[KING_W] = hist[slot + 3]
        st[KING_B] = hist[slot + 4]
        st[HASH] = hist[slot + 5]
        st[TURN] = 1 - st[TURN]

        us = st[TURN]
        them = 1 - us
        from_sq = jit_move_from(move)
        to_sq = jit_move_to(move)
        promo = jit_move_promo(move)
        captured = jit_move_captured(move)

        placed = board[to_sq]
        board[from_sq] = (1 | (us << 3)) if promo != 0 else placed
        board[to_sq] = 0

        if (move & MOVE_EP) != 0:
            board[to_sq - 16 if us == 0 else to_sq + 16] = 1 | (them << 3)
        elif captured != 0:
            board[to_sq] = captured

        if (move & MOVE_CASTLE) != 0:
            if (to_sq & 7) == 6:
                board[to_sq + 1] = board[to_sq - 1]
                board[to_sq - 1] = 0
            else:
                board[to_sq - 2] = board[to_sq + 1]
                board[to_sq + 1] = 0


    @njit(cache=False)
    def jit_perft(board: Any, st: Any, hist: Any, moves: Any, depth: Any, ply: Any) -> Any:
        if depth == 0:
            return 1
        base = ply * JIT_MOVE_SLOTS
        count = jit_generate(board, st, moves, base, False)
        total = 0
        for i in range(count):
            move = moves[base + i]
            if jit_make_move(board, st, hist, move, ply):
                total += 1 if depth == 1 else jit_perft(board, st, hist, moves, depth - 1, ply + 1)
                jit_unmake_move(board, st, hist, move, ply)
        return total


    # ------------------------------------------------------------------ python-side setup
    PIECE_FROM_LETTER = {"p": 1, "n": 2, "b": 3, "r": 4, "q": 5, "k": 6}


    def jit_new_state(fen: str) -> Any:
        board = _np.zeros(128, dtype=_np.int8)
        st = _np.zeros(STATE_SLOTS, dtype=_np.int64)
        parts = fen.split()
        rank, file = 7, 0
        for ch in parts[0]:
            if ch == "/":
                rank -= 1
                file = 0
            elif ch.isdigit():
                file += int(ch)
            else:
                kind = PIECE_FROM_LETTER[ch.lower()]
                colour = 1 if ch.islower() else 0
                square = rank * 16 + file
                board[square] = kind | (colour << 3)
                if kind == 6:
                    st[KING_W if colour == 0 else KING_B] = square
                file += 1
        st[TURN] = 0 if parts[1] == "w" else 1
        rights = 0
        if len(parts) > 2 and parts[2] != "-":
            for ch, bit in (("K", WK), ("Q", WQ), ("k", BK), ("q", BQ)):
                if ch in parts[2]:
                    rights |= bit
        st[CASTLING] = rights
        if len(parts) > 3 and parts[3] != "-":
            st[EP] = "abcdefgh".index(parts[3][0]) + (int(parts[3][1]) - 1) * 16
        else:
            st[EP] = -1
        st[HALFMOVE] = int(parts[4]) if len(parts) > 4 else 0
        st[HASH] = jit_full_hash(board, st)
        return board, st


    def jit_scratch() -> Any:
        hist = _np.zeros(JIT_MAX_PLY * HIST_STRIDE, dtype=_np.int64)
        moves = _np.zeros(JIT_MAX_PLY * JIT_MOVE_SLOTS, dtype=_np.int32)
        return hist, moves


    def jit_square_name(square: int) -> str:
        return "abcdefgh"[square & 7] + str((square >> 4) + 1)


    def jit_move_uci(move: int) -> str:
        promo = (move >> 16) & 7
        return (jit_square_name(move & 0xFF) + jit_square_name((move >> 8) & 0xFF) +
                (" nbrq"[promo - 1] if promo else ""))







    JIT_MATE = 30000
    JIT_MATE_THRESHOLD = JIT_MATE - 1000

    JIT_NULL_R = 2
    JIT_DELTA = 200
    JIT_ASPIRATION = 40

    JIT_MVV_LVA = _np.array([0, 100, 320, 330, 500, 900, 0], dtype=_np.int32)
    JIT_SEE_VALUE = _np.array([0, 100, 320, 330, 500, 900, 20000], dtype=_np.int32)
    JIT_FUTILITY = _np.array([0, 150, 320, 520], dtype=_np.int32)
    JIT_LMR = _np.zeros(32 * 64, dtype=_np.int32)
    for _d in range(3, 32):
        for _i in range(4, 64):
            JIT_LMR[_d * 64 + _i] = int(0.75 + _np.log(_d) * _np.log(_i) / 2.25)

    TT_BITS = 21
    TT_SIZE = 1 << TT_BITS
    TT_MASK = TT_SIZE - 1

    # info[] slots
    NODES, NODE_LIMIT, STOPPED, REP_LEN = 0, 1, 2, 3


    @njit(cache=False)
    def jit_evaluate(board: Any, st: Any) -> Any:
        """Tapered material and piece-square score, plus the structural terms, in centipawns
        from the side to move's point of view."""
        mg = 0
        eg = 0
        phase = 0
        white_bishops = 0
        black_bishops = 0
        white_pawn_files = _np.zeros(8, dtype=_np.int32)
        black_pawn_files = _np.zeros(8, dtype=_np.int32)
        # Furthest-advanced and furthest-back pawn on each file. These four little arrays are
        # all the pawn structure terms need, which turns passed-pawn detection from a scan
        # over the board per pawn into a couple of comparisons.
        white_top = _np.full(8, -1, dtype=_np.int32)
        black_bottom = _np.full(8, 8, dtype=_np.int32)
        black_top = _np.full(8, -1, dtype=_np.int32)
        white_bottom = _np.full(8, 8, dtype=_np.int32)
        white_king = -1
        black_king = -1
        white_pawn_count = 0
        black_pawn_count = 0

        for sq in range(128):
            if (sq & 0x88) != 0:
                continue
            piece = board[sq]
            if piece == 0:
                continue
            kind = piece & 7
            colour = piece >> 3
            index = (sq >> 4) * 8 + (sq & 7)
            if colour == 0:
                mg += JIT_MG_TABLE[(kind * 2 + 1) * 64 + index]
                eg += JIT_EG_TABLE[(kind * 2 + 1) * 64 + index]
            else:
                mg -= JIT_MG_TABLE[(kind * 2) * 64 + index]
                eg -= JIT_EG_TABLE[(kind * 2) * 64 + index]
            phase += JIT_PHASE_WEIGHT[kind]
            file = sq & 7
            rank = sq >> 4
            if kind == 1:
                if colour == 0:
                    white_pawn_files[file] += 1
                    white_pawn_count += 1
                    if rank > white_top[file]:
                        white_top[file] = rank
                    if rank < white_bottom[file]:
                        white_bottom[file] = rank
                else:
                    black_pawn_files[file] += 1
                    black_pawn_count += 1
                    if rank > black_top[file]:
                        black_top[file] = rank
                    if rank < black_bottom[file]:
                        black_bottom[file] = rank
            elif kind == 3:
                if colour == 0:
                    white_bishops += 1
                else:
                    black_bishops += 1
            elif kind == 6:
                if colour == 0:
                    white_king = sq
                else:
                    black_king = sq

        if white_bishops > 1:
            mg += JIT_BISHOP_PAIR_MG
            eg += JIT_BISHOP_PAIR_EG
        if black_bishops > 1:
            mg -= JIT_BISHOP_PAIR_MG
            eg -= JIT_BISHOP_PAIR_EG

        for sq in range(128):
            if (sq & 0x88) != 0:
                continue
            piece = board[sq]
            if piece == 0:
                continue
            kind = piece & 7
            colour = piece >> 3
            file = sq & 7
            rank = sq >> 4
            if kind == 4:
                if colour == 0:
                    if white_pawn_files[file] == 0:
                        mg += JIT_ROOK_SEMI_OPEN if black_pawn_files[file] != 0 else JIT_ROOK_OPEN
                else:
                    if black_pawn_files[file] == 0:
                        mg -= JIT_ROOK_SEMI_OPEN if white_pawn_files[file] != 0 else JIT_ROOK_OPEN
            elif kind == 1:
                low = file - 1 if file > 0 else 0
                high = file + 1 if file < 7 else 7
                if colour == 0:
                    blocked = False
                    for f in range(low, high + 1):
                        if black_top[f] > rank:
                            blocked = True
                    if not blocked:
                        mg += JIT_PASSED_MG[rank]
                        eg += JIT_PASSED_EG[rank]
                    left = white_pawn_files[file - 1] if file > 0 else 0
                    right = white_pawn_files[file + 1] if file < 7 else 0
                    if left == 0 and right == 0:
                        mg -= JIT_ISOLATED_MG
                        eg -= JIT_ISOLATED_EG
                    if white_top[file] > rank:
                        mg -= JIT_DOUBLED_MG
                        eg -= JIT_DOUBLED_EG
                else:
                    relative = 7 - rank
                    blocked = False
                    for f in range(low, high + 1):
                        if white_bottom[f] < rank:
                            blocked = True
                    if not blocked:
                        mg -= JIT_PASSED_MG[relative]
                        eg -= JIT_PASSED_EG[relative]
                    left = black_pawn_files[file - 1] if file > 0 else 0
                    right = black_pawn_files[file + 1] if file < 7 else 0
                    if left == 0 and right == 0:
                        mg += JIT_ISOLATED_MG
                        eg += JIT_ISOLATED_EG
                    if black_bottom[file] < rank:
                        mg += JIT_DOUBLED_MG
                        eg += JIT_DOUBLED_EG

        if white_king >= 0:
            file = white_king & 7
            penalty = 0
            if white_pawn_files[file] == 0:
                penalty += JIT_SHELTER
            if file > 0 and white_pawn_files[file - 1] == 0:
                penalty += JIT_SHELTER
            if file < 7 and white_pawn_files[file + 1] == 0:
                penalty += JIT_SHELTER
            mg -= penalty
        if black_king >= 0:
            file = black_king & 7
            penalty = 0
            if black_pawn_files[file] == 0:
                penalty += JIT_SHELTER
            if file > 0 and black_pawn_files[file - 1] == 0:
                penalty += JIT_SHELTER
            if file < 7 and black_pawn_files[file + 1] == 0:
                penalty += JIT_SHELTER
            mg += penalty

        if phase > JIT_TOTAL_PHASE:
            phase = JIT_TOTAL_PHASE
        total = mg * phase + eg * (JIT_TOTAL_PHASE - phase)
        score = total // JIT_TOTAL_PHASE

        if score > JIT_MOP_UP or score < -JIT_MOP_UP:
            strong_white = score > 0
            weak_pawns = black_pawn_count if strong_white else white_pawn_count
            weak_king = black_king if strong_white else white_king
            strong_king = white_king if strong_white else black_king
            if weak_pawns == 0 and weak_king >= 0 and strong_king >= 0:
                spread = abs((weak_king >> 4) - (strong_king >> 4)) + abs(
                    (weak_king & 7) - (strong_king & 7))
                bonus = 47 * JIT_CENTER_DISTANCE[(weak_king >> 4) * 8 + (weak_king & 7)] + 16 * (
                    14 - spread)
                score += bonus if strong_white else -bonus

        score = (score * (200 - st[3])) // 200
        if st[0] == 1:
            score = -score
        return score + JIT_TEMPO


    @njit(cache=False)
    def jit_cheapest_attacker(occ: Any, square: Any, colour: Any) -> Any:
        best = -1
        best_value = 1 << 30
        pawn = 1 | (colour << 3)
        if colour == 0:
            for step in (-15, -17):
                sq = square + step
                if jit_on_board(sq) and occ[sq] == pawn and JIT_SEE_VALUE[1] < best_value:
                    best_value = JIT_SEE_VALUE[1]
                    best = sq
        else:
            for step in (15, 17):
                sq = square + step
                if jit_on_board(sq) and occ[sq] == pawn and JIT_SEE_VALUE[1] < best_value:
                    best_value = JIT_SEE_VALUE[1]
                    best = sq
        for i in range(8):
            sq = square + KNIGHT_STEPS[i]
            if jit_on_board(sq):
                p = occ[sq]
                if p != 0 and (p >> 3) == colour and (p & 7) == 2 and JIT_SEE_VALUE[2] < best_value:
                    best_value = JIT_SEE_VALUE[2]
                    best = sq
            sq = square + KING_STEPS[i]
            if jit_on_board(sq):
                p = occ[sq]
                if p != 0 and (p >> 3) == colour and (p & 7) == 6 and JIT_SEE_VALUE[6] < best_value:
                    best_value = JIT_SEE_VALUE[6]
                    best = sq
        for i in range(4):
            step = BISHOP_STEPS[i]
            sq = square + step
            while jit_on_board(sq):
                p = occ[sq]
                if p != 0:
                    kind = p & 7
                    if ((p >> 3) == colour and (kind == 3 or kind == 5)
                            and JIT_SEE_VALUE[kind] < best_value):
                        best_value = JIT_SEE_VALUE[kind]
                        best = sq
                    break
                sq += step
            step = ROOK_STEPS[i]
            sq = square + step
            while jit_on_board(sq):
                p = occ[sq]
                if p != 0:
                    kind = p & 7
                    if ((p >> 3) == colour and (kind == 4 or kind == 5)
                            and JIT_SEE_VALUE[kind] < best_value):
                        best_value = JIT_SEE_VALUE[kind]
                        best = sq
                    break
                sq += step
        return best


    @njit(cache=False)
    def jit_see(board: Any, st: Any, move: Any, occ: Any) -> Any:
        """Material a capture wins once both sides trade optimally on the target square."""
        target = jit_move_to(move)
        for i in range(128):
            occ[i] = board[i]
        if (move & MOVE_EP) != 0:
            captured = 1
            occ[target - 16 if st[0] == 0 else target + 16] = 0
        else:
            captured = board[target] & 7
        attacker = board[jit_move_from(move)] & 7
        occ[jit_move_from(move)] = 0

        gains = _np.zeros(40, dtype=_np.int32)
        gains[0] = JIT_SEE_VALUE[captured]
        colour = 1 - st[0]
        index = 0
        while index < 38:
            found = jit_cheapest_attacker(occ, target, colour)
            if found < 0:
                break
            index += 1
            gains[index] = JIT_SEE_VALUE[attacker] - gains[index - 1]
            a = -gains[index - 1]
            b = gains[index]
            if (a if a > b else b) < 0:
                break
            attacker = occ[found] & 7
            occ[found] = 0
            colour = 1 - colour
        while index > 0:
            a = -gains[index - 1]
            b = gains[index]
            gains[index - 1] = -(a if a > b else b)
            index -= 1
        return gains[0]


    @njit(cache=False)
    def jit_score_moves(
            board: Any, st: Any, moves: Any, scores: Any, base: Any, count: Any, tt_move: Any,
            ply: Any, counter_move: Any, killers: Any, hist_heur: Any, occ: Any
    ) -> Any:
        for i in range(count):
            move = moves[base + i]
            if move == tt_move:
                scores[base + i] = 1 << 24
                continue
            captured = jit_move_captured(move) & 7
            promo = jit_move_promo(move)
            if captured != 0:
                attacker = board[jit_move_from(move)] & 7
                value = (1 << 20) + JIT_MVV_LVA[captured] * 16 - attacker
                if (JIT_SEE_VALUE[attacker] > JIT_SEE_VALUE[captured]
                        and jit_see(board, st, move, occ) < 0):
                    value -= 1 << 21
                scores[base + i] = value
            elif promo != 0:
                scores[base + i] = (1 << 20) + JIT_MVV_LVA[promo] * 16
            elif move == killers[ply * 2] or move == killers[ply * 2 + 1]:
                scores[base + i] = 1 << 19
            elif move == counter_move:
                scores[base + i] = (1 << 19) - 1
            else:
                scores[base + i] = hist_heur[
                    st[0] * 16384 + jit_move_from(move) * 128 + jit_move_to(move)]


    @njit(inline="always")
    def jit_pick_next(moves: Any, scores: Any, base: Any, count: Any, index: Any) -> Any:
        """Swap the best remaining move into position `index`. Selecting on demand beats
        sorting the whole list, because most nodes cut off after a few moves."""
        best = index
        for j in range(index + 1, count):
            if scores[base + j] > scores[base + best]:
                best = j
        if best != index:
            moves[base + index], moves[base + best] = moves[base + best], moves[base + index]
            scores[base + index], scores[base + best] = (scores[base + best],
                                                         scores[base + index])


    @njit(cache=False)
    def jit_quiesce(
            board: Any, st: Any, hist: Any, moves: Any, scores: Any, occ: Any, info: Any,
            killers: Any, hist_heur: Any, alpha: Any, beta: Any, ply: Any
    ) -> Any:
        info[NODES] += 1
        if info[NODES] >= info[NODE_LIMIT]:
            info[STOPPED] = 1
            return alpha
        if ply >= JIT_MAX_PLY - 2:
            return jit_evaluate(board, st)

        checked = jit_in_check(board, st, st[0])
        if not checked:
            stand = jit_evaluate(board, st)
            if stand >= beta:
                return stand
            if stand > alpha:
                alpha = stand
        else:
            stand = -JIT_MATE + ply

        base = ply * JIT_MOVE_SLOTS
        count = jit_generate(board, st, moves, base, not checked)
        jit_score_moves(board, st, moves, scores, base, count, 0, ply, 0, killers, hist_heur, occ)

        best = stand
        legal = 0
        for i in range(count):
            jit_pick_next(moves, scores, base, count, i)
            move = moves[base + i]
            if not checked:
                captured = jit_move_captured(move) & 7
                gain = JIT_MVV_LVA[captured] if captured != 0 else 100
                if jit_move_promo(move) != 0:
                    gain += 800
                if stand + gain + JIT_DELTA < alpha:
                    continue
                if captured != 0:
                    attacker = board[jit_move_from(move)] & 7
                    if (JIT_SEE_VALUE[attacker] > JIT_SEE_VALUE[captured]
                            and jit_see(board, st, move, occ) < 0):
                        continue
            if not jit_make_move(board, st, hist, move, ply):
                continue
            legal += 1
            value = -jit_quiesce(board, st, hist, moves, scores, occ, info, killers, hist_heur,
                             -beta, -alpha, ply + 1)
            jit_unmake_move(board, st, hist, move, ply)
            if info[STOPPED] == 1:
                return alpha
            if value > best:
                best = value
            if value > alpha:
                alpha = value
                if alpha >= beta:
                    break
        if checked and legal == 0:
            return -JIT_MATE + ply
        return best


    @njit(cache=False)
    def jit_has_material(board: Any, st: Any) -> Any:
        us = st[0]
        for sq in range(128):
            if (sq & 0x88) != 0:
                continue
            piece = board[sq]
            if piece != 0 and (piece >> 3) == us:
                kind = piece & 7
                if kind != 1 and kind != 6:
                    return True
        return False


    @njit(cache=False)
    def jit_negamax(
            board: Any, st: Any, hist: Any, moves: Any, scores: Any, occ: Any, info: Any,
            killers: Any, hist_heur: Any, counter: Any, rep: Any, tt_key: Any, tt_move_a: Any,
            tt_score_a: Any, tt_meta: Any, key: Any, depth: Any, alpha: Any, beta: Any, ply: Any,
            can_null: Any, previous: Any
    ) -> Any:
        info[NODES] += 1
        if info[NODES] >= info[NODE_LIMIT]:
            info[STOPPED] = 1
            return alpha

        alpha_original = alpha
        if ply > 0:
            for i in range(info[REP_LEN]):
                if rep[i] == key:
                    return 0
            if st[3] >= 100:
                return 0

        slot = int(key & TT_MASK)
        tt_move = 0
        if tt_meta[slot] != 0 and tt_key[slot] == key:
            tt_move = tt_move_a[slot]
            meta = tt_meta[slot]
            entry_depth = meta >> 2
            flag = meta & 3
            if entry_depth >= depth and ply > 0:
                value = tt_score_a[slot]
                if value > JIT_MATE_THRESHOLD:
                    value -= ply
                elif value < -JIT_MATE_THRESHOLD:
                    value += ply
                if flag == EXACT:
                    return value
                if flag == LOWER and value > alpha:
                    alpha = value
                elif flag == UPPER and value < beta:
                    beta = value
                if alpha >= beta:
                    return value

        checked = jit_in_check(board, st, st[0])
        if checked:
            depth += 1
        if depth <= 0:
            return jit_quiesce(board, st, hist, moves, scores, occ, info, killers, hist_heur,
                           alpha, beta, ply)

        is_pv = (beta - alpha) > 1
        if tt_move == 0 and depth >= 4:
            depth -= 1
        static = 0 if checked else jit_evaluate(board, st)

        if (not is_pv) and (not checked) and depth <= 3 and static - 120 * depth >= beta:
            return static

        if (can_null and (not is_pv) and (not checked) and depth >= 3 and static >= beta
                and jit_has_material(board, st)):
            saved_ep = st[2]
            saved_half = st[3]
            st[2] = -1
            st[0] = 1 - st[0]
            st[3] += 1
            null_key = key ^ Z_SIDE
            if saved_ep >= 0:
                null_key ^= Z_EP[saved_ep]
            saved_hash = st[6]
            st[6] = null_key
            rep[info[REP_LEN]] = key
            info[REP_LEN] += 1
            value = -jit_negamax(board, st, hist, moves, scores, occ, info, killers, hist_heur,
                             counter, rep, tt_key, tt_move_a, tt_score_a, tt_meta, null_key,
                             depth - 1 - JIT_NULL_R - depth // 6, -beta, -beta + 1,
                             ply + 1, False, 0)
            info[REP_LEN] -= 1
            st[0] = 1 - st[0]
            st[2] = saved_ep
            st[3] = saved_half
            st[6] = saved_hash
            if info[STOPPED] == 1:
                return alpha
            if value >= beta:
                return beta

        counter_move = 0
        if previous != 0:
            counter_move = counter[jit_move_from(previous) * 128 + jit_move_to(previous)]

        base = ply * JIT_MOVE_SLOTS
        count = jit_generate(board, st, moves, base, False)
        jit_score_moves(board, st, moves, scores, base, count, tt_move, ply, counter_move,
                    killers, hist_heur, occ)

        best_score = -JIT_MATE - 1
        best_move = 0
        legal = 0
        rep[info[REP_LEN]] = key
        info[REP_LEN] += 1

        for i in range(count):
            jit_pick_next(moves, scores, base, count, i)
            move = moves[base + i]
            quiet = jit_move_captured(move) == 0 and jit_move_promo(move) == 0
            if (quiet and (not is_pv) and (not checked) and depth <= 3 and legal > 0
                    and static + JIT_FUTILITY[depth] <= alpha):
                continue
            if not jit_make_move(board, st, hist, move, ply):
                continue
            legal += 1

            child_key = st[6]
            if legal == 1:
                value = -jit_negamax(board, st, hist, moves, scores, occ, info, killers,
                                 hist_heur, counter, rep, tt_key, tt_move_a, tt_score_a,
                                 tt_meta, child_key, depth - 1, -beta, -alpha, ply + 1,
                                 True, move)
            else:
                reduction = 0
                if quiet and depth >= 3 and i >= 4 and not jit_in_check(board, st, st[0]):
                    d_index = depth if depth < 32 else 31
                    m_index = i if i < 64 else 63
                    reduction = JIT_LMR[d_index * 64 + m_index]
                    if reduction > depth - 2:
                        reduction = depth - 2
                    if is_pv and reduction > 0:
                        reduction -= 1
                value = -jit_negamax(board, st, hist, moves, scores, occ, info, killers,
                                 hist_heur, counter, rep, tt_key, tt_move_a, tt_score_a,
                                 tt_meta, child_key, depth - 1 - reduction, -alpha - 1,
                                 -alpha, ply + 1, True, move)
                if reduction > 0 and value > alpha:
                    value = -jit_negamax(board, st, hist, moves, scores, occ, info, killers,
                                     hist_heur, counter, rep, tt_key, tt_move_a, tt_score_a,
                                     tt_meta, child_key, depth - 1, -alpha - 1, -alpha,
                                     ply + 1, True, move)
                if value > alpha and value < beta:
                    value = -jit_negamax(board, st, hist, moves, scores, occ, info, killers,
                                     hist_heur, counter, rep, tt_key, tt_move_a, tt_score_a,
                                     tt_meta, child_key, depth - 1, -beta, -alpha, ply + 1,
                                     True, move)
            jit_unmake_move(board, st, hist, move, ply)
            if info[STOPPED] == 1:
                info[REP_LEN] -= 1
                return alpha

            if value > best_score:
                best_score = value
                best_move = move
            if value > alpha:
                alpha = value
            if alpha >= beta:
                if quiet:
                    if move != killers[ply * 2]:
                        killers[ply * 2 + 1] = killers[ply * 2]
                        killers[ply * 2] = move
                    h = st[0] * 16384 + jit_move_from(move) * 128 + jit_move_to(move)
                    bump = hist_heur[h] + depth * depth
                    hist_heur[h] = bump if bump < (1 << 18) else (1 << 18)
                    if previous != 0:
                        counter[jit_move_from(previous) * 128 + jit_move_to(previous)] = move
                break

        info[REP_LEN] -= 1

        if legal == 0:
            return -JIT_MATE + ply if checked else 0

        stored = best_score
        if stored > JIT_MATE_THRESHOLD:
            stored += ply
        elif stored < -JIT_MATE_THRESHOLD:
            stored -= ply
        if best_score <= alpha_original:
            flag = UPPER
        elif best_score >= beta:
            flag = LOWER
        else:
            flag = EXACT
        existing = (tt_meta[slot] >> 2) if tt_meta[slot] != 0 else -1
        if depth >= existing or flag == EXACT:
            tt_key[slot] = key
            tt_move_a[slot] = best_move
            tt_score_a[slot] = stored
            tt_meta[slot] = (depth << 2) | flag
        return best_score


    @njit(cache=False)
    def jit_search_root(
            board: Any, st: Any, hist: Any, moves: Any, scores: Any, occ: Any, info: Any,
            killers: Any, hist_heur: Any, counter: Any, rep: Any, tt_key: Any, tt_move_a: Any,
            tt_score_a: Any, tt_meta: Any, depth: Any, alpha: Any, beta: Any, previous_best: Any,
            out: Any
    ) -> Any:
        """One pass at a fixed depth. out[0] = score, out[1] = best move."""
        key = st[6]
        base = 0
        count = jit_generate(board, st, moves, base, False)
        jit_score_moves(board, st, moves, scores, base, count, previous_best, 0, 0, killers,
                    hist_heur, occ)

        best_score = -JIT_MATE - 1
        best_move = previous_best
        legal = 0
        rep[info[REP_LEN]] = key
        info[REP_LEN] += 1
        for i in range(count):
            jit_pick_next(moves, scores, base, count, i)
            move = moves[base + i]
            if not jit_make_move(board, st, hist, move, 0):
                continue
            legal += 1
            child_key = st[6]
            if legal == 1:
                value = -jit_negamax(board, st, hist, moves, scores, occ, info, killers,
                                 hist_heur, counter, rep, tt_key, tt_move_a, tt_score_a,
                                 tt_meta, child_key, depth - 1, -beta, -alpha, 1, True, move)
            else:
                value = -jit_negamax(board, st, hist, moves, scores, occ, info, killers,
                                 hist_heur, counter, rep, tt_key, tt_move_a, tt_score_a,
                                 tt_meta, child_key, depth - 1, -alpha - 1, -alpha, 1,
                                 True, move)
                if value > alpha:
                    value = -jit_negamax(board, st, hist, moves, scores, occ, info, killers,
                                     hist_heur, counter, rep, tt_key, tt_move_a, tt_score_a,
                                     tt_meta, child_key, depth - 1, -beta, -alpha, 1,
                                     True, move)
            jit_unmake_move(board, st, hist, move, 0)
            if info[STOPPED] == 1:
                break
            if value > best_score:
                best_score = value
                best_move = move
            if value > alpha:
                alpha = value
        info[REP_LEN] -= 1
        out[0] = best_score
        out[1] = best_move
        return legal


    def jit_make_workspace() -> Any:
        return {
            "hist": _np.zeros(JIT_MAX_PLY * 6, dtype=_np.int64),
            "moves": _np.zeros(JIT_MAX_PLY * JIT_MOVE_SLOTS, dtype=_np.int32),
            "scores": _np.zeros(JIT_MAX_PLY * JIT_MOVE_SLOTS, dtype=_np.int32),
            "occ": _np.zeros(128, dtype=_np.int8),
            "info": _np.zeros(8, dtype=_np.int64),
            "killers": _np.zeros(JIT_MAX_PLY * 2, dtype=_np.int32),
            "hist_heur": _np.zeros(2 * 16384, dtype=_np.int32),
            "counter": _np.zeros(128 * 128, dtype=_np.int32),
            "rep": _np.zeros(JIT_MAX_PLY + 512, dtype=_np.int64),
            "tt_key": _np.zeros(TT_SIZE, dtype=_np.int64),
            "tt_move": _np.zeros(TT_SIZE, dtype=_np.int32),
            "tt_score": _np.zeros(TT_SIZE, dtype=_np.int32),
            "tt_meta": _np.zeros(TT_SIZE, dtype=_np.int32),
            "out": _np.zeros(2, dtype=_np.int64),
        }


# ======================================================================================
# Driver
# ======================================================================================

JIT_STATE: dict[str, Any] = {}


def _jit_warmup() -> None:
    """Compile everything by running a tiny search. This runs on a worker thread so that
    slow hardware costs a few opening moves of strength rather than blowing the 60 second
    import budget, which would lose every game outright."""
    global JIT_READY, JIT_ERROR
    try:
        work = jit_make_workspace()
        board, st = jit_new_state(chess.STARTING_FEN)
        work["info"][NODE_LIMIT] = 40000
        jit_search_root(board, st, work["hist"], work["moves"], work["scores"], work["occ"],
                    work["info"], work["killers"], work["hist_heur"], work["counter"],
                    work["rep"], work["tt_key"], work["tt_move"], work["tt_score"],
                    work["tt_meta"], 2, -31000, 31000, 0, work["out"])
        JIT_STATE["work"] = work
        JIT_READY = True
    except Exception as exc:
        JIT_ERROR = repr(exc)


def _machine_is_fast_enough() -> bool:
    """Time a trivial compile and extrapolate. Compiling the real search takes about
    eleven times as long, so if the estimate does not fit comfortably inside the 60
    second import budget we skip the JIT entirely and play with the Python engine -
    a weaker agent still beats an agent that never finishes starting up."""
    try:
        started = time.monotonic()

        @njit
        def _probe(x: Any) -> Any:
            total = 0
            for i in range(x):
                total += i
            return total

        _probe(4)
        probe = time.monotonic() - started
    except Exception:
        return False
    return probe * 11.0 < 40.0


if HAVE_NUMBA:
    # Compilation must happen on the main thread: numba deadlocks if it is driven from a
    # worker, so there is no way to overlap it with anything.
    if _machine_is_fast_enough():
        _jit_warmup()
    else:
        JIT_ERROR = "machine too slow to compile within the import budget"

# Positions already played this game, so the JIT search can jit_see repetitions.
HISTORY_FENS: list[str] = []


def _jit_choose(fen: str, time_left_ms: int) -> str:
    """Iterative deepening on a node budget, with the wall clock enforced between
    iterations. The node cap is derived from a running estimate of our own speed."""
    work = JIT_STATE["work"]
    board, st = jit_new_state(fen)

    rep = work["rep"]
    length = 0
    for past in HISTORY_FENS[-80:]:
        past_board, past_st = jit_new_state(past)
        rep[length] = jit_full_hash(past_board, past_st)
        length += 1
    base_rep = length

    pieces = int((board != 0).sum())
    usable = max(0.0, time_left_ms - 300.0)
    expected = 22 if pieces > 20 else (28 if pieces > 10 else 18)
    soft = min(usable / expected + 350.0, usable * 0.25)
    hard = min(soft * 3.0, usable * 0.45)
    soft = max(soft, 10.0)
    hard = max(hard, 20.0)

    started = time.monotonic()
    speed = JIT_STATE.get("nps", 250000.0)
    best = 0
    previous = 0
    score = 0
    for depth in range(1, 64):
        remaining = hard - (time.monotonic() - started) * 1000.0
        if remaining <= 0:
            break
        window = 40
        alpha = -31000 if depth < 4 else score - window
        beta = 31000 if depth < 4 else score + window
        stopped = False
        while True:
            remaining = hard - (time.monotonic() - started) * 1000.0
            if remaining <= 0:
                stopped = True
                break
            work["info"][NODES] = 0
            work["info"][STOPPED] = 0
            work["info"][REP_LEN] = base_rep
            work["info"][NODE_LIMIT] = max(4000, int(remaining / 1000.0 * speed))
            tick = time.monotonic()
            jit_search_root(board, st, work["hist"], work["moves"], work["scores"],
                        work["occ"], work["info"], work["killers"], work["hist_heur"],
                        work["counter"], rep, work["tt_key"], work["tt_move"],
                        work["tt_score"], work["tt_meta"], depth, alpha, beta, best,
                        work["out"])
            spent = time.monotonic() - tick
            nodes = int(work["info"][NODES])
            if spent > 0.02:
                speed = 0.6 * speed + 0.4 * (nodes / spent)
            if work["info"][STOPPED] == 1:
                stopped = True
                break
            value = int(work["out"][0])
            if alpha < value < beta:
                score = value
                best = int(work["out"][1])
                break
            if value <= alpha:
                alpha = max(-31000, alpha - window * 3)
            else:
                beta = min(31000, beta + window * 3)
                best = int(work["out"][1])
            window *= 2
        if stopped:
            break
        elapsed = (time.monotonic() - started) * 1000.0
        unstable = best != previous
        previous = best
        if elapsed > soft * (1.0 if unstable else 0.6):
            break

    JIT_STATE["nps"] = speed
    return jit_move_uci(best) if best else ""


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in `fen`."""
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if not moves:
        return "0000"
    fallback = moves[0].uci()

    try:
        HISTORY_FENS.append(fen)
        if not JIT_READY:
            # Still compiling, or numba unavailable: the pure-Python engine handles the
            # whole move, including its own book and repetition bookkeeping.
            return _python_get_move(fen, time_left_ms)

        if len(moves) == 1:
            return fallback

        opening = BOOK.get(" ".join(fen.split()[:4]))
        if opening:
            playable = [m for m in (chess.Move.from_uci(u) for u in opening) if m in moves]
            if playable:
                return random.choice(playable).uci()

        if time_left_ms <= PANIC_MS:
            searcher = Searcher(time.monotonic() + 0.05)
            return searcher.order(board, moves, None, 0)[0].uci()

        uci = _jit_choose(fen, time_left_ms)
        if uci and chess.Move.from_uci(uci) in moves:
            return uci
        return _python_get_move(fen, time_left_ms)
    except Exception as exc:  # never lose a game to an unhandled error
        # The output cap is 4096 bytes per move and passing it loses the game.
        # A numba TypingError repr alone runs to several kilobytes, so this is
        # truncated rather than trusted.
        print(f"search failed, playing fallback: {exc!r}"[:200])
        return fallback
