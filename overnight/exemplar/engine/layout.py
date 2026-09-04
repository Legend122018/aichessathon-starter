"""Memory layout for the engine's mutable state.

Numba freezes module-level numpy arrays as *read-only* constants, so anything
the search writes has to be passed in as an argument.  Passing twenty separate
arrays down a hot recursive call would cost real nodes/second, so all int32
state is packed into one array `I` and all uint64 state into `tt` / `hist` /
`undo`.  Offsets below are compile-time constants, so `I[OI_HISTORY + x]` folds
to a single indexed load.
"""
import numpy as np

MAX_PLY = 128
MAX_MOVES = 256
MAX_HIST = 1024
UN_N = 8
PVSTRIDE = MAX_PLY + 2

# ---- int32 state block ---------------------------------------------------
_off = 0
def _alloc(n):
    global _off
    o = _off
    _off += n
    return o

NPARAM = 1003                      # kept in sync with evaluate._SPEC (asserted there)

OI_EVALP   = _alloc(NPARAM)
OI_MBUF    = _alloc(MAX_PLY * MAX_MOVES)
OI_MSCR    = _alloc(MAX_PLY * MAX_MOVES)
OI_SBUF    = _alloc(MAX_PLY * MAX_MOVES)
OI_SSCR    = _alloc(MAX_PLY * MAX_MOVES)
OI_KILL    = _alloc(MAX_PLY * 2)
OI_HISTORY = _alloc(2 * 64 * 64)
OI_CAPHIST = _alloc(12 * 64 * 7)
OI_COUNTER = _alloc(12 * 64)
OI_CONTHIST= _alloc(12 * 64 * 12 * 64)
OI_SSEVAL  = _alloc(MAX_PLY + 8)
OI_SSMOVE  = _alloc(MAX_PLY + 8)
OI_SSPIECE = _alloc(MAX_PLY + 8)
OI_SSEXCL  = _alloc(MAX_PLY + 8)
OI_PV      = _alloc(PVSTRIDE * PVSTRIDE)
OI_PVLEN   = _alloc(PVSTRIDE)
OI_QLIST   = _alloc(MAX_PLY * 64)
OI_CLIST   = _alloc(MAX_PLY * 64)
OI_ROOTMV  = _alloc(MAX_MOVES)
OI_ROOTSC  = _alloc(MAX_MOVES)
OI_ROOTNODE= _alloc(MAX_MOVES)
OI_ROOTPV  = _alloc(MAX_PLY)
I_LEN = _off

# ---- int64 control block -------------------------------------------------
_off = 0
ON_NODES   = _alloc(1)
ON_QNODES  = _alloc(1)
ON_SELDEP  = _alloc(1)
ON_STOP    = _alloc(1)
ON_TTAGE   = _alloc(1)
ON_ROOTN   = _alloc(1)
ON_BESTMV  = _alloc(1)
ON_BESTSC  = _alloc(1)
ON_DEPTH   = _alloc(1)
ON_TBHIT   = _alloc(1)
ON_PONDER  = _alloc(1)
ON_NODELIM = _alloc(1)
ON_SPARAM  = _alloc(64)            # runtime-tunable search parameters
N_LEN = _off

# ---- uint64 TT block -----------------------------------------------------
TT_BITS_DEFAULT = 22


def tt_alloc(bits=TT_BITS_DEFAULT):
    size = 1 << bits
    return np.zeros(2 * size, dtype=np.uint64), size


def new_int_state():
    return np.zeros(I_LEN, dtype=np.int32)


def new_ctl_state():
    return np.zeros(N_LEN, dtype=np.int64)
