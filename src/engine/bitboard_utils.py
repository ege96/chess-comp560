import numpy as np

# Bitboard constants for piece types and colors
WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)

# Board masks
A_FILE = 0x0101010101010101
H_FILE = 0x8080808080808080
RANK_1 = 0x00000000000000FF
RANK_8 = 0xFF00000000000000

# Directions for sliding pieces
NORTH, SOUTH, EAST, WEST = 8, -8, 1, -1
NORTH_EAST, NORTH_WEST, SOUTH_EAST, SOUTH_WEST = 9, 7, -7, -9
SLIDING_DIRECTIONS = {
    BISHOP: [NORTH_EAST, NORTH_WEST, SOUTH_EAST, SOUTH_WEST],
    ROOK:   [NORTH, SOUTH, EAST, WEST],
    QUEEN:  [NORTH, SOUTH, EAST, WEST, NORTH_EAST, NORTH_WEST, SOUTH_EAST, SOUTH_WEST],
}

# Precomputed knight, king, pawn attacks (initialize later)
KNIGHT_MOVES = [0] * 64
KING_MOVES = [0] * 64
WHITE_PAWN_ATTACKS = [0] * 64
BLACK_PAWN_ATTACKS = [0] * 64

for sq in range(64):
    # Knight
    attacks = 0
    for d in (17, 15, 10, 6, -17, -15, -10, -6):
        to = sq + d
        if 0 <= to < 64 and abs((to % 8) - (sq % 8)) <= 2:
            attacks |= 1 << to
    KNIGHT_MOVES[sq] = attacks
    # King
    attacks = 0
    for d in (8, -8, 1, -1, 9, 7, -9, -7):
        to = sq + d
        if 0 <= to < 64 and abs((to % 8) - (sq % 8)) <= 1:
            attacks |= 1 << to
    KING_MOVES[sq] = attacks
    # Pawn
    w_att, b_att = 0, 0
    if sq % 8 != 0 and sq + 7 < 64:
        w_att |= 1 << (sq + 7)
    if sq % 8 != 7 and sq + 9 < 64:
        w_att |= 1 << (sq + 9)
    if sq % 8 != 0 and sq - 9 >= 0:
        b_att |= 1 << (sq - 9)
    if sq % 8 != 7 and sq - 7 >= 0:
        b_att |= 1 << (sq - 7)
    WHITE_PAWN_ATTACKS[sq] = w_att
    BLACK_PAWN_ATTACKS[sq] = b_att


def pop_lsb(bb: int):
    """
    Remove and return index of least-significant bit.
    Returns (index, new_bb).
    """
    lsb = bb & -bb
    idx = lsb.bit_length() - 1
    return idx, bb & (bb - 1)


def bitscan(bb: int):
    """
    Yield indices of all set bits from least to most.
    """
    while bb:
        lsb = bb & -bb
        idx = lsb.bit_length() - 1
        yield idx
        bb &= bb - 1


def mirror64(bb: int) -> int:
    """
    Mirror bitboard vertically (rank <-> rank).
    """
    k1 = 0x00FF00FF00FF00FF
    k2 = 0x0000FFFF0000FFFF
    bb = ((bb >> 8) & k1) | ((bb & k1) << 8)
    bb = ((bb >> 16) & k2) | ((bb & k2) << 16)
    bb = ((bb >> 32) | (bb << 32)) & 0xFFFFFFFFFFFFFFFF
    return bb


def square_mask(sq: int) -> int:
    return 1 << sq

def get_file(sq: int) -> int:
    return sq % 8

def get_rank(sq: int) -> int:
    return sq // 8
