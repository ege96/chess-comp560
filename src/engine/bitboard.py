from engine.BaseEngine import BaseEngine
from engine.bitboard_utils import *
import chess
from typing import Optional, Tuple, Dict

# Piece values and piece-square tables (same as minimax.py, but for bitboard use)
PIECE_VALUES = [100, 320, 330, 500, 900, 20000]

# Piece-square tables (midgame, white's perspective)
pst = {
    PAWN: [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0
    ],
    KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    ROOK: [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         0,  0,  0,  5,  5,  0,  0,  0
    ],
    QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20
    ],
}

MOBILITY_BONUS = 1

class BitboardPosition:
    """Bitboard representation of a chess position."""
    def __init__(self, board: chess.Board):
        # 2x6 array: [color][piece_type] = bitboard
        self.pieces = [[0 for _ in range(6)] for _ in range(2)]
        self.occupancy = [0, 0]  # [white, black]
        self.all_occupancy = 0
        self.side_to_move = WHITE if board.turn == chess.WHITE else BLACK
        self.castling_rights = board.castling_rights
        self.ep_square = board.ep_square
        self.halfmove_clock = board.halfmove_clock
        self.fullmove_number = board.fullmove_number
        # Fill bitboards from python-chess board
        for sq in range(64):
            piece = board.piece_at(sq)
            if piece:
                color = WHITE if piece.color == chess.WHITE else BLACK
                ptype = piece.piece_type - 1
                self.pieces[color][ptype] |= 1 << sq
                self.occupancy[color] |= 1 << sq
                self.all_occupancy |= 1 << sq

class BitboardEngine(BaseEngine):
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth

    def get_best_move(self, board: chess.Board) -> str:
        pos = BitboardPosition(board)
        _, move = self.minimax(pos, self.max_depth, float('-inf'), float('inf'), pos.side_to_move)
        if move is None:
            # Defensive: fallback to first legal move if minimax fails
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return legal_moves[0].uci()
            return "0000"
        return move

    def get_evaluation(self, board: chess.Board) -> float:
        pos = BitboardPosition(board)
        return self.evaluate(pos)

    def evaluate(self, pos: BitboardPosition) -> float:
        # Material and PST
        score = 0
        for color in [WHITE, BLACK]:
            sign = 1 if color == WHITE else -1
            for ptype in range(6):
                bb = pos.pieces[color][ptype]
                value = PIECE_VALUES[ptype]
                for sq in bitscan(bb):
                    score += sign * value
                    # PST: mirror for black
                    idx = sq if color == WHITE else (sq ^ 56)
                    score += sign * pst[ptype][idx]
        # Mobility (simple: count moves)
        # For performance, you may want to cache or approximate
        # Here, just count pseudo-legal moves for both sides
        score += MOBILITY_BONUS * (self.count_moves(pos, WHITE) - self.count_moves(pos, BLACK))
        return score

    def minimax(self, pos: BitboardPosition, depth: int, alpha: float, beta: float, maximizing_color: int) -> Tuple[float, Optional[str]]:
        if depth == 0 or self.is_game_over(pos):
            return self.evaluate(pos), None
        best_move = None
        if maximizing_color == WHITE:
            max_eval = float('-inf')
            for move, new_pos in self.generate_moves(pos, WHITE):
                eval, _ = self.minimax(new_pos, depth - 1, alpha, beta, BLACK)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move, new_pos in self.generate_moves(pos, BLACK):
                eval, _ = self.minimax(new_pos, depth - 1, alpha, beta, WHITE)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def is_game_over(self, pos: BitboardPosition) -> bool:
        # Simple: no king or no moves
        for color in [WHITE, BLACK]:
            if pos.pieces[color][KING] == 0:
                return True
        # No legal moves (stalemate/checkmate)
        if self.count_moves(pos, pos.side_to_move) == 0:
            return True
        return False

    def count_moves(self, pos: BitboardPosition, color: int) -> int:
        return sum(1 for _ in self.generate_moves(pos, color))

    def generate_moves(self, pos: BitboardPosition, color: int):
        # Generate pseudo-legal moves for all pieces of the given color
        # For brevity, only implement pawn, knight, king, and basic sliding moves
        for ptype in range(6):
            bb = pos.pieces[color][ptype]
            for from_sq in bitscan(bb):
                if ptype == PAWN:
                    # Pawn pushes and captures
                    direction = 8 if color == WHITE else -8
                    to_sq = from_sq + direction
                    if 0 <= to_sq < 64 and not (pos.all_occupancy & square_mask(to_sq)):
                        yield self.move_uci(from_sq, to_sq), self.make_move(pos, from_sq, to_sq, color, PAWN)
                    # Captures
                    attacks = WHITE_PAWN_ATTACKS[from_sq] if color == WHITE else BLACK_PAWN_ATTACKS[from_sq]
                    for to_sq in bitscan(attacks & pos.occupancy[1 - color]):
                        yield self.move_uci(from_sq, to_sq), self.make_move(pos, from_sq, to_sq, color, PAWN)
                elif ptype == KNIGHT:
                    for to_sq in bitscan(KNIGHT_MOVES[from_sq] & ~pos.occupancy[color]):
                        yield self.move_uci(from_sq, to_sq), self.make_move(pos, from_sq, to_sq, color, KNIGHT)
                elif ptype == KING:
                    for to_sq in bitscan(KING_MOVES[from_sq] & ~pos.occupancy[color]):
                        yield self.move_uci(from_sq, to_sq), self.make_move(pos, from_sq, to_sq, color, KING)
                else:  # Sliding pieces
                    for d in SLIDING_DIRECTIONS[ptype]:
                        to_sq = from_sq
                        while True:
                            to_sq += d
                            if not (0 <= to_sq < 64):
                                break
                            # Edge wrap
                            if abs(get_file(to_sq) - get_file(from_sq)) > 2:
                                break
                            if pos.occupancy[color] & square_mask(to_sq):
                                break
                            yield self.move_uci(from_sq, to_sq), self.make_move(pos, from_sq, to_sq, color, ptype)
                            if pos.occupancy[1 - color] & square_mask(to_sq):
                                break
        # TODO: Add castling, en passant, promotions, etc.

    def move_uci(self, from_sq: int, to_sq: int) -> str:
        # Convert from/to squares to UCI string
        files = 'abcdefgh'
        return f"{files[from_sq % 8]}{1 + from_sq // 8}{files[to_sq % 8]}{1 + to_sq // 8}"

    def make_move(self, pos: BitboardPosition, from_sq: int, to_sq: int, color: int, ptype: int) -> BitboardPosition:
        # Return a new BitboardPosition with the move applied
        new_pos = BitboardPosition.__new__(BitboardPosition)
        new_pos.pieces = [row[:] for row in pos.pieces]
        new_pos.occupancy = pos.occupancy[:]
        new_pos.all_occupancy = pos.all_occupancy
        new_pos.side_to_move = 1 - color
        new_pos.castling_rights = pos.castling_rights
        new_pos.ep_square = None
        new_pos.halfmove_clock = pos.halfmove_clock + 1
        new_pos.fullmove_number = pos.fullmove_number + (1 if color == BLACK else 0)
        # Remove piece from from_sq
        new_pos.pieces[color][ptype] &= ~square_mask(from_sq)
        new_pos.occupancy[color] &= ~square_mask(from_sq)
        # Capture if any
        for enemy_ptype in range(6):
            if new_pos.pieces[1 - color][enemy_ptype] & square_mask(to_sq):
                new_pos.pieces[1 - color][enemy_ptype] &= ~square_mask(to_sq)
                new_pos.occupancy[1 - color] &= ~square_mask(to_sq)
        # Place piece on to_sq
        new_pos.pieces[color][ptype] |= square_mask(to_sq)
        new_pos.occupancy[color] |= square_mask(to_sq)
        # Update all_occupancy
        new_pos.all_occupancy = new_pos.occupancy[WHITE] | new_pos.occupancy[BLACK]
        return new_pos 