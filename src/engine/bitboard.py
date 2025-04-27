from engine.BaseEngine import BaseEngine
from engine.bitboard_utils import *
import chess

# Leverage python-chess for generating legal moves, and bitboards for evaluation.

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
    def __init__(self, board: chess.Board):
        self.board = board
        self.side_to_move = WHITE if board.turn == chess.WHITE else BLACK
        # build bitboards
        self.pieces = [[0]*6 for _ in range(2)]
        self.occupancy = [0,0]
        self.all_occupancy = 0
        for sq in range(64):
            pc = board.piece_at(sq)
            if pc:
                c = WHITE if pc.color == chess.WHITE else BLACK
                t = pc.piece_type - 1
                mask = 1 << sq
                self.pieces[c][t] |= mask
                self.occupancy[c] |= mask
                self.all_occupancy |= mask

class BitboardEngine(BaseEngine):
    def __init__(self, max_depth=4):
        self.max_depth = max_depth

    def get_best_move(self, board: chess.Board) -> str:
        pos = BitboardPosition(board)
        _, uci = self._minimax(pos, self.max_depth, -float('inf'), float('inf'), pos.side_to_move)
        return uci or (list(board.legal_moves)[0].uci() if board.legal_moves else '0000')

    def evaluate(self, pos: BitboardPosition) -> float:
        score = 0
        for color in (WHITE, BLACK):
            sign = 1 if color==WHITE else -1
            for p in range(6):
                bb = pos.pieces[color][p]
                val = PIECE_VALUES[p]
                for sq in bitscan(bb):
                    score += sign*val
                    # pst contribution (mirror black)
                    idx = sq if color==WHITE else (sq^56)
                    score += sign*pst[p][idx]
        # mobility
        score += MOBILITY_BONUS*(sum(1 for _ in self._gen_moves(pos, WHITE)) - sum(1 for _ in self._gen_moves(pos, BLACK)))
        return score

    def _minimax(self, pos, depth, alpha, beta, side):
        if depth==0 or pos.board.is_game_over():
            return self.evaluate(pos), None
        best_move = None
        if side==WHITE:
            maxv = -float('inf')
            for uci,new in self._gen_moves(pos,WHITE):
                ev,_ = self._minimax(new, depth-1, alpha, beta, BLACK)
                if ev>maxv:
                    maxv, best_move = ev, uci
                alpha = max(alpha, ev)
                if beta<=alpha: break
            return maxv, best_move
        else:
            minv = float('inf')
            for uci,new in self._gen_moves(pos,BLACK):
                ev,_ = self._minimax(new, depth-1, alpha, beta, WHITE)
                if ev<minv:
                    minv, best_move = ev, uci
                beta = min(beta, ev)
                if beta<=alpha: break
            return minv, best_move

    def _gen_moves(self, pos: BitboardPosition, color: int):
        # use python-chess for legal moves
        board = pos.board
        for mv in board.legal_moves:
            nb = board.copy(stack=False)
            nb.push(mv)
            yield mv.uci(), BitboardPosition(nb)