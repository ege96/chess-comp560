from engine.BaseEngine import BaseEngine
import chess
from typing import Tuple, Optional, Dict

# Sunfish piece values and piece-square tables
piece = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}
pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}
# Pad tables and join piece and pst dictionaries
for k, table in pst.items():
    padrow = lambda row: (0,) + tuple(x + piece[k] for x in row) + (0,)
    pst[k] = sum((padrow(table[i * 8 : i * 8 + 8]) for i in range(8)), ())
    pst[k] = (0,) * 20 + pst[k] + (0,) * 20

# Sunfish piece type mapping to python-chess
PIECE_TYPE_TO_CHAR = {
    1: 'P',
    2: 'N',
    3: 'B',
    4: 'R',
    5: 'Q',
    6: 'K',
}

# Sunfish piece values for python-chess types
piece_values = {
    1: 100,  # Pawn
    2: 280,  # Knight
    3: 320,  # Bishop
    4: 479,  # Rook
    5: 929,  # Queen
    6: 60000 # King
}

# Helper to get mirrored square index
def mirrored_square(sq):
    return sq ^ 56

class MinimaxEngine(BaseEngine):
    # Make piece values a class attribute
    piece_values: Dict[int, int] = piece_values
    pst: Dict[str, tuple] = pst
    MOBILITY_BONUS = 1

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        # Determine if it's endgame - rough heuristic based on queen count
        # Could be refined further (e.g., total material)
        self.is_endgame_phase = False # Will be updated in get_evaluation

    def get_evaluation(self, board: chess.Board) -> float:
        """Evaluate the current board position based on material, piece-square tables, and mobility."""
        if board.is_checkmate():
            # Return a large value favoring the side that delivered checkmate
            # Adding depth ensures faster mates are preferred
            return -float('inf') - self.max_depth if board.turn == chess.WHITE else float('inf') + self.max_depth
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0 # Draw condition

        # Determine game phase (simplistic: no queens or few minor pieces = endgame)
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        minors = len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.WHITE)) + \
                 len(board.pieces(chess.KNIGHT, chess.BLACK)) + len(board.pieces(chess.BISHOP, chess.BLACK))
        self.is_endgame_phase = queens == 0 or (queens <= 1 and minors <= 2)


        material_score = 0
        pst_score = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # Material Score
                # Use Sunfish piece values and PSTs
                pt = piece.piece_type
                color = piece.color
                value = self.piece_values[pt]
                material_score += value if color == chess.WHITE else -value

                # Use Sunfish PSTs
                char = PIECE_TYPE_TO_CHAR[pt]
                piece_pst = self.pst[char]
                # Get PST value - mirror index for black pieces
                if color == chess.WHITE:
                    pst_val = piece_pst[square + 21]  # Sunfish uses 0-based padding, offset by 21
                    pst_score += pst_val
                else:
                    mirror_sq = 119 - (square + 21)
                    pst_val = piece_pst[mirror_sq]
                    pst_score -= pst_val


        # Mobility Score (simple version: count legal moves)
        # Note: Calculating legal moves can be expensive, use cautiously or find ways to optimize.
        # This adds a small bonus for having more available moves.
        board.turn = chess.WHITE # Evaluate mobility from white's perspective
        white_mobility = board.legal_moves.count()
        board.turn = chess.BLACK # Evaluate mobility from black's perspective
        black_mobility = board.legal_moves.count()
        board.turn = not board.turn # Restore original turn

        # Mobility score contributes less than material/PSTs usually
        # Scale bonus to avoid overpowering material, e.g., 0.1 per move difference
        mobility_score = self.MOBILITY_BONUS * (white_mobility - black_mobility)


        # Combine scores (adjust weights as needed)
        total_score = material_score + pst_score + mobility_score

        # Ensure the score is from the perspective of the current player
        # The minimax function handles maximizing/minimizing based on board.turn
        # But the raw evaluation should be consistent (e.g. +ve favors White)
        # return total_score if board.turn == chess.WHITE else -total_score
        # Let's keep evaluation absolute (positive = white advantage)
        return total_score


    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[float, Optional[chess.Move]]:
        """Minimax algorithm with alpha-beta pruning."""
        if depth == 0 or board.is_game_over():
            # Pass remaining depth to evaluation for mate distance scoring
            eval_score = self.get_evaluation(board)
            # Adjust checkmate score based on depth
            if abs(eval_score) > 10000: # Check if it's a mate score ( > King value)
                 eval_score += depth if eval_score > 0 else -depth
            return eval_score, None

        if maximizing:
            max_eval = float('-inf')
            best_move = None
            # Move Ordering: Evaluate captures first (simple heuristic)
            moves = sorted(board.legal_moves, key=lambda move: board.is_capture(move), reverse=True)
            for move in moves:
                board.push(move)
                eval, _ = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()

                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break # Beta cut-off
            if best_move is None and list(board.legal_moves):
                best_move = list(board.legal_moves)[0]
            return max_eval, best_move
        else: # Minimizing player
            min_eval = float('inf')
            best_move = None
            moves = sorted(board.legal_moves, key=lambda move: board.is_capture(move), reverse=True)
            for move in moves:
                board.push(move)
                eval, _ = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()

                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break # Alpha cut-off
            if best_move is None and list(board.legal_moves):
                best_move = list(board.legal_moves)[0]
            return min_eval, best_move

    def get_best_move(self, board: chess.Board) -> str:
        """Get the best move for the current position using minimax."""
        _, best_move = self.minimax(
            board,
            self.max_depth,
            float('-inf'),
            float('inf'),
            board.turn == chess.WHITE # maximizing is True if it's White's turn
        )

        if best_move is None:
            # If minimax returns no move but game isn't over (e.g. stalemate detection failed?)
            # or if only illegal moves were somehow generated, pick first legal if available.
            try:
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    return legal_moves[0].uci()
                else: # No legal moves -> game must be over
                    return "0000" # Null move for game over
            except Exception:
                 return "0000" # Fallback null move

        return best_move.uci()
