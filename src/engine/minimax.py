from engine.BaseEngine import BaseEngine
import chess
from typing import Tuple, Optional, Dict
import torch
from engine.nn_eval import ChessNet, board_to_tensor

# Piece Square Tables (Midgame) - values from White's perspective
# Mirrored vertically for Black
pst_pawn_mg = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]
pst_knight_mg = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]
pst_bishop_mg = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]
pst_rook_mg = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
]
pst_queen_mg = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]
pst_king_mg = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
]

# Piece Square Tables (Endgame) - King specifically needs different values
pst_king_eg = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
]

# Helper to get mirrored square index
def mirrored_square(sq):
    return sq ^ 56

class MinimaxEngine(BaseEngine):
    # Make piece values a class attribute
    piece_values: Dict[chess.PieceType, int] = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    # Piece square tables as class attributes
    pst: Dict[chess.PieceType, list[int]] = {
        chess.PAWN: pst_pawn_mg,
        chess.KNIGHT: pst_knight_mg,
        chess.BISHOP: pst_bishop_mg,
        chess.ROOK: pst_rook_mg,
        chess.QUEEN: pst_queen_mg,
        chess.KING: pst_king_mg # Default to midgame king table
    }
    pst_king_endgame = pst_king_eg # Separate endgame table for king

    # Mobility bonus per legal move
    MOBILITY_BONUS = 1

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.model = ChessNet()
        self.model.load_state_dict(torch.load("chess_rl_model.pth", map_location="cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()
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

        tensor = torch.tensor(board_to_tensor(board)).unsqueeze(0)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
            self.model.cuda()
        with torch.no_grad():
            value = self.model(tensor).item()
        return value

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
            # for move in board.legal_moves:
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
            # Handle case where no move is found (shouldn't happen if not game_over)
            if best_move is None and list(board.legal_moves):
                best_move = list(board.legal_moves)[0]
            return max_eval, best_move
        else: # Minimizing player
            min_eval = float('inf')
            best_move = None
            # Move Ordering: Evaluate captures first
            moves = sorted(board.legal_moves, key=lambda move: board.is_capture(move), reverse=True)
            # for move in board.legal_moves:
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
            # Handle case where no move is found
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
            # This is defensive coding.
            try:
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    return legal_moves[0].uci()
                else: # No legal moves -> game must be over
                    return "0000" # Null move for game over
            except Exception:
                 return "0000" # Fallback null move

        return best_move.uci()
