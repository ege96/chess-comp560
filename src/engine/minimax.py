from src.engine.BaseEngine import BaseEngine
import chess
from typing import Tuple, Optional

class MinimaxEngine(BaseEngine):
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        # Piece values for material evaluation
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

    def get_evaluation(self, board: chess.Board) -> float:
        """Evaluate the current board position based on material."""
        if board.is_checkmate():
            return -float('inf') if board.turn else float('inf')
            
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        return score

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[float, Optional[chess.Move]]:
        """Minimax algorithm with alpha-beta pruning."""
        if depth == 0 or board.is_game_over():
            return self.get_evaluation(board), None

        if maximizing:
            max_eval = float('-inf')
            best_move = None
            for move in board.legal_moves:
                board.push(move)
                eval, _ = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in board.legal_moves:
                board.push(move)
                eval, _ = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def get_best_move(self, board: chess.Board) -> str:
        """Get the best move for the current position using minimax."""
        _, best_move = self.minimax(
            board,
            self.max_depth,
            float('-inf'),
            float('inf'),
            board.turn
        )
        
        if best_move is None:
            # If no legal moves, return a null move
            return "0000"
            
        return best_move.uci()
