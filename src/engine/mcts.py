import chess
import random
import math
from typing import Optional, List, Tuple
from engine.BaseEngine import BaseEngine

# --- Evaluation Components (similar to Minimax) ---
# Piece Square Tables (Midgame) - values from White's perspective
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

piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000 # Large value, but not used directly in MCTS eval probability
}

pst_tables = {
    chess.PAWN: pst_pawn_mg,
    chess.KNIGHT: pst_knight_mg,
    chess.BISHOP: pst_bishop_mg,
    chess.ROOK: pst_rook_mg,
    chess.QUEEN: pst_queen_mg,
    chess.KING: pst_king_mg # Default midgame
}

def mirrored_square(sq):
    return sq ^ 56

# --- MCTS Node and Engine ---
class MCTSNode:
    def __init__(self, state: chess.Board, parent: Optional['MCTSNode']=None, move: Optional[chess.Move]=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = list(state.legal_moves)
        self.children = []
        # The player who just moved to this state
        self.player_just_moved = not state.turn

    def uct_select_child(self, exploration_constant: float):
        """Select child with highest UCB1 value."""
        return max(
            self.children,
            key=lambda c: c.wins / c.visits + exploration_constant * math.sqrt(math.log(self.visits) / c.visits),
        )

    def add_child(self, move: chess.Move):
        """Add a new child node for the given move."""
        next_state = self.state.copy()
        next_state.push(move)
        child = MCTSNode(next_state, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result: float):
        """Update this node's statistics from simulation result."""
        self.visits += 1
        self.wins += result

class MCTSEngine(BaseEngine):
    """Monte Carlo Tree Search engine."""
    def __init__(self, max_iterations: int = 20000, exploration_constant: float = 1.414, max_playout_moves: int = 100):
        # Configure MCTS parameters: iterations, exploration constant, and playout move limit
        # Increased default iterations significantly for strength.
        self.max_iterations = max_iterations
        self.C = exploration_constant
        self.max_playout_moves = max_playout_moves

    def get_best_move(self, board: chess.Board) -> str:
        root = MCTSNode(board.copy())
        for _ in range(self.max_iterations):
            node = root
            state = board.copy()
            # Selection
            while not node.untried_moves and node.children:
                node = node.uct_select_child(self.C)
                state.push(node.move)
            # Expansion
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                state.push(move)
                node = node.add_child(move)
            # Simulation (random playout with move limit and capture bias)
            playout_count = 0
            while not state.is_game_over() and playout_count < self.max_playout_moves:
                possible_moves = list(state.legal_moves)
                # Separate captures and non-captures
                captures = [m for m in possible_moves if state.is_capture(m)]
                non_captures = [m for m in possible_moves if not state.is_capture(m)]

                # Bias towards captures: If captures exist, choose one randomly.
                # Otherwise, choose a non-capture randomly.
                if captures:
                    move_to_play = random.choice(captures)
                elif non_captures: # Check if non_captures list is not empty
                    move_to_play = random.choice(non_captures)
                else:
                    # Should not happen if game is not over, but handle defensively
                    break # No legal moves found

                state.push(move_to_play)
                playout_count += 1
            # If terminal, get exact result; otherwise use heuristic evaluation
            if state.is_game_over():
                result = self._get_result(state, root.player_just_moved)
            else:
                result = self._evaluate_nonterminal(state, root.player_just_moved)
            # Backpropagation
            while node:
                node.visits += 1
                # If the node's mover is the same as root's mover, use result directly, else invert
                if node.player_just_moved == root.player_just_moved:
                    node.wins += result
                else:
                    node.wins += 1.0 - result
                node = node.parent
        # Choose the move with highest visit count
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move.uci()

    def get_evaluation(self, board: chess.Board) -> float:
        """Simple material-only evaluation (interface compliance)."""
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,
        }
        score = 0.0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                score += values[piece.piece_type] if piece.color == chess.WHITE else -values[piece.piece_type]
        return score

    def _get_result(self, state: chess.Board, player: bool) -> float:
        if state.is_checkmate():
            winner = not state.turn
            return 1.0 if winner == player else 0.0
        # Draw or other terminal result
        return 0.5

    def _evaluate_nonterminal(self, state: chess.Board, player: bool) -> float:
        """Heuristic evaluation using material + PST, converted to win probability."""

        # Determine game phase (simple heuristic)
        queens = len(state.pieces(chess.QUEEN, chess.WHITE)) + len(state.pieces(chess.QUEEN, chess.BLACK))
        minors = len(state.pieces(chess.KNIGHT, chess.WHITE)) + len(state.pieces(chess.BISHOP, chess.WHITE)) + \
                 len(state.pieces(chess.KNIGHT, chess.BLACK)) + len(state.pieces(chess.BISHOP, chess.BLACK))
        is_endgame = queens == 0 or (queens <= 1 and minors <= 2)

        material_score = 0
        pst_score = 0

        for square in chess.SQUARES:
            piece = state.piece_at(square)
            if piece is not None:
                # Material Score
                value = piece_values[piece.piece_type]
                material_score += value if piece.color == chess.WHITE else -value

                # Piece-Square Table Score
                piece_pst = pst_tables[piece.piece_type]
                # Use endgame king table if applicable
                if piece.piece_type == chess.KING and is_endgame:
                    piece_pst = pst_king_eg

                # Get PST value - mirror index for black pieces
                pst_val = piece_pst[square] if piece.color == chess.WHITE else piece_pst[mirrored_square(square)]
                pst_score += pst_val if piece.color == chess.WHITE else -pst_val

        # Combine scores (raw centipawn value)
        total_eval_cp = material_score + pst_score

        # Convert centipawn evaluation to win probability for White using logistic function
        # K = 400 is a common value, adjust if needed based on eval scale
        prob_white = 1.0 / (1.0 + math.exp(-total_eval_cp / 400.0))

        # Return probability from perspective of 'player'
        return prob_white if player else 1.0 - prob_white 