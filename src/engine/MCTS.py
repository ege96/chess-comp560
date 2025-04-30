from .BaseEngine import BaseEngine
import chess
import random
import math
import time
from typing import Optional, Dict
from .minimax import MinimaxEngine

class MCTSNode:
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, move: Optional[chess.Move] = None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_moves = list(board.legal_moves)
        self.is_terminal = board.is_game_over()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.4):
        # UCB1 formula
        return max(self.children, key=lambda child: child.value / (child.visits + 1e-6) + c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6)))

    def expand(self):
        move = self.untried_moves.pop()
        next_board = self.board.copy()
        next_board.push(move)
        child_node = MCTSNode(next_board, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        self.visits += 1
        self.value += reward

class MCTSEngine(BaseEngine):
    def __init__(self, n_simulations: int = 1000, rollout_depth: int = 10, time_limit: float = 1.0):
        self.n_simulations = n_simulations
        self.rollout_depth = rollout_depth
        self.time_limit = time_limit  # seconds per move
        self.eval_engine = MinimaxEngine(max_depth=2)  # Use shallow depth for speed

    def get_best_move(self, board: chess.Board) -> str:
        root = MCTSNode(board)
        end_time = time.time() + self.time_limit
        simulations = 0
        while simulations < self.n_simulations and time.time() < end_time:
            node = root
            # Selection
            while not node.is_terminal and node.is_fully_expanded():
                node = node.best_child()
            # Expansion
            if not node.is_terminal and not node.is_fully_expanded():
                node = node.expand()
            # Simulation
            reward = self.rollout(node.board)
            # Backpropagation
            while node:
                node.update(reward if node.board.turn != board.turn else -reward)
                node = node.parent
            simulations += 1
        # Pick most visited child
        if not root.children:
            # Defensive: fallback to first legal move
            moves = list(board.legal_moves)
            return moves[0].uci() if moves else "0000"
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move.uci()

    def rollout(self, board: chess.Board) -> float:
        # Fast playout: random moves up to rollout_depth or until game over
        rollout_board = board.copy()
        for _ in range(self.rollout_depth):
            if rollout_board.is_game_over():
                break
            moves = list(rollout_board.legal_moves)
            move = random.choice(moves)
            rollout_board.push(move)
        # Use MinimaxEngine's evaluation at the end of the rollout
        eval_score = self.eval_engine.get_evaluation(rollout_board)
        # Normalize: positive means advantage for root's player, negative means disadvantage
        # If it's the same player's turn as at root, use as is; else, flip sign
        return eval_score if rollout_board.turn == board.turn else -eval_score

    def material_eval(self, board: chess.Board) -> float:
        # Simple material count, white positive
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        white = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in piece_values.items())
        black = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in piece_values.items())
        return white - black

    def get_evaluation(self, board: chess.Board) -> float:
        # Use material as a quick evaluation
        return self.material_eval(board)
