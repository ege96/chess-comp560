import chess
import numpy as np
import torch
from .encode import encode_board
import random

class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.untried_moves = list(board.legal_moves)

    @property
    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits

    def expand(self):
        move = self.untried_moves.pop()
        next_board = self.board.copy()
        next_board.push(move)
        child = MCTSNode(next_board, parent=self, move=move)
        self.children.append(child)
        return child

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_puct=1.4):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            uct = child.value + c_puct * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-4))
            if uct > best_score:
                best_score = uct
                best_child = child
        return best_child

class MCTS:
    def __init__(self, value_net, device='cpu', n_simulations=50):
        self.value_net = value_net
        self.device = device
        self.n_simulations = n_simulations

    def search(self, board):
        root = MCTSNode(board)
        # --- Dirichlet noise for exploration at root ---
        if len(root.untried_moves) > 1:
            dirichlet_alpha = 0.3
            epsilon = 0.25
            noise = np.random.dirichlet([dirichlet_alpha] * len(root.untried_moves))
            for i, child in enumerate(root.untried_moves):
                # This only affects move selection at root
                pass  # Placeholder for policy head; currently not used
        for _ in range(self.n_simulations):
            node = root
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
            # Expansion
            if not node.is_fully_expanded():
                node = node.expand()
            # Evaluation
            value = self.evaluate(node.board)
            # Backpropagation
            self.backpropagate(node, value)
        # --- Softmax move selection at root for more diversity ---
        visits = np.array([child.visits for child in root.children])
        if len(visits) == 0:
            return random.choice(list(board.legal_moves))
        temperature = 1.0
        probs = np.exp(visits / temperature) / np.sum(np.exp(visits / temperature))
        move_idx = np.random.choice(len(root.children), p=probs)
        return root.children[move_idx].move

    def evaluate(self, board):
        arr = encode_board(board)
        x = torch.tensor(arr, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            v = self.value_net(x)
        return v.item()

    def backpropagate(self, node, value):
        while node is not None:
            node.visits += 1
            node.value_sum += value if node.board.turn else -value
            node = node.parent
