import chess
import numpy as np
import torch
from .encode import encode_board
import random

# Map chess.Move to a unique index for policy head
# AlphaZero uses 8x8x73 encoding, but for simplicity, use python-chess move.uci() to int
move_index_map = {}
reverse_move_index_map = {}
def move_to_index(move):
    # Use hash of UCI string, mod policy head size (4672)
    idx = abs(hash(move.uci())) % 4672
    move_index_map[move] = idx
    reverse_move_index_map[idx] = move
    return idx

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

    def best_child(self, c_puct=1.4, priors=None):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            prior = 1.0
            if priors is not None and hasattr(self, 'move_priors'):
                prior = self.move_priors.get(child.move, 1e-8)
            uct = child.value + c_puct * prior * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-4))
            if uct > best_score:
                best_score = uct
                best_child = child
        return best_child


class MCTS:
    def __init__(self, value_net, device='cpu', n_simulations=50):
        self.value_net = value_net
        self.device = device
        self.n_simulations = n_simulations

    def search(self, board, temperature=1.0, return_policy_target=False):
        root = MCTSNode(board)
        legal_moves = list(board.legal_moves)
        arr = encode_board(board)
        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, _ = self.value_net(x)
        policy_logits = policy_logits.squeeze(0).cpu().numpy()
        move_indices = [move_to_index(m) for m in legal_moves]
        priors = np.exp(policy_logits[move_indices] - np.max(policy_logits[move_indices]))
        priors /= np.sum(priors)
        if len(legal_moves) > 1:
            dirichlet_alpha = 0.3
            epsilon = 0.25
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
            priors = (1 - epsilon) * priors + epsilon * noise
        root.move_priors = dict(zip(legal_moves, priors))
        for _ in range(self.n_simulations):
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.best_child(priors=root.move_priors)
            if not node.is_fully_expanded():
                node = node.expand()
            value = self.evaluate(node.board)
            self.backpropagate(node, value)
        visits = np.array([child.visits for child in root.children])
        if len(visits) == 0:
            move = random.choice(list(board.legal_moves))
            if return_policy_target:
                visit_dist = np.zeros(4672, dtype=np.float32)
                visit_dist[move_to_index(move)] = 1.0
                return move, visit_dist
            return move
        # Temperature-based move selection
        probs = np.exp(np.log(visits + 1e-8) / temperature)
        probs /= np.sum(probs)
        move_idx = np.random.choice(len(root.children), p=probs)
        move = root.children[move_idx].move
        # --- AlphaZero-style visit count distribution as policy target ---
        visit_dist = np.zeros(4672, dtype=np.float32)
        for child, v in zip(root.children, visits):
            idx = move_to_index(child.move)
            visit_dist[idx] = v
        if visit_dist.sum() > 0:
            visit_dist /= visit_dist.sum()
        if return_policy_target:
            return move, visit_dist
        return move

    def evaluate(self, board):
        arr = encode_board(board)
        x = torch.tensor(arr, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, value = self.value_net(x)
        return value.item()

    def backpropagate(self, node, value):
        while node is not None:
            node.visits += 1
            node.value_sum += value if node.board.turn else -value
            node = node.parent
