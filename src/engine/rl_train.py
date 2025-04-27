import torch
import torch.optim as optim
import numpy as np
import chess
import random
import logging
import yaml
from nn_eval import ChessNet, board_to_tensor
from replay_buffer import ReplayBuffer
import math

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logging
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler = logging.FileHandler(config['log_file'])
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []  # Remove any default handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"Using device: {DEVICE}")
logger.info(f"Config: {config}")

def select_move(board, model, epsilon=0.1):
    legal_moves = list(board.legal_moves)
    if random.random() < epsilon:
        return random.choice(legal_moves)
    # Evaluate all moves, pick the best
    best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
    best_move = None
    for move in legal_moves:
        board.push(move)
        tensor = torch.tensor(board_to_tensor(board)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            score = model(tensor).item()
        board.pop()
        if (board.turn == chess.WHITE and score > best_score) or (board.turn == chess.BLACK and score < best_score):
            best_score = score
            best_move = move
    return best_move if best_move else random.choice(legal_moves)

def mcts_select_move(board, model, simulations=50):
    """AlphaZero-style MCTS with batched leaf evaluation."""
    class Node:
        def __init__(self, board, parent=None, move=None):
            self.board = board.copy()
            self.parent = parent
            self.move = move
            self.children = {}
            self.N = 0  # Visit count
            self.W = 0  # Total value
            self.Q = 0  # Mean value
            self.P = None  # Prior probability
            self.is_expanded = False
            self.value = None

    root = Node(board)
    tensor = torch.tensor(board_to_tensor(board)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, policy_logits = model(tensor)
        policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
    legal_moves = list(board.legal_moves)
    move_indices = [move_to_index(m) for m in legal_moves]
    policy_probs = {m: policy[idx] for m, idx in zip(legal_moves, move_indices)}
    for move in legal_moves:
        child_board = board.copy()
        child_board.push(move)
        child = Node(child_board, parent=root, move=move)
        child.P = policy_probs[move]
        root.children[move] = child

    # For batching
    leaf_nodes = []
    search_paths = []

    for _ in range(simulations):
        node = root
        path = [node]
        # Selection
        while node.children:
            max_ucb = -float('inf')
            best_move = None
            for move, child in node.children.items():
                score = child.Q + 1.0 * child.P * math.sqrt(root.N) / (1 + child.N)
                if score > max_ucb:
                    max_ucb = score
                    best_move = move
            node = node.children[best_move]
            path.append(node)
        # Expansion
        if not node.is_expanded and not node.board.is_game_over():
            leaf_nodes.append(node)
            search_paths.append(path)
        else:
            # Terminal node, backup immediately
            value = 0
            if node.board.is_checkmate():
                value = 1 if node.board.turn == chess.BLACK else -1
            for n in reversed(path):
                n.N += 1
                n.W += value
                n.Q = n.W / n.N
                value = -value

    # Batched evaluation of all leaves
    if leaf_nodes:
        leaf_tensors = torch.stack([torch.tensor(board_to_tensor(n.board)).to(DEVICE) for n in leaf_nodes])
        with torch.no_grad():
            values, policy_logits = model(leaf_tensors)
            values = values.squeeze().cpu().numpy()
            policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
        for i, node in enumerate(leaf_nodes):
            node.value = values[i]
            node.is_expanded = True
            # Expand children
            legal_moves = list(node.board.legal_moves)
            move_indices = [move_to_index(m) for m in legal_moves]
            policy_probs = {m: policies[i][idx] for m, idx in zip(legal_moves, move_indices)}
            for move in legal_moves:
                child_board = node.board.copy()
                child_board.push(move)
                child = Node(child_board, parent=node, move=move)
                child.P = policy_probs[move]
                node.children[move] = child
        # Backup
        for i, path in enumerate(search_paths):
            value = leaf_nodes[i].value
            for n in reversed(path):
                n.N += 1
                n.W += value
                n.Q = n.W / n.N
                value = -value

    # Visit count distribution for all legal moves
    visits = np.zeros(4672, dtype=np.float32)
    for move in root.children:
        idx = move_to_index(move)
        visits[idx] = root.children[move].N
    if visits.sum() > 0:
        visits /= visits.sum()
    best_move = max(root.children.items(), key=lambda item: item[1].N)[0]
    return best_move, visits

# Helper: map move to index for policy head
from chess import Move

def move_to_index(move):
    # python-chess move indexing: https://github.com/niklasf/python-chess/blob/master/docs/square.rst
    # We'll use a simple mapping: from_square * 73 + to_square (max 4672 moves)
    # This is not perfect but works for most cases
    return move.from_square * 73 + move.to_square

def play_game(model, epsilon=0.1, use_mcts=False, mcts_simulations=50):
    board = chess.Board()
    states, rewards, next_states, dones, policy_targets = [], [], [], [], []
    while not board.is_game_over():
        state = board_to_tensor(board)
        if use_mcts:
            move, policy_target = mcts_select_move(board, model, simulations=mcts_simulations)
        else:
            move = select_move(board, model, epsilon)
            # Uniform random policy for non-MCTS
            policy_target = np.zeros(4672, dtype=np.float32)
            legal_moves = list(board.legal_moves)
            for m in legal_moves:
                policy_target[move_to_index(m)] = 1.0
            policy_target /= policy_target.sum()
        board.push(move)
        reward = config['move_penalty']
        if board.is_checkmate():
            reward = 1 if board.turn == chess.BLACK else -1
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            reward = 0
        next_state = board_to_tensor(board)
        done = board.is_game_over()
        states.append(state)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        policy_targets.append(policy_target)
    logger.info(f"Game rewards: {rewards}")
    return states, rewards, next_states, dones, policy_targets

def train():
    model = ChessNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    buffer = ReplayBuffer(config['replay_buffer_size'])
    batch_size = config['batch_size']
    num_episodes = config['num_episodes']
    use_mcts = config.get('use_mcts', False)
    mcts_simulations = config.get('mcts_simulations', 50)

    for episode in range(num_episodes):
        states, rewards, next_states, dones, policy_targets = play_game(
            model, epsilon=config['epsilon'], use_mcts=use_mcts, mcts_simulations=mcts_simulations)
        for s, r, ns, d, pt in zip(states, rewards, next_states, dones, policy_targets):
            buffer.push(s, r, ns, d)

        if len(buffer) < batch_size:
            continue

        # Sample batch
        state_batch, reward_batch, next_state_batch, done_batch, policy_target_batch = buffer.sample(batch_size)
        state_batch = torch.tensor(np.array(state_batch)).to(DEVICE)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(DEVICE)
        next_state_batch = torch.tensor(np.array(next_state_batch)).to(DEVICE)
        done_batch = torch.tensor(done_batch, dtype=torch.float32).to(DEVICE)
        policy_target_batch = torch.tensor(np.array(policy_target_batch)).to(DEVICE)

        # Compute targets
        with torch.no_grad():
            next_values, _ = model(next_state_batch)
            next_values = next_values.squeeze()
            targets = reward_batch + (1 - done_batch) * config['discount'] * next_values

        # Compute predictions
        values, policy_logits = model(state_batch)
        values = values.squeeze()
        # Value loss
        value_loss = torch.nn.functional.mse_loss(values, targets)
        # Policy loss (cross-entropy with visit distribution)
        policy_loss = -torch.sum(policy_target_batch * torch.nn.functional.log_softmax(policy_logits, dim=1), dim=1).mean()
        loss = value_loss + policy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.info(f"Episode {episode}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), config['model_save_path'])
    logger.info(f"Model saved to {config['model_save_path']}")

if __name__ == "__main__":
    train() 