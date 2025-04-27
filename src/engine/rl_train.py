import torch
import torch.optim as optim
import numpy as np
import chess
import random
import logging
import yaml
import math
import time
import os
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from nn_eval import ChessNet, board_to_tensor
from replay_buffer import PrioritizedReplayBuffer

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
            score, _ = model(tensor)
            score = score.item()
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
    """Maps a chess.Move to a unique index for the policy head.
    Handles promotion moves correctly by encoding the promotion piece type.
    Total indices: 64*64 (from-to squares) + 64*64*4 (possible promotions from-to-piece) ≈ 4672 max.
    """
    from_sq = move.from_square
    to_sq = move.to_square
    # Base index for non-promotion moves
    index = from_sq * 64 + to_sq
    
    # Handle promotions
    if move.promotion is not None:
        # Add offset based on promotion piece (knight=1, bishop=2, rook=3, queen=4)
        # We add a large offset to avoid collision with regular moves
        promotion_offset = {
            chess.KNIGHT: 0,
            chess.BISHOP: 1,
            chess.ROOK: 2,
            chess.QUEEN: 3
        }
        index = 64*64 + from_sq*64*4 + to_sq*4 + promotion_offset[move.promotion]
    
    return index

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

def evaluate_model(model, num_games=10, opponent_model=None, mcts_simulations=25):
    """Evaluate model performance by playing games against a baseline or random play."""
    wins, draws, losses = 0, 0, 0
    
    for game_idx in range(num_games):
        board = chess.Board()
        while not board.is_game_over():
            # Current model's turn
            if board.turn == chess.WHITE:
                move, _ = mcts_select_move(board, model, simulations=mcts_simulations)
            # Opponent's turn
            else:
                if opponent_model:
                    move, _ = mcts_select_move(board, opponent_model, simulations=mcts_simulations)
                else:
                    # Random opponent if no opponent_model
                    move = random.choice(list(board.legal_moves))
            board.push(move)
            
        # Game result
        if board.is_checkmate():
            if board.turn == chess.BLACK:  # WHITE won
                wins += 1
            else:  # BLACK won
                losses += 1
        else:  # Draw
            draws += 1
            
    return wins, draws, losses

def get_epsilon(episode, config):
    """Calculate epsilon based on annealing schedule."""
    start_epsilon = config.get('start_epsilon', 0.5)
    end_epsilon = config.get('end_epsilon', 0.05)
    decay_episodes = config.get('epsilon_decay_episodes', 5000)
    
    if episode >= decay_episodes:
        return end_epsilon
    else:
        return start_epsilon - (start_epsilon - end_epsilon) * (episode / decay_episodes)

def train():
    """Main training loop with all improvements."""
    # Create checkpoint directory
    checkpoints_dir = config.get('checkpoints_dir', 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    model = ChessNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scaler = GradScaler()  # For mixed precision training
    
    # Prioritized replay buffer
    buffer = PrioritizedReplayBuffer(
        capacity=config['replay_buffer_size'],
        alpha=config.get('per_alpha', 0.6),
        beta_start=config.get('per_beta_start', 0.4)
    )
    
    batch_size = config['batch_size']
    num_episodes = config['num_episodes']
    use_mcts = config.get('use_mcts', False)
    mcts_simulations = config.get('mcts_simulations', 50)
    
    # Evaluation settings
    eval_frequency = config.get('eval_frequency', 500)
    checkpoint_frequency = config.get('checkpoint_frequency', 1000)
    
    # Initialize baseline model (copy of initial model)
    baseline_model = ChessNet().to(DEVICE)
    baseline_model.load_state_dict(model.state_dict())
    baseline_model.eval()
    
    # Training metrics
    best_eval_score = 0
    episode_rewards = []
    value_losses = []
    policy_losses = []
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        # Epsilon annealing
        epsilon = get_epsilon(episode, config)
        
        # Update beta for PER
        buffer.update_beta(episode)
        
        # Play one episode
        states, rewards, next_states, dones, policy_targets = play_game(
            model, epsilon=epsilon, use_mcts=use_mcts, mcts_simulations=mcts_simulations)
        
        episode_rewards.append(sum(rewards))
        
        # Store experiences in replay buffer
        for s, r, ns, d, pt in zip(states, rewards, next_states, dones, policy_targets):
            buffer.push(s, r, ns, d, pt)
            
        if len(buffer) < batch_size:
            continue
            
        # Sample batch with priorities
        state_batch, reward_batch, next_state_batch, done_batch, policy_target_batch, indices, weights = buffer.sample(batch_size)
        
        state_batch = torch.tensor(np.array(state_batch)).to(DEVICE)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(DEVICE)
        next_state_batch = torch.tensor(np.array(next_state_batch)).to(DEVICE)
        done_batch = torch.tensor(done_batch, dtype=torch.float32).to(DEVICE)
        policy_target_batch = torch.tensor(np.array(policy_target_batch)).to(DEVICE)
        weights = weights.to(DEVICE)  # Importance sampling weights
        
        # Mixed precision training
        with autocast():
            # Compute targets
            with torch.no_grad():
                next_values, _ = model(next_state_batch)
                next_values = next_values.squeeze()
                targets = reward_batch + (1 - done_batch) * config['discount'] * next_values
                
            # Compute predictions
            values, policy_logits = model(state_batch)
            values = values.squeeze()
            
            # Value loss
            value_loss = torch.nn.functional.mse_loss(values, targets, reduction='none')
            weighted_value_loss = (value_loss * weights).mean()
            
            # Policy loss
            policy_loss = -torch.sum(policy_target_batch * torch.nn.functional.log_softmax(policy_logits, dim=1), dim=1)
            weighted_policy_loss = (policy_loss * weights).mean()
            
            # Combined loss
            loss = weighted_value_loss + weighted_policy_loss
            
        # Update priorities in replay buffer
        td_errors = value_loss.detach().cpu().numpy()
        buffer.update_priorities(indices, td_errors + 1e-6)  # Add small constant for stability
        
        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Log metrics
        value_losses.append(weighted_value_loss.item())
        policy_losses.append(weighted_policy_loss.item())
        
        if episode % 10 == 0:
            logger.info(f"Episode {episode}, Epsilon: {epsilon:.3f}, Value Loss: {weighted_value_loss:.4f}, "
                        f"Policy Loss: {weighted_policy_loss:.4f}, Reward: {sum(rewards):.2f}")
            
        # Periodic evaluation
        if episode > 0 and episode % eval_frequency == 0:
            wins, draws, losses = evaluate_model(model, num_games=10, opponent_model=baseline_model)
            eval_score = wins + 0.5 * draws
            logger.info(f"Evaluation after episode {episode}: Wins: {wins}, Draws: {draws}, Losses: {losses}")
            
            # Update baseline if current model is better
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                baseline_model.load_state_dict(model.state_dict())
                logger.info(f"New best model with score {eval_score}!")
                
        # Periodic checkpointing
        if episode > 0 and episode % checkpoint_frequency == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f"model_ep{episode}.pth")
            torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'value_loss': weighted_value_loss.item(),
                'policy_loss': weighted_policy_loss.item(),
                'best_eval_score': best_eval_score
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            
    # Training complete
    elapsed_time = time.time() - start_time
    logger.info(f"Training complete. Total time: {elapsed_time:.2f}s. Episodes: {num_episodes}")
    
    # Save final model
    final_path = config['model_save_path']
    torch.save(model.state_dict(), final_path)
    logger.info(f"Final model saved to {final_path}")
    
    # Final evaluation
    wins, draws, losses = evaluate_model(model, num_games=20)
    logger.info(f"Final evaluation: Wins: {wins}, Draws: {draws}, Losses: {losses}")

if __name__ == "__main__":
    train() 