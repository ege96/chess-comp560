import torch
import torch.optim as optim
import numpy as np
import chess
import random
from nn_eval import ChessNet, board_to_tensor
from replay_buffer import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

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

def play_game(model, epsilon=0.1):
    board = chess.Board()
    states, rewards, next_states, dones = [], [], [], []
    while not board.is_game_over():
        state = board_to_tensor(board)
        move = select_move(board, model, epsilon)
        board.push(move)
        reward = -0.01  # Small penalty for each move (reward shaping)
        if board.is_checkmate():
            reward = 1 if board.turn == chess.BLACK else -1  # Because turn flips after move
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            reward = 0
        next_state = board_to_tensor(board)
        done = board.is_game_over()
        states.append(state)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
    print(f"Game rewards: {rewards}")  # Debug: print rewards for each game
    return states, rewards, next_states, dones

def train():
    model = ChessNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    buffer = ReplayBuffer(10000)
    batch_size = 64
    num_episodes = 10000

    for episode in range(num_episodes):
        states, rewards, next_states, dones = play_game(model, epsilon=0.2)
        for s, r, ns, d in zip(states, rewards, next_states, dones):
            buffer.push(s, r, ns, d)

        if len(buffer) < batch_size:
            continue

        # Sample batch
        state_batch, reward_batch, next_state_batch, done_batch = buffer.sample(batch_size)
        state_batch = torch.tensor(np.array(state_batch)).to(DEVICE)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(DEVICE)
        next_state_batch = torch.tensor(np.array(next_state_batch)).to(DEVICE)
        done_batch = torch.tensor(done_batch, dtype=torch.float32).to(DEVICE)

        # Compute targets
        with torch.no_grad():
            next_values = model(next_state_batch).squeeze()
            targets = reward_batch + (1 - done_batch) * 0.99 * next_values

        # Compute predictions
        values = model(state_batch).squeeze()
        loss = torch.nn.functional.mse_loss(values, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode}, Loss: {loss.item():.4f}")  # Print every batch

    torch.save(model.state_dict(), "chess_rl_model.pth")

if __name__ == "__main__":
    train() 