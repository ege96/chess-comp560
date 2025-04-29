import torch
import torch.optim as optim
import numpy as np
import chess
from .model import ChessValueNet
from .encode import encode_board
from .mcts import MCTS


class RLPipeline:
    def __init__(self, device=None, n_simulations=500):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = ChessValueNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.mcts = MCTS(self.model, device=self.device, n_simulations=n_simulations)
        self.memory = []  # (state, value)
        self.losses = []  # Track training loss

    def self_play_game(self, max_moves=300):
        board = chess.Board()
        game_memory = []
        for _ in range(max_moves):
            if board.is_game_over():
                break
            state = encode_board(board)
            move = self.mcts.search(board)
            board.push(move)
            game_memory.append(state)
        # Assign the final result to all states
        result = board.result()
        print(f"[Self-play] Game result: {result}")
        if result == '1-0':
            value = 1
        elif result == '0-1':
            value = -1
        else:
            value = 0
        self.memory.extend((s, value) for s in game_memory)

    def train(self, batch_size=32, epochs=1):
        if len(self.memory) < batch_size:
            return
        for _ in range(epochs):
            import random
            batch = random.sample(self.memory, batch_size)
            states = np.array([b[0] for b in batch], dtype=np.float32)
            states = torch.from_numpy(states).to(self.device)
            values = torch.tensor([b[1] for b in batch], dtype=torch.float32).to(self.device)
            preds = self.model(states)
            loss = ((preds - values) ** 2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def print_replay_buffer_stats(self, n=20):
        import collections
        values = [v for (_, v) in self.memory]
        counter = collections.Counter(values)
        print(f"[Replay buffer] Reward distribution: {dict(counter)} (total: {len(values)})")
        print(f"[Replay buffer] Example values: {values[:min(n, len(values))]}")

    def print_model_predictions(self, n=10):
        import random
        if len(self.memory) == 0:
            print("[Replay buffer] No samples to evaluate.")
            return
        samples = random.sample(self.memory, min(n, len(self.memory)))
        states = np.array([s for (s, _) in samples], dtype=np.float32)
        targets = [v for (_, v) in samples]
        states_tensor = torch.from_numpy(states).to(self.device)
        with torch.no_grad():
            preds = self.model(states_tensor).cpu().numpy()
        print("[Model predictions vs targets]:")
        for i in range(len(samples)):
            print(f"  Pred: {preds[i]: .3f} | Target: {targets[i]: .3f}")


    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
