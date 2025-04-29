import torch
import torch.optim as optim
import numpy as np
import chess
from .model import ChessValueNet
from .encode import encode_board
from .mcts import MCTS


class RLPipeline:
    def __init__(self, device=None, n_simulations=100):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = ChessValueNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.mcts = MCTS(self.model, device=self.device, n_simulations=n_simulations)
        self.memory = []  # (state, value)
        self.losses = []  # Track training loss

    def self_play_game(self, max_moves=300):
        from .mcts import move_to_index
        board = chess.Board()
        game_memory = []
        move_count = 0
        halfmove_clock = 0
        for _ in range(max_moves):
            if board.is_game_over():
                break
            state = encode_board(board)
            # --- AlphaZero-style opening randomization ---
            if move_count < 5:
                move = np.random.choice(list(board.legal_moves))
                policy_target = np.zeros(4672, dtype=np.float32)
                policy_target[move_to_index(move)] = 1.0
            else:
                temperature = 1.0 if move_count < 20 else 0.1
                move, policy_target = self.mcts.search(board, temperature=temperature, return_policy_target=True)
            board.push(move)
            game_memory.append((state, None, policy_target))  # Value assigned later
            move_count += 1
            if board.halfmove_clock >= 100:
                print("[Self-play] 50-move rule triggered, ending game as draw.")
                break
        result = board.result()
        print(f"[Self-play] Game result: {result}")
        if result == '1-0':
            value = 1
        elif result == '0-1':
            value = -1
        else:
            value = 0
        self.memory.extend((s, value, p) for (s, _, p) in game_memory)

    @staticmethod
    def self_play_game_static(args):
        """
        Static method for parallel self-play (for use with multiprocessing.Pool).
        args: (model_state_dict, device, n_simulations, max_moves)
        """
        import torch
        from .model import ChessValueNet
        from .mcts import MCTS, move_to_index
        from .encode import encode_board
        import numpy as np
        model_state_dict, device, n_simulations, max_moves = args
        model = ChessValueNet().to(device)
        model.load_state_dict(model_state_dict)
        mcts = MCTS(model, device=device, n_simulations=n_simulations)
        board = chess.Board()
        game_memory = []
        move_count = 0
        for _ in range(max_moves):
            if board.is_game_over():
                break
            state = encode_board(board)
            if move_count < 5:
                move = np.random.choice(list(board.legal_moves))
                policy_target_distribution = np.zeros(4672, dtype=np.float32)
                policy_target_distribution[move_to_index(move)] = 1.0
            else:
                temperature = 1.0 if move_count < 20 else 0.1
                move, policy_target_distribution = mcts.search(board, temperature=temperature, return_policy_target=True)
            board.push(move)
            game_memory.append((state, None, policy_target_distribution))
            move_count += 1
            if board.halfmove_clock >= 100:
                break
        result = board.result()
        if result == '1-0':
            value = 1
        elif result == '0-1':
            value = -1
        else:
            value = 0
        return [(s, value, p) for (s, _, p) in game_memory]

    def train(self, batch_size=32, epochs=1):
        if len(self.memory) < batch_size:
            return
        for _ in range(epochs):
            import random
            batch = random.sample(self.memory, batch_size)
            states = np.array([b[0] for b in batch], dtype=np.float32)
            states = torch.from_numpy(states).to(self.device)
            values = torch.tensor([b[1] for b in batch], dtype=torch.float32).to(self.device)

            # KL-divergence policy loss with visit count distributions
            if len(batch[0]) == 3:
                policy_targets = torch.from_numpy(np.stack([b[2] for b in batch])).float().to(self.device)  # [batch, 4672]
            else:
                policy_targets = None

            policy_logits, value_preds = self.model(states)
            value_loss = ((value_preds - values) ** 2).mean()

            if policy_targets is not None:
                # Use log_softmax for numerical stability
                log_probs = torch.nn.functional.log_softmax(policy_logits, dim=1)
                # KL-divergence: D_KL(target || predicted)
                # torch.nn.functional.kl_div expects input as log-probs, target as probs
                policy_loss = torch.nn.functional.kl_div(log_probs, policy_targets, reduction='batchmean')
                loss = value_loss + policy_loss
            else:
                loss = value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def print_replay_buffer_stats(self, n=20):
        import collections
        values = [t[1] for t in self.memory]
        counter = collections.Counter(values)
        print(f"[Replay buffer] Reward distribution: {dict(counter)} (total: {len(values)})")
        print(f"[Replay buffer] Example values: {values[:min(n, len(values))]}")

    def print_model_predictions(self, n=10):
        import random
        if len(self.memory) == 0:
            print("[Replay buffer] No samples to evaluate.")
            return
        samples = random.sample(self.memory, min(n, len(self.memory)))
        states = np.array([t[0] for t in samples], dtype=np.float32)
        targets = [t[1] for t in samples]
        states_tensor = torch.from_numpy(states).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.model(states_tensor)
            preds = value.cpu().numpy()
        print("[Model predictions vs targets]:")
        for i in range(len(samples)):
            print(f"  Pred: {preds[i]: .3f} | Target: {targets[i]: .3f}")


    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
