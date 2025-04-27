import torch
import torch.nn as nn
import numpy as np
import chess

def board_to_tensor(board: chess.Board) -> np.ndarray:
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            idx = piece_map[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
            row, col = divmod(square, 8)
            tensor[idx, row, col] = 1
    return tensor

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.fc_value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output: evaluation
        )
        self.fc_policy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, 4672)  # 4672 is the max number of legal chess moves in python-chess
        )

    def forward(self, x):
        x = self.conv(x)
        value = self.fc_value(x)
        policy_logits = self.fc_policy(x)
        return value, policy_logits 