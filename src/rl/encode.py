import numpy as np
import chess

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode a python-chess board into a (8, 8, 12) binary numpy array.
    Planes: [white_pawns, white_knights, ..., black_kings]
    """
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    arr = np.zeros((8, 8, 12), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            plane = piece_map[piece.symbol()]
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            arr[row, col, plane] = 1
    return arr.flatten()  # Shape: (768,)
