import chess

from enum import Enum
from typing import List

class GameStatus(Enum):
    """Enum for game status."""
    IN_PROGRESS = 1
    WHITE_WON = 2
    BLACK_WON = 3
    STALEMATE = 4


class Game:
    """Class representing a chess game.""" 
    
    def __init__(self):
        """Initialize a new chess game."""
        self.board: chess.Board = chess.Board()
        self.game_over: bool = False
        self.winner: GameStatus = GameStatus.IN_PROGRESS
        self.move_history: List[chess.Move] = []
        
        
    def make_move(self, move) -> None:
        """Make a move on the board."""
        move = chess.Move.from_uci(move)
        
        if move in self.board.legal_moves:
            self.board.push(move)
            self.move_history.append(move)
            
            if self.board.is_checkmate():
                self.game_over = True
                if self.board.turn:
                    self.winner = GameStatus.BLACK_WON
                else:
                    self.winner = GameStatus.WHITE_WON
                    
            elif self.board.is_stalemate():
                self.game_over = True
                self.winner = GameStatus.STALEMATE
                
        else:
            raise ValueError("Illegal move")
        
    
    def get_recent_moves(self, n: int = -1) -> List[str]:
        """Get the last n moves in UCI format."""
        if n == -1:
            n = len(self.move_history)
            
        return [move.uci() for move in self.move_history[-n:]]

    def get_fen(self) -> str:
        """Get the FEN representation of the board."""
        return self.board.fen()
    
    def get_game_status(self) -> GameStatus:
        """Get the status of the game."""
        return self.winner
    
    def get_legal_moves(self) -> List[str]:
        """Get the legal moves for the current player."""
        return [move.uci() for move in self.board.legal_moves]
    
    
