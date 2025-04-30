from abc import ABC, abstractmethod
import chess
from ..logger import setup_logger, log_execution_time

logger = setup_logger(__name__)

class BaseEngine(ABC):
    """Abstract base class for chess engines."""
    
    @abstractmethod
    def get_best_move(self, board: chess.Board) -> str:
        """Get the best move for the given board."""
        pass
    
    @abstractmethod
    def get_evaluation(self, board: chess.Board) -> float:
        """Get the evaluation of the given board."""
        pass    