from abc import ABC, abstractmethod
import chess
from logger import setup_logger, log_execution_time

logger = setup_logger(__name__)

class BaseEngine(ABC):
    """Abstract base class for chess engines."""
    
    @abstractmethod
    def get_best_move(self, board: chess.Board) -> str:
        """Get the best move for the given board."""
        pass
    
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if 'get_best_move' in cls.__dict__:
            original = cls.__dict__['get_best_move']
            wrapped = log_execution_time(logger)(original)
            setattr(cls, 'get_best_move', wrapped)
    
