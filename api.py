from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Import CORS middleware
import uvicorn
import chess
import logging
import sys
import os
from pydantic import BaseModel # Import BaseModel

# Add src directory to Python path to allow engine import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.engine.minimax import MinimaxEngine
from src.engine.mcts import MCTSEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chess Engine API",
    description="API to interact with the Python Chess engine.",
    version="0.1.0",
)

# CORS Configuration
origins = [
    "http://localhost",
    "http://localhost:3000",  # Common React dev port
    "http://localhost:5173",  # Common Vite dev port
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    # Add other origins if needed (e.g., your deployed frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"], # Allow GET, POST and preflight OPTIONS
    allow_headers=["Content-Type", "Authorization"], # Allow common headers
)

# In-memory storage for the game state (replace with a proper game management system if needed)
# For simplicity, we'll manage a single global game board.
board = chess.Board()

# Instantiate engines with default parameters
minimax_engine = MinimaxEngine(max_depth=3)
mcts_engine = MCTSEngine()

# Define request body model
class MoveRequest(BaseModel):
    fen: str
    engine: str # Although frontend only has one, keep for potential future use

@app.get("/")
async def read_root():
    """Root endpoint providing basic information about the API."""
    return {"message": "Welcome to the Chess Engine API!"}

@app.post("/new_game", status_code=201)
async def new_game():
    """Starts a new game by resetting the board."""
    global board
    board.reset()
    logger.info("New game started. Board reset.")
    return {"message": "New game started.", "board_fen": board.fen()}

@app.get("/board")
async def get_board_state():
    """Returns the current state of the board in FEN format."""
    return {"board_fen": board.fen(), "turn": "white" if board.turn == chess.WHITE else "black"}

@app.get("/possible_moves/{square}")
async def get_possible_moves(square: str):
    """Returns a list of legal moves for the piece on the given square."""
    try:
        sq = chess.parse_square(square.lower())
        piece = board.piece_at(sq)
        if not piece:
            raise HTTPException(status_code=404, detail=f"No piece found at square {square}")

        legal_moves = [move.uci() for move in board.legal_moves if move.from_square == sq]
        if not legal_moves:
            return {"message": f"No legal moves for the piece at {square}", "moves": []}
        return {"square": square, "piece": piece.symbol(), "moves": legal_moves}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid square format: {square}. Use algebraic notation (e.g., 'e2').")
    except Exception as e:
        logger.error(f"Error getting possible moves for {square}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/move")
async def make_engine_move(request: MoveRequest):
    """Receives the board FEN after player's move, calculates and makes the engine's move."""
    global board
    try:
        # Set the board to the state *after* the player's move
        board.set_fen(request.fen)
        logger.info(f"Received FEN from frontend: {request.fen}. It's { 'White' if board.turn == chess.WHITE else 'Black'}'s turn.")

        # Check if game was already over after player's move
        game_over = board.is_game_over()
        outcome = board.outcome() if game_over else None
        if game_over:
             logger.info(f"Game already over after player move. Outcome: {outcome}")
             # Return status without making an engine move
             return {
                 "message": "Game over.",
                 "bestMove": None, # No move made by engine
                 "board_fen": board.fen(),
                 "turn": "white" if board.turn == chess.WHITE else "black",
                 "game_over": game_over,
                 "checkmate": board.is_checkmate(),
                 "stalemate": board.is_stalemate(),
                 "insufficient_material": board.is_insufficient_material(),
                 "outcome": outcome.termination.name if outcome else None,
                 "winner": "white" if outcome and outcome.winner == chess.WHITE else ("black" if outcome and outcome.winner == chess.BLACK else None)
             }

        # It's engine's turn. Select engine based on request.engine
        if request.engine.lower() == 'mcts':
            engine_to_use = mcts_engine
        elif request.engine.lower() == 'minimax':
            engine_to_use = minimax_engine
        else:
            raise HTTPException(status_code=400, detail=f"Unknown engine: {request.engine}")
        logger.info(f"Calculating engine move for {'White' if board.turn == chess.WHITE else 'Black'} using {request.engine} engine")
        best_move_uci = engine_to_use.get_best_move(board)
        logger.info(f"Engine ({request.engine}) calculated move: {best_move_uci}")

        if not best_move_uci or best_move_uci == "0000":
             # Should not happen unless engine fails or game is already over
             # (which we checked) - handle defensively
             logger.warning("Engine returned no valid move.")
             # Re-check game state just in case
             game_over = board.is_game_over()
             outcome = board.outcome() if game_over else None
             return {
                 "message": "Engine could not find a move. Game might be over.",
                 "bestMove": None,
                 "board_fen": board.fen(),
                 "turn": "white" if board.turn == chess.WHITE else "black",
                 "game_over": game_over,
                 "checkmate": board.is_checkmate(),
                 "stalemate": board.is_stalemate(),
                 "insufficient_material": board.is_insufficient_material(),
                 "outcome": outcome.termination.name if outcome else None,
                 "winner": "white" if outcome and outcome.winner == chess.WHITE else ("black" if outcome and outcome.winner == chess.BLACK else None)
             }

        # Validate and make the engine's move
        move = chess.Move.from_uci(best_move_uci)
        if move in board.legal_moves:
            board.push(move)
            logger.info(f"Engine move {best_move_uci} made. New FEN: {board.fen()}")
            game_over = board.is_game_over()
            outcome = None
            if game_over:
                outcome = board.outcome()
                logger.info(f"Game over after engine move. Outcome: {outcome}")

            # Return the move made by the engine and the new state
            return {
                "message": f"Engine ({request.engine}) played {best_move_uci}.",
                "bestMove": best_move_uci, # Send back the move made by engine
                "board_fen": board.fen(),
                "turn": "white" if board.turn == chess.WHITE else "black",
                "game_over": game_over,
                "checkmate": board.is_checkmate(),
                "stalemate": board.is_stalemate(),
                "insufficient_material": board.is_insufficient_material(),
                "outcome": outcome.termination.name if outcome else None,
                "winner": "white" if outcome and outcome.winner == chess.WHITE else ("black" if outcome and outcome.winner == chess.BLACK else None)
            }
        else:
            # This case indicates an error in the engine logic or board state
            logger.error(f"Engine generated illegal move: {best_move_uci} from FEN {request.fen}")
            legal_moves_uci = [m.uci() for m in board.legal_moves]
            raise HTTPException(status_code=500, detail=f"Engine generated illegal move: {best_move_uci}. Legal moves: {legal_moves_uci}")

    except chess.InvalidFenError:
         logger.error(f"Invalid FEN received from frontend: {request.fen}")
         raise HTTPException(status_code=400, detail=f"Invalid FEN string received: {request.fen}")
    except ValueError as ve:
        # Handle potential errors from Move.from_uci if engine returns bad string
        logger.error(f"Error processing engine move {best_move_uci}: {ve}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Invalid move format from engine: {best_move_uci}")
    except Exception as e:
        logger.error(f"Error during engine move calculation for FEN {request.fen}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during engine move")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 