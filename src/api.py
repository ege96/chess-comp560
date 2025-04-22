from fastapi import FastAPI
from pydantic import BaseModel
from game.chess_game import Game       
from engine.minimax import MinimaxEngine
# (later) from src.engine.mcts import MonteCarloEngine

app = FastAPI()
minimax = MinimaxEngine(max_depth=3)

class MoveRequest(BaseModel):
    fen: str
    engine: str

@app.post("/move")
def get_move(req: MoveRequest):
    game = Game()
    game.board.set_fen(req.fen)
    if req.engine == "minimax":
        best = minimax.get_best_move(game.board)
    elif req.engine == "montecarlo":
        # placeholder until you add the engine
        best = minimax.get_best_move(game.board)
    else:
        raise ValueError("unknown engine")
    return {"bestMove": best}
