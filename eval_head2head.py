import chess
from src.engine.minimax import MinimaxEngine

def play_game(engine1, engine2, max_moves=120):
    board = chess.Board()
    engines = [engine1, engine2]
    move_num = 0
    while not board.is_game_over() and move_num < max_moves:
        engine = engines[move_num % 2]
        move = engine.get_best_move(board)
        board.push(chess.Move.from_uci(move))
        move_num += 1
    return board.result()

def evaluate_models(n_games=20, depth=3, rl_model_path='models/chess_value_net.pth'):
    engine_heur = MinimaxEngine(max_depth=depth, use_rl=False)
    engine_rl = MinimaxEngine(max_depth=depth, use_rl=True, rl_model_path=rl_model_path)
    results = {"rl_win":0, "heur_win":0, "draw":0}
    for i in range(n_games):
        if i % 2 == 0:
            res = play_game(engine_rl, engine_heur)
            if res == "1-0": results["rl_win"] += 1
            elif res == "0-1": results["heur_win"] += 1
            else: results["draw"] += 1
        else:
            res = play_game(engine_heur, engine_rl)
            if res == "1-0": results["heur_win"] += 1
            elif res == "0-1": results["rl_win"] += 1
            else: results["draw"] += 1
        print(f"Game {i+1}/{n_games} result: {res}")
    print("\nFinal Results:")
    print(f"RL wins:   {results['rl_win']}")
    print(f"Heur wins: {results['heur_win']}")
    print(f"Draws:     {results['draw']}")

if __name__ == "__main__":
    evaluate_models()
