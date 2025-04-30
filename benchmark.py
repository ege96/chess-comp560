import chess
import chess.engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.engine.minimax import MinimaxEngine
from src.engine.MCTS import MCTSEngine
import multiprocessing as mp

# Path to your Stockfish binary
STOCKFISH_PATH = "stockfish.exe"

# Settings
ELOS = [1320, 1600, 2000, 2400]
GAMES_PER_SETTING = 5  # Number of games per (engine, elo)
DEPTH = 3  # Search depth for both Minimax and Stockfish


def play_match(engine, stockfish_elo, num_games):
    """
    Play num_games between our engine and Stockfish at given elo.
    Alternates colors each game. Returns (wins, draws, losses) from engine's perspective.
    """
    print(f"[INFO] Starting {num_games} games vs Stockfish ELO {stockfish_elo}...")
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
    wins = draws = losses = 0

    for i in range(num_games):
        board = chess.Board()
        engine_white = (i % 2 == 0)
        print(f"[MATCH] Game {i+1}/{num_games} | Engine as {'White' if engine_white else 'Black'} vs Stockfish({stockfish_elo})")
        move_num = 1
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                if engine_white:
                    move_uci = engine.get_best_move(board)
                else:
                    result = sf.play(board, chess.engine.Limit(depth=DEPTH))
                    move_uci = result.move.uci()
            else:
                if not engine_white:
                    move_uci = engine.get_best_move(board)
                else:
                    result = sf.play(board, chess.engine.Limit(depth=DEPTH))
                    move_uci = result.move.uci()
            board.push(chess.Move.from_uci(move_uci))
            move_num += 1
        outcome = board.outcome()
        if outcome.winner is None:
            draws += 1
            print(f"[RESULT] Draw | Moves: {move_num-1}")
        else:
            engine_won = (outcome.winner == chess.WHITE and engine_white) or \
                         (outcome.winner == chess.BLACK and not engine_white)
            if engine_won:
                wins += 1
                print(f"[RESULT] Engine win | Moves: {move_num-1}")
            else:
                losses += 1
                print(f"[RESULT] Engine loss | Moves: {move_num-1}")

    sf.quit()
    print(f"[INFO] Finished {num_games} games vs Stockfish ELO {stockfish_elo}. W: {wins} D: {draws} L: {losses}")
    return wins, draws, losses


def run_task(params):
    engine_name, EngineClass, kwargs, elo = params
    print(f"[TASK] Starting task: Engine={engine_name}, ELO={elo}")
    engine = EngineClass(**kwargs)
    wins, draws, losses = play_match(engine, elo, GAMES_PER_SETTING)
    print(f"[TASK] Finished task: Engine={engine_name}, ELO={elo} | W: {wins} D: {draws} L: {losses}")
    return {
        "engine": engine_name,
        "elo": elo,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "games": GAMES_PER_SETTING,
    }


if __name__ == '__main__':
    # Build parameter list for multiprocessing
    tasks = []
    tasks += [("minimax", MinimaxEngine, {"max_depth": DEPTH}, elo) for elo in ELOS]
    tasks += [("mcts", MCTSEngine, {"n_simulations": 500, "rollout_depth": 10, "time_limit": 1.0}, elo) for elo in ELOS]

    print(f"[INFO] Created {len(tasks)} tasks: ")
    for t in tasks:
        print(f"  Engine: {t[0]}, ELO: {t[3]}")

    with mp.Pool() as pool:
        print("[INFO] Starting multiprocessing pool...")
        results = pool.map(run_task, tasks)
        print("[INFO] All tasks completed.")

    df = pd.DataFrame(results)
    # Compute win rate and 95% CI
    z = 1.96
    df['win_rate'] = df['wins'] / df['games']
    df['ci'] = z * np.sqrt(df['win_rate'] * (1 - df['win_rate']) / df['games'])

    print("\n[SUMMARY] Results DataFrame:")
    print(df)

    # Plot results with error bars
    plt.figure()
    for engine_name in df['engine'].unique():
        sub = df[df['engine'] == engine_name]
        plt.errorbar(sub['elo'], sub['win_rate'], yerr=sub['ci'], label=engine_name, marker='o')
    plt.xlabel('Stockfish ELO')
    plt.ylabel('Engine Win Rate')
    plt.title('Engine Performance vs Stockfish Strength')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
