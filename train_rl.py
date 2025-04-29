import os
import torch
from src.rl.pipeline import RLPipeline
from src.rl.log_utils import LiveLossPlotter
import torch.multiprocessing as mp


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = RLPipeline(device=device, n_simulations=100)  # Increase n_simulations for stronger MCTS
    n_games = 200  # Number of self-play games
    batch_size = 64
    epochs = 2
    save_path = os.path.join('models', 'chess_value_net.pth')
    os.makedirs('models', exist_ok=True)


    mp.set_start_method('spawn', force=True)


    num_parallel_games = min(8, mp.cpu_count())  # Use up to 8 or number of cores
    try:
        for i in range(n_games):
            print(f"Self-play batch {i+1}/{n_games} (parallel games: {num_parallel_games})")
            # Prepare args for each parallel game
            model_state_dict = pipeline.model.state_dict()
            args = [(model_state_dict, pipeline.device, 100, 120) for _ in range(num_parallel_games)]
            with mp.Pool(processes=num_parallel_games) as pool:
                results = pool.map(RLPipeline.self_play_game_static, args)
            # Flatten and add to replay buffer
            for game in results:
                pipeline.memory.extend(game)
            pipeline.train(batch_size=batch_size, epochs=epochs)
            if (i+1) % 10 == 0:
                pipeline.save(save_path)
                print(f"[Checkpoint] Saved model to {save_path}")
            if (i+1) % 5 == 0:
                pipeline.print_replay_buffer_stats()
                pipeline.print_model_predictions()

        pipeline.save(save_path)
        print(f"[Done] Final model saved to {save_path}")
    finally:
        pass
        # plotter.stop()

if __name__ == "__main__":
    main()

