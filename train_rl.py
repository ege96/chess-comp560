import os
import torch
from src.rl.pipeline import RLPipeline
from src.rl.log_utils import LiveLossPlotter

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = RLPipeline(device=device, n_simulations=100)  # Increase n_simulations for stronger MCTS
    n_games = 200  # Number of self-play games
    batch_size = 64
    epochs = 2
    save_path = os.path.join('models', 'chess_value_net.pth')
    os.makedirs('models', exist_ok=True)

    # Start live loss plotting in a background thread
    plotter = LiveLossPlotter(pipeline.losses, interval=2.0)
    plotter.start()

    try:
        for i in range(n_games):
            print(f"Self-play game {i+1}/{n_games}")
            pipeline.self_play_game(max_moves=120)
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
        plotter.stop()

if __name__ == "__main__":
    main()

