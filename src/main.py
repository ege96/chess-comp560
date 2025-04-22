import sys
import argparse
from src.game.chess_game import Game, GameStatus
from src.engine.minimax import MinimaxEngine
from src.engine.mcts import MCTSEngine
from src.logger import setup_logger

logger = setup_logger(__name__)

def print_board(board):
    """Print the chess board in a readable format."""
    print()
    print(board)
    print()

def main():
    """Main function to run the chess game."""
    # Parse command-line arguments for engine selection and parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=['minimax','mcts'], default='minimax', help='Engine type to use')
    parser.add_argument('--depth', type=int, default=3, help='Max depth for minimax engine')
    parser.add_argument('--iterations', type=int, default=1000, help='Max iterations for MCTS engine')
    parser.add_argument('--playout', type=int, default=100, help='Max moves per playout for MCTS engine')
    args = parser.parse_args()

    # Initialize game
    game = Game()

    # Initialize selected engine
    if args.engine == 'minimax':
        engine = MinimaxEngine(max_depth=args.depth)
        engine_desc = f"Minimax with alpha-beta pruning (depth: {args.depth})"
    else:
        # Initialize MCTS engine with iteration and playout limits
        engine = MCTSEngine(max_iterations=args.iterations, max_playout_moves=args.playout)
        engine_desc = f"Monte Carlo Tree Search (iterations: {args.iterations}, playout limit: {args.playout})"

    print(f"Welcome to Chess with {args.engine.title()}Engine!")
    print(f"Engine is using {engine_desc}")
    print("You play as White, engine plays as Black")
    print("Enter moves in UCI format (e.g., 'e2e4')")
    print("Type 'quit' to exit, 'help' for commands")
    
    while not game.game_over:
        print_board(game.board)
        
        # Player's turn (White)
        print("Your move (White):")
        
        while True:
            user_input = input("> ").strip()
            
            if user_input.lower() == 'quit':
                print("Thanks for playing!")
                sys.exit(0)
            elif user_input.lower() == 'help':
                print("Commands: 'quit' to exit, 'legal' to see legal moves, 'fen' for board state")
                continue
            elif user_input.lower() == 'legal':
                print("Legal moves:", ", ".join(game.get_legal_moves()))
                continue
            elif user_input.lower() == 'fen':
                print("FEN:", game.get_fen())
                continue
            
            try:
                game.make_move(user_input)
                break
            except ValueError as e:
                print(f"Invalid move: {e}. Try again.")
        
        # Check if game ended after player's move
        if game.game_over:
            break
            
        # Engine's turn (Black)
        print("Engine is thinking...")
        engine_move = engine.get_best_move(game.board)
        print(f"Engine plays: {engine_move}")
        
        try:
            game.make_move(engine_move)
        except ValueError as e:
            print(f"Engine error: {e}")
            break
    
    # Game over
    print_board(game.board)
    status = game.get_game_status()
    
    if status == GameStatus.WHITE_WON:
        print("Congratulations! You won!")
    elif status == GameStatus.BLACK_WON:
        print("The engine won. Better luck next time!")
    elif status == GameStatus.STALEMATE:
        print("Game ended in a stalemate.")
    
    print("Final position:", game.get_fen())

if __name__ == "__main__":
    main()
