import sys
import os

# Ensure src is in the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.game.chess_game import Game, GameStatus
from src.engine.minimax import MinimaxEngine
from src.engine.bitboard import BitboardEngine
from src.logger import setup_logger

logger = setup_logger(__name__)

def print_board(board):
    """Print the chess board in a readable format."""
    print()
    print(board)
    print()

def main():
    """Main function to run the chess game."""
    # Engine selection
    print("Select engine:")
    print("1. MinimaxEngine (classic)")
    print("2. BitboardEngine (fast bitboard)")
    while True:
        engine_choice = input("Enter 1 or 2: ").strip()
        if engine_choice == '1':
            EngineClass = MinimaxEngine
            engine_name = "MinimaxEngine"
            break
        elif engine_choice == '2':
            EngineClass = BitboardEngine
            engine_name = "BitboardEngine"
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    # Initialize game and engine
    game = Game()
    engine_depth = 6  # Adjust based on desired difficulty
    engine = EngineClass(max_depth=engine_depth)
    
    print(f"Welcome to Chess with {engine_name}!")
    print(f"Engine is using {engine_name} with alpha-beta pruning (depth: {engine_depth})")
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
