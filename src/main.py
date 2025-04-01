import sys
from src.game.chess_game import Game, GameStatus
from src.engine.minimax import MinimaxEngine
from src.logger import setup_logger

logger = setup_logger(__name__)

def print_board(board):
    """Print the chess board in a readable format."""
    print()
    print(board)
    print()

def main():
    """Main function to run the chess game."""
    # Initialize game and engine
    game = Game()
    engine_depth = 3  # Adjust based on desired difficulty
    engine = MinimaxEngine(max_depth=engine_depth)
    
    print("Welcome to Chess with MinimaxEngine!")
    print(f"Engine is using Minimax with alpha-beta pruning (depth: {engine_depth})")
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
