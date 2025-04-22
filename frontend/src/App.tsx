import React, { useState, useEffect, useCallback } from 'react';
import { Chess, Square } from 'chess.js';
import { Chessboard } from 'react-chessboard';
import './App.css';

// Types for the supported engines
export type Engine = 'minimax' | 'mcts';

// Change this via Vite env var when you deploy
const API_BASE: string = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

// Type for the backend move response
interface MoveResponse {
  message: string;
  bestMove: string | null; // Engine's move UCI, or null if game over before engine moved
  board_fen: string;
  turn: 'white' | 'black';
  game_over: boolean;
  checkmate: boolean;
  stalemate: boolean;
  insufficient_material: boolean;
  outcome: string | null; // e.g., "CHECKMATE", "STALEMATE"
  winner: 'white' | 'black' | null;
}

const ChessApp: React.FC = () => {
  const [engine, setEngine] = useState<Engine>('minimax');
  const [game] = useState(() => new Chess());
  const [fen, setFen] = useState<string>(game.fen());
  const [isThinking, setIsThinking] = useState<boolean>(false);
  const [message, setMessage] = useState<string>('You play White – make your move');

  /** Attempt to make the player move. Returns true if legal. */
  const makePlayerMove = useCallback(
    (uci: string): boolean => {
      let moveResult = false;
      try {
         moveResult = !!game.move(uci); // game.move returns Move object or null
      } catch (e) {
          // Catches illegal move format or other errors from chess.js
          console.warn("Illegal move attempt:", uci, e);
          // Optionally update message: setMessage("Illegal move: " + uci);
          return false; // Indicate move was not successful
      }

      if (moveResult) {
        const currentFen = game.fen();
        setFen(currentFen);

        // Check if player's move ended the game
        if (game.isGameOver()) {
          // const outcome = game.outcome(); // chess.js doesn't have outcome()
          let endMessage = 'Game Over.';
          // Determine winner based on whose turn it *would* be if game continued
          const winner = game.turn() === 'b' ? 'White' : 'Black'; // If it's black's turn, white delivered checkmate

          if (game.isCheckmate()) { // Checkmate check
            // endMessage = `Checkmate! ${outcome.winner === 'w' ? 'White' : 'Black'} wins.`;
            endMessage = `Checkmate! ${winner} wins.`;
          } else if (game.isStalemate()) {
            endMessage = 'Stalemate.';
          } else if (game.isInsufficientMaterial()) {
            endMessage = 'Draw by insufficient material.';
          } else if (game.isThreefoldRepetition()) {
            endMessage = 'Draw by threefold repetition.';
          } else if (game.isDraw()) {
            endMessage = 'Draw.'; // Other draw conditions
          }
          setMessage(endMessage);
          setIsThinking(false); // Don't fetch engine move if game is over
        } else {
          // Game not over, let the engine think
          setMessage('Engine is thinking...');
          setIsThinking(true);
        }
      }
      return moveResult; // Return true if the move was valid, false otherwise
    },
    [game] // Dependency array
  );

  /** Ask the backend for the engine reply after the player moves. */
  useEffect(() => {
    if (!isThinking) return;

    const fetchMove = async () => {
      try {
        const res = await fetch(`${API_BASE}/move`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fen: game.fen(), engine }),
        });
        const data: MoveResponse = await res.json(); // Use the defined interface

        // Update board only if engine made a move
        if (data.bestMove) {
           try {
              game.move(data.bestMove);
              setFen(game.fen()); // Update FEN after engine move
           } catch (e) {
               console.error("Error applying engine move:", data.bestMove, e);
               setMessage("Error applying engine move. Check backend/engine.");
               // Decide if we should stop thinking here or let finally handle it
           }
        }

        // Update message based on game state from backend response
        if (data.game_over) {
            let endMessage = 'Game Over.';
            if (data.checkmate) {
                endMessage = `Checkmate! ${data.winner === 'white' ? 'White' : 'Black'} wins.`;
            } else if (data.stalemate) {
                endMessage = 'Stalemate.';
            } else if (data.insufficient_material) {
                endMessage = 'Draw by insufficient material.';
            } else {
                // Use the generic outcome name from backend if available
                endMessage = data.outcome ? `Game Over: ${data.outcome}` : 'Game Over.';
            }
            setMessage(endMessage);
        } else if (data.bestMove) {
            // Engine moved, game not over
             setMessage(`Engine (${engine}) played ${data.bestMove}`);
        } else {
            // Engine didn't move (e.g., game was over before its turn), use backend message
             setMessage(data.message || 'Waiting for player...');
        }
      } catch (err) {
        console.error(err);
        setMessage('Engine error – check the backend');
      } finally {
        setIsThinking(false);
      }
    };

    fetchMove();
  }, [isThinking, engine, game]);

  /** react‑chessboard callback */
  const onDrop = (from: Square, to: Square): boolean => { // Use Square type from chess.js
    const uci = `${from}${to}`;
    return makePlayerMove(uci);
  };

  /** Reset the game */
  const resetGame = (): void => {
    game.reset();
    setFen(game.fen());
    setMessage('New game started – your move');
    setIsThinking(false);
  };

  return (
    <div className="app">
      <h1>Chess Engine Playground</h1>

      <div className="controls">
        <label htmlFor="engine">Engine:&nbsp;</label>
        <select
          id="engine"
          value={engine}
          onChange={(e) => setEngine(e.target.value as Engine)}
        >
          <option value="minimax">minimax</option>
          <option value="mcts">mcts (Monte Carlo Tree Search)</option>
        </select>

        <button onClick={resetGame}>Reset</button>
      </div>

      <Chessboard
        position={fen}
        onPieceDrop={onDrop}
        boardWidth={480}
        arePiecesDraggable={!isThinking && !game.isGameOver()}
      />

      <p className="message">{message}</p>
    </div>
  );
};

export default ChessApp;
