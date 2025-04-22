import React, { useState, useEffect, useCallback } from 'react';
import { Chess } from 'chess.js';
import { Chessboard } from 'react-chessboard';
import './App.css';

// Types for the supported engines
export type Engine = 'minimax' | 'montecarlo';

// Change this via Vite env var when you deploy
const API_BASE: string = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

const ChessApp: React.FC = () => {
  const [engine, setEngine] = useState<Engine>('minimax');
  const [game] = useState(() => new Chess());
  const [fen, setFen] = useState<string>(game.fen());
  const [isThinking, setIsThinking] = useState<boolean>(false);
  const [message, setMessage] = useState<string>('You play White – make your move');

  /** Attempt to make the player move. Returns true if legal. */
  const makePlayerMove = useCallback(
    (uci: string): boolean => {
      const result = game.move(uci);
      if (result) {
        setFen(game.fen());
        setIsThinking(true);
      }
      return !!result;
    },
    [game]
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
        const data: { bestMove: string } = await res.json();
        game.move(data.bestMove);
        setFen(game.fen());
        setMessage(`Engine (${engine}) played ${data.bestMove}`);
      } catch (err) {
        console.error(err);
        setMessage('Engine error – check the backend');
      } finally {
        setIsThinking(false);
      }
    };

    fetchMove();
  }, [isThinking, engine, game]);

  /** react‑chessboard callback */
  const onDrop = (from: string, to: string): boolean => {
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
          <option value="montecarlo" disabled>
            montecarlo (coming soon)
          </option>
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
