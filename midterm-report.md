# COMP 560 Final Project Midterm Report

## Team

- **Alex Tang** (PID: 730676334)
- **Eric Ge** (PID: 730662573)
- **Guru Balamurugan** (PID: 730617829)
- **Kyler Chen** (PID: 730564343)

## Abstract

This project focuses on developing a chess engine utilizing the Minimax algorithm with Alpha-Beta pruning to improve computational efficiency. A heuristic evaluation function is designed to guide the engine's decision-making process, incorporating material balance, piece positioning, and space control. The initial implementation will rely on a manually calculated heuristic, but future iterations will incorporate reinforcement learning techniques to optimize the heuristic dynamically. Our goal is to develop a robust chess engine capable of strategic gameplay, eventually improving beyond traditional static heuristics through machine learning-based enhancements.

# Introduction

Chess has long served as a benchmark for artificial intelligence research due to its well-defined rules and vast search space. Traditional chess engines leverage brute-force search techniques enhanced by heuristics to evaluate board states efficiently. The Minimax algorithm, in conjunction with Alpha-Beta pruning, remains a foundational approach to reducing computational complexity while ensuring optimal decision-making in adversarial environments.

This project aims to build a chess engine that employs the Minimax algorithm with Alpha-Beta pruning and a handcrafted heuristic evaluation function. The heuristic considers factors such as material balance, piece positioning, pawn structure, and king safety. Over time, we will explore reinforcement learning techniques to refine the heuristic dynamically, allowing the engine to learn from gameplay and improve its decision-making autonomously. By integrating machine learning into a traditionally rule-based system, we aim to enhance the adaptability and performance of our chess AI.

## Motivation

The motivation behind this project stems from the intersection of classical AI methods and modern machine learning. While Minimax with Alpha-Beta pruning is well-established in game-playing AI, its performance is highly dependent on the quality of the heuristic function. Many traditional heuristics rely on human intuition and predefined rules, limiting their adaptability to novel gameplay patterns.

By incorporating reinforcement learning into the heuristic evaluation function, our project seeks to bridge the gap between classical AI and self-learning systems. The resulting chess engine will not only leverage computational efficiency but also dynamically adjust its strategy based on prior games, leading to improved decision-making over time. This hybrid approach aligns with broader AI research goals, where machine learning enhances existing deterministic strategies to yield more intelligent and adaptive agents.

## Experiment Design

Our experimental setup will involve several phases of implementation and evaluation. The project will progress through the following key stages:

1. **Baseline Implementation**: Develop a basic Minimax chess engine with Alpha-Beta pruning. Implement a handcrafted heuristic function that evaluates board states based on:
   - Material balance (assigning standard values to pieces: pawn = 1, knight = 3, bishop = 3, rook = 5, queen = 9)
   - Piece positioning (checking if pieces are in danger, or vice versa)
   - Space control (favoring central control and mobility)
   - Pawn structure (encouraging connected pawns and penalizing isolated ones)

2. **Performance Evaluation**: Measure the engine's performance against existing chess AI engines and human players at different skill levels. Performance metrics include:
   - Win rate against benchmark engines
   - Average number of moves per game
   - Evaluation function consistency across different board states

3. **Reinforcement Learning Integration**: Train a reinforcement learning model to optimize the heuristic function dynamically. The model will be trained through self-play, adjusting heuristic weights based on the outcomes of games.
   - Define a reward function that assigns positive values for advantageous moves and penalizes poor decisions.
   - Use policy gradient methods or deep Q-learning to refine heuristic parameters.

4. **Comparative Analysis**: Compare the performance of the manually crafted heuristic against the learned heuristic. Key evaluation criteria include:
   - Improved decision-making efficiency
   - Enhanced strategic adaptability
   - Reduction in evaluation function errors

By iterating through these stages, we aim to create a chess engine that not only excels in tactical play but also evolves its strategic approach over time, blending traditional AI techniques with modern reinforcement learning methodologies.
