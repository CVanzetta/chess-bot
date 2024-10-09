import random
import chess.engine
import numpy as np
from chess_bot_model import create_model
from chess_bot_module import ChessBot, board_to_vector, move_to_index
import concurrent.futures
import tensorflow as tf
from tensorflow import keras
import signal
import sys
import psutil
from collections import deque
import os

# Disable GPU usage if needed
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Define the log directory for TensorBoard
log_dir = "logs/fit/"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
file_writer = tf.summary.create_file_writer(log_dir)

# Load Stockfish
engine_path = "D:/Code/stockfish/stockfish-windows-x86-64-avx2"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

# Configure Stockfish level (from 1 to 20)
engine.configure({"Skill Level": 1, "Threads": 4, "Hash": 512})  # Optimizations

# Variable to track execution state
shutdown_flag = False

# Handle keyboard interrupt for graceful shutdown
def signal_handler(sig, frame):
    global shutdown_flag
    shutdown_flag = True
    print("\nInterrupt detected. Shutting down...")

signal.signal(signal.SIGINT, signal_handler)

# Decreasing epsilon management
def get_epsilon(total_games, initial_epsilon=0.1, min_epsilon=0.01, decay_rate=0.0005):
    return max(min_epsilon, initial_epsilon * np.exp(-decay_rate * total_games))

# Advanced reward function
def advanced_reward_function(board, move, captured_piece, game_result):
    reward = 0
    if captured_piece is not None:
        reward += get_piece_value(captured_piece)
    # Bonus for piece development and center control
    if move is not None and move.to_square in [chess.E4, chess.D4, chess.E5, chess.D5]:
        reward += 0.5
    # Reward for checkmate
    if board.is_checkmate():
        reward += 100 if game_result == "1-0" else -100
    # Reward for check
    if board.is_check():
        reward += 5
    return reward

# Function to get the value of a piece
def get_piece_value(piece):
    """Returns the value of a captured piece."""
    piece_values = {
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,  # White
        'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0  # Black
    }
    return piece_values.get(piece.symbol(), 0)

# Log metrics to TensorBoard
def log_metrics(episode, reward, moves_count, win_rate, file_writer):
    """Log metrics to TensorBoard"""
    with file_writer.as_default():
        tf.summary.scalar('reward', reward, step=episode)
        tf.summary.scalar('moves_count', moves_count, step=episode)
        tf.summary.scalar('win_rate', win_rate, step=episode)

# Function to play a game
def play_game(bot1, engine):
    """Play a game between Bot 1 and Stockfish"""
    board = chess.Board()
    states = []      # Board states
    moves_taken = [] # Moves taken by the bot
    moves_count = 0
    total_reward = 0
    current_player = 'bot'  # The bot starts the game

    while not board.is_game_over() and not shutdown_flag:
        print(f"Turn {moves_count + 1}, player: {current_player}")
        # Store the current board state before the move
        state_vector = board_to_vector(board)
        states.append(state_vector)

        if current_player == 'bot':
            # Bot's turn
            legal_moves = list(board.legal_moves)
            capture_moves = [move for move in legal_moves if board.piece_at(move.to_square) is not None]

            if capture_moves:
                move = random.choice(capture_moves)
                print(f"Capture possible, bot plays: {move}")
            else:
                move = bot1.make_move(board)

            if move is None:
                print("No possible move for the bot, game over.")
                break

            moves_taken.append(move)  # Record the move
            # Check if an opponent piece is present on the target square before pushing the move
            captured_piece = board.piece_at(move.to_square)
            board.push(move)  # Apply the move on the board

            print(f"Bot plays: {move}, captured piece: {captured_piece}")
            total_reward += advanced_reward_function(board, move, captured_piece, None)
            current_player = 'stockfish'
        else:
            # Stockfish's turn
            result = engine.play(board, chess.engine.Limit(time=0.5))  # Reduced thinking time for more games
            move = result.move
            if move is None:
                print("No possible move for Stockfish, game over.")
                break
            captured_piece = board.piece_at(move.to_square)
            board.push(move)

            print(f"Stockfish plays: {move}, captured piece: {captured_piece}")
            total_reward -= advanced_reward_function(board, move, captured_piece, None)
            current_player = 'bot'

        moves_count += 1

    # Determine the reward based on the result
    if not shutdown_flag:
        result = board.result()  # "1-0", "0-1" or "1/2-1/2"
        print(f"Game result: {result}")
        total_reward += advanced_reward_function(board, None, None, result)

    return states, moves_taken, total_reward, moves_count

# Function to simulate a game
def simulate_game(episode, epsilon):
    """Function executed in parallel to simulate a game between Bot 1 and Stockfish"""
    print(f"Simulating game {episode + 1} with epsilon = {epsilon}")
    model1 = create_model()  # Create the model locally in each process
    bot1 = ChessBot(model1, epsilon=epsilon)

    # Play a game against Stockfish
    states, moves_taken, reward, moves_count = play_game(bot1, engine)
    return states, moves_taken, reward, moves_count

# Main entry point
if __name__ == '__main__':
    try:
        # Create a model for Bot 1
        model1 = create_model()

        victories = 0
        total_games = 0
        win_rate_100 = 0  # To store the win rate after every 100 games
        replay_buffer = deque(maxlen=50000)  # Maximum size of the replay buffer

        # Use `concurrent.futures` to run multiple games in parallel
        while not shutdown_flag:
            cpu_usage = psutil.cpu_percent(interval=1)
            print(f"CPU usage: {cpu_usage}%")
            if cpu_usage < 50:
                max_workers = 8
            elif cpu_usage > 75:
                max_workers = 4
            else:
                max_workers = 6

            epsilon = get_epsilon(total_games)
            print(f"Epsilon for the next games: {epsilon}")

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(simulate_game, episode, epsilon) for episode in range(200)]  # Simulate 20 games

                for future in concurrent.futures.as_completed(futures):
                    if shutdown_flag:
                        break

                    try:
                        states, moves_taken, reward, moves_count = future.result()
                    except Exception as e:
                        print(f"Error in a game: {e}")
                        continue  # Skip to the next game if an error occurs

                    # Calculate the win rate for Bot 1
                    if reward > 0:
                        victories += 1
                    total_games += 1
                    win_rate = victories / total_games

                    # Log to console
                    print(f"Game {total_games} completed: reward = {reward}, moves = {moves_count}, win_rate = {win_rate:.2f}")

                    # Log metrics to TensorBoard
                    if total_games % 10 == 0:
                        log_metrics(total_games, reward, moves_count, win_rate, file_writer)

                    # Add to the replay buffer
                    for state, move in zip(states, moves_taken):
                        replay_buffer.append((state, move, reward))

                    # Train Bot 1 every 10 games
                    if total_games % 10 == 0 and len(replay_buffer) >= 20:
                        print(f"Training the model after {total_games} games")
                        X_train, y_train = [], []
                        mini_batch = random.sample(replay_buffer, 20)

                        N = 64 * 64 * 6  # Total number of possible moves

                        for state, move, reward in mini_batch:
                            X_train.append(state)
                            target = np.zeros(N)
                            index = move_to_index(move)
                            target[index] = reward
                            y_train.append(target)

                        X_train = np.array(X_train)
                        y_train = np.array(y_train)
                        model1.fit(X_train, y_train, epochs=1, verbose=0, callbacks=[tensorboard_callback])

                    # Calculate and display the win rate every 100 games
                    if total_games % 100 == 0:
                        win_rate_100 = victories / total_games
                        print(f"Win rate after {total_games} games: {win_rate_100:.2f}")

                        # Log the win rate of the last 100 games to TensorBoard
                        log_metrics(total_games, reward, moves_count, win_rate_100, file_writer)

        # End message when all games are completed
        print(f"Training complete: final win rate = {win_rate:.2f} after {total_games} games.")

        # Save the model after training
        model1.save('chess_bot_model.keras')

    finally:
        # Close the Stockfish engine even in case of exception or interruption
        try:
            engine.quit()
        except Exception as e:
            print(f"Error closing the Stockfish engine: {e}")
        print("Stockfish engine closed.")