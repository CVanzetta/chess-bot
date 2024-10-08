import random
import chess.engine
import numpy as np
from chess_bot_model import create_model
from chess_bot import ChessBot, board_to_vector, move_to_index
import concurrent.futures
import tensorflow as tf
from tensorflow import keras
import signal
import sys
import psutil
from collections import deque

# Définir le répertoire des logs pour TensorBoard
log_dir = "logs/fit/"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
file_writer = tf.summary.create_file_writer(log_dir)

# Charger Stockfish
engine_path = "C:/Users/C_VANZETTA/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe"  # Ajustez le chemin si nécessaire
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

# Régler le niveau de Stockfish (de 1 à 20)
engine.configure({"Skill Level": 5, "Threads": 4, "Hash": 512})  # Optimisations

# Variable pour suivre l'état d'exécution
shutdown_flag = False

# Gérer l'interruption clavier pour arrêter proprement
def signal_handler(sig, frame):
    global shutdown_flag
    shutdown_flag = True
    print("\nInterruption détectée. Fermeture en cours...")
    engine.quit()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def get_piece_value(piece):
    """Retourne la valeur d'une pièce capturée."""
    piece_values = {
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,  # Blancs
        'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0  # Noirs
    }
    return piece_values.get(piece.symbol(), 0)

def log_metrics(episode, reward, moves_count, win_rate, file_writer):
    """Enregistrer des métriques dans TensorBoard"""
    with file_writer.as_default():
        tf.summary.scalar('reward', reward, step=episode)
        tf.summary.scalar('moves_count', moves_count, step=episode)
        tf.summary.scalar('win_rate', win_rate, step=episode)

def play_game(bot1, engine):
    """Jouer une partie entre Bot 1 et Stockfish"""
    board = chess.Board()
    states = []      # États de l'échiquier
    moves_taken = [] # Coups joués par le bot
    moves_count = 0
    total_reward = 0
    current_player = 'bot'  # Le bot commence la partie

    while not board.is_game_over():
        # Stocker l'état actuel de l'échiquier avant le coup
        state_vector = board_to_vector(board)
        states.append(state_vector)

        if current_player == 'bot':
            # Le bot joue
            move = bot1.make_move(board)
            moves_taken.append(move)  # Enregistrer le coup joué
            # Vérifier si une pièce adverse est présente sur la case cible avant de pousser le coup
            captured_piece = board.piece_at(move.to_square)
            board.push(move)  # Applique le coup sur l'échiquier

            if captured_piece is not None:
                capture_reward = get_piece_value(captured_piece)
                total_reward += capture_reward  # Le bot gagne des points
            current_player = 'stockfish'
        else:
            # Stockfish joue
            result = engine.play(board, chess.engine.Limit(time=0.5))  # Temps de réflexion réduit pour plus de parties
            move = result.move
            captured_piece = board.piece_at(move.to_square)
            board.push(move)
            if captured_piece is not None:
                capture_reward = get_piece_value(captured_piece)
                total_reward -= capture_reward  # Le bot perd des points
            current_player = 'bot'

        moves_count += 1

    # Déterminer la récompense en fonction du résultat
    result = board.result()  # "1-0", "0-1" ou "1/2-1/2"
    if result == "1-0":  # Bot 1 gagne
        total_reward += 10
    elif result == "0-1":  # Bot 1 perd
        total_reward -= 10

    return states, moves_taken, total_reward, moves_count

def simulate_game(episode):
    """Fonction exécutée en parallèle pour simuler une partie entre Bot 1 et Stockfish"""
    model1 = create_model()  # Crée le modèle localement dans chaque processus
    bot1 = ChessBot(model1, epsilon=0.1)

    # Jouer une partie contre Stockfish
    states, moves_taken, reward, moves_count = play_game(bot1, engine)
    return states, moves_taken, reward, moves_count

if __name__ == '__main__':
    try:
        # Créer un modèle pour Bot 1
        model1 = create_model()

        victories = 0
        total_games = 0
        win_rate_100 = 0  # Pour stocker le taux de victoire après chaque 100 parties
        replay_buffer = deque(maxlen=10000)  # Taille maximale du buffer de relecture

        # Utilisation de `concurrent.futures` pour exécuter plusieurs parties en parallèle
        while not shutdown_flag:
            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage < 50:
                max_workers = 8
            elif cpu_usage > 75:
                max_workers = 4
            else:
                max_workers = 6

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(simulate_game, episode) for episode in range(20)]  # Simuler 20 parties

                for future in concurrent.futures.as_completed(futures):
                    if shutdown_flag:
                        break

                    try:
                        states, moves_taken, reward, moves_count = future.result()
                    except Exception as e:
                        print(f"Erreur dans une partie : {e}")
                        continue  # Passer à la partie suivante si une erreur se produit

                    # Calculer le taux de victoire pour Bot 1
                    if reward > 0:
                        victories += 1
                    total_games += 1
                    win_rate = victories / total_games

                    # Suivi dans la console
                    print(f"Partie {total_games} terminée : reward = {reward}, moves = {moves_count}, win_rate = {win_rate:.2f}")

                    # Enregistrer les métriques dans TensorBoard
                    if total_games % 10 == 0:
                        log_metrics(total_games, reward, moves_count, win_rate, file_writer)

                    # Ajouter à la mémoire de relecture
                    for state, move in zip(states, moves_taken):
                        replay_buffer.append((state, move, reward))

                    # Entraîner Bot 1 toutes les 10 parties
                    if total_games % 10 == 0 and len(replay_buffer) >= 10:
                        X_train, y_train = [], []
                        mini_batch = random.sample(replay_buffer, 10)

                        N = 64 * 64 * 6  # Nombre total de coups possibles

                        for state, move, reward in mini_batch:
                            X_train.append(state)
                            target = np.zeros(N)
                            index = move_to_index(move)
                            target[index] = reward
                            y_train.append(target)

                        X_train = np.array(X_train)
                        y_train = np.array(y_train)
                        model1.fit(X_train, y_train, epochs=1, verbose=0, callbacks=[tensorboard_callback])

                    # Calculer et afficher le win rate toutes les 100 parties
                    if total_games % 100 == 0:
                        win_rate_100 = victories / total_games
                        print(f"Taux de victoire après {total_games} parties : {win_rate_100:.2f}")

                        # Enregistrer le win rate des 100 dernières parties dans TensorBoard
                        log_metrics(total_games, reward, moves_count, win_rate_100, file_writer)

        # Message de fin lorsque toutes les parties sont terminées
        print(f"Fin de l'entraînement : taux de victoire final = {win_rate:.2f} après {total_games} parties.")

        # Sauvegarder le modèle après l'entraînement
        model1.save('chess_bot_model.keras')

    finally:
        # Fermer le moteur Stockfish même en cas d'exception ou d'interruption
        engine.quit()
        print("Moteur Stockfish fermé.")