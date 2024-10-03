import chess.engine
import numpy as np
from chess_bot_model import create_model
from chess_bot import ChessBot, board_to_vector
import concurrent.futures
import tensorflow as tf
import signal
import sys

# Définir le répertoire des logs pour TensorBoard
log_dir = "logs/fit/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
file_writer = tf.summary.create_file_writer(log_dir)

# Charger Stockfish avec subprocess
engine_path = "C:/Users/C_VANZETTA/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

# Régler le niveau de Stockfish (de 1 à 20)
engine.configure({"Skill Level": 20, "Threads": 2, "Hash": 256})  # Optimisations

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
    states = []  # On enregistre les états de l'échiquier à chaque coup
    current_bot = bot1  # Bot 1 commence la partie
    moves_count = 0
    total_reward = 0

    while not board.is_game_over():
        # Stocker l'état actuel de l'échiquier avant le coup
        states.append(board_to_vector(board))

        if current_bot == bot1:
            # Bot 1 fait un coup
            move = current_bot.make_move(board)
        else:
            # Stockfish fait un coup avec un temps de réflexion plus long (1 seconde)
            result = engine.play(board, chess.engine.Limit(time=5.0))
            move = result.move

        board.push(move)  # Applique le coup sur l'échiquier

        # Vérifier si une pièce a été capturée
        captured_piece = board.piece_at(move.to_square)
        if captured_piece is not None:
            capture_reward = get_piece_value(captured_piece)
            total_reward += capture_reward  # Ajouter les points pour la pièce capturée

        # Alterner entre Bot 1 et Stockfish
        current_bot = bot1 if current_bot != bot1 else None
        moves_count += 1

    # Déterminer la récompense en fonction du résultat
    result = board.result()  # 1-0, 0-1 ou 1/2-1/2
    if result == "1-0":  # Bot 1 gagne
        total_reward += 1
    elif result == "0-1":  # Bot 1 perd
        total_reward -= 1

    return states, total_reward, moves_count

def simulate_game(episode, model1):
    """Fonction exécutée en parallèle pour simuler une partie entre Bot 1 et Stockfish"""
    bot1 = ChessBot(model1, epsilon=0.3)

    # Jouer une partie contre Stockfish
    states, reward, moves_count = play_game(bot1, engine)
    return states, reward, moves_count

if __name__ == '__main__':
    try:
        # Créer un modèle pour Bot 1
        model1 = create_model()

        victories = 0
        total_games = 0
        win_rate_100 = 0  # Pour stocker le taux de victoire après chaque 100 parties

        # Utilisation de `concurrent.futures` pour exécuter plusieurs parties en parallèle
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(simulate_game, episode, model1) for episode in range(1000)]  # Simuler 1000 parties

            for future in concurrent.futures.as_completed(futures):
                if shutdown_flag:
                    break

                try:
                    states, reward, moves_count = future.result()
                except Exception as e:
                    print(f"Erreur dans une partie : {e}")
                    continue  # Passer à la partie suivante si une erreur se produit

                # Calculer le taux de victoire pour Bot 1
                if reward > 0:
                    victories += 1
                total_games += 1
                win_rate = victories / total_games

                # Suivi dans la console
                print(f"Partie {total_games}/1000 terminée : reward = {reward}, moves = {moves_count}, win_rate = {win_rate:.2f}")

                # Enregistrer les métriques dans TensorBoard
                log_metrics(total_games, reward, moves_count, win_rate, file_writer)

                # Entraîner Bot 1 après chaque partie
                X_train = np.array(states)
                y_train = np.array([[reward]] * len(states))
                model1.fit(X_train, y_train, epochs=1, verbose=0, callbacks=[tensorboard_callback])

                # Calculer et afficher le win rate toutes les 100 parties
                if total_games % 100 == 0:
                    win_rate_100 = victories / total_games
                    print(f"Win rate après {total_games} parties : {win_rate_100:.2f}")

                    # Enregistrer le win rate des 100 dernières parties dans TensorBoard
                    log_metrics(total_games, reward, moves_count, win_rate_100, file_writer)

        # Message de fin lorsque toutes les parties sont terminées
        print(f"Fin de l'entraînement : win rate final = {win_rate:.2f} après {total_games} parties.")

        # Sauvegarder le modèle après l'entraînement
        model1.save('chess_bot_model.keras')

    finally:
        # Fermer le moteur Stockfish même en cas d'exception ou d'interruption
        engine.quit()
        print("Moteur Stockfish fermé.")
