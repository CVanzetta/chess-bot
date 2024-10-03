import tensorflow as tf
import numpy as np
from chess_bot_model import create_model
from chess_bot import ChessBot, board_to_vector
import chess

def play_game(bot1, bot2):
    board = chess.Board()
    states = []  # On enregistre les états de l'échiquier à chaque coup
    current_bot = bot1  # Le bot 1 commence la partie

    while not board.is_game_over():
        # Stocker l'état actuel de l'échiquier avant le coup
        states.append(board_to_vector(board))

        move = current_bot.make_move(board)
        board.push(move)  # Applique le coup sur l'échiquier

        # Alterner entre bot1 et bot2
        current_bot = bot2 if current_bot == bot1 else bot1

    # Déterminer la récompense en fonction du résultat
    result = board.result()  # 1-0, 0-1 ou 1/2-1/2
    if result == "1-0":  # Bot 1 gagne
        reward = 1
    elif result == "0-1":  # Bot 1 perd
        reward = -1
    else:
        reward = 0  # Match nul

    return states, reward

def update_model(model, X_train, y_train, epochs=1, tensorboard_callback=None):
    model.fit(X_train, y_train, epochs=epochs, verbose=0, callbacks=[tensorboard_callback])

# Entraînement du modèle
model = create_model()
bot1 = ChessBot(model, epsilon=0.1)
bot2 = ChessBot(model, epsilon=0.1)

# Initialiser le callback TensorBoard
log_dir = "logs/fit/"  # Dossier où seront stockés les logs
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

for episode in range(100):  # Nombre de parties d'entraînement
    # Jouer une partie
    states, reward = play_game(bot1, bot2)

    # Convertir les états de l'échiquier en numpy array pour l'entraînement
    X_train = np.array(states)  # Les états de l'échiquier avant chaque coup
    y_train = np.array([[reward]] * len(states))  # La même récompense pour chaque état

    # Mise à jour du modèle avec TensorBoard callback
    update_model(bot1.model, X_train, y_train, tensorboard_callback=tensorboard_callback)

    # Réduction progressive de l'exploration
    bot1.epsilon *= 0.99
    bot2.epsilon *= 0.99

# Sauvegarder le modèle après l'entraînement
model.save('chess_bot_model.keras')
