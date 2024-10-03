from chess_bot import ChessBot
from chess_bot_model import create_model
import chess

# Créez un nouveau modèle ou chargez un modèle entraîné
model = create_model()  # Ou charger un modèle existant avec model = tf.keras.models.load_model('chess_bot_model.h5')

# Instanciez un bot avec ce modèle
bot = ChessBot(model, epsilon=0.1)  # epsilon est le facteur d'exploration

# Créez un échiquier
board = chess.Board()

# Faites jouer quelques coups au bot
while not board.is_game_over():
    move = bot.make_move(board)  # Le bot choisit un coup
    board.push(move)  # Applique le coup sur l'échiquier
    print(board)  # Affiche l'état de l'échiquier après chaque coup

# Affiche le résultat de la partie
print("Résultat de la partie:", board.result())
