import random
import chess
import numpy as np

class ChessBot:
    def __init__(self, model, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon  # Facteur d'exploration

    def make_move(self, board):
        # Exploration (jouer un coup aléatoire avec probabilité epsilon)
        if random.random() < self.epsilon:
            legal_moves = list(board.legal_moves)
            return random.choice(legal_moves)

        # Exploitation (utiliser le modèle pour choisir le meilleur coup)
        board_vector = board_to_vector(board)
        board_vector = np.array(board_vector)  # Convertir en numpy array
        move_scores = self.model.predict(board_vector.reshape(1, -1))
        legal_moves = list(board.legal_moves)
        best_move = max(legal_moves, key=lambda move: move_scores[0][move.to_square])
        return best_move

def board_to_vector(board):
    # Convertit l'échiquier en un vecteur de 64 cases
    piece_map = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6, 'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6}
    vector = [0] * 64
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            vector[square] = piece_map[piece.symbol()]
    return vector
