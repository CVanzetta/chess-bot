import random
import chess
import numpy as np

def board_to_vector(board):
    """Convertit l'échiquier en un vecteur de 64 cases."""
    piece_map = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                 'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6}
    vector = [0] * 64
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            vector[square] = piece_map[piece.symbol()]
    return np.array(vector)

def move_to_index(move):
    """
    Mappe un coup à un indice unique entre 0 et N-1.
    """
    from_square = move.from_square  # Valeur entre 0 et 63
    to_square = move.to_square      # Valeur entre 0 et 63
    promotion = move.promotion if move.promotion else 0  # Valeur entre 0 et 5
    # Calculer un indice unique
    index = from_square * 64 + to_square
    if promotion:
        index += promotion * 4096  # 64*64 = 4096
    return index

class ChessBot:
    def __init__(self, model, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon  # Taux d'exploration

    def make_move(self, board):
        import tensorflow as tf  # Assurez-vous d'importer tensorflow ici si nécessaire

        if random.random() < self.epsilon:
            legal_moves = list(board.legal_moves)
            return random.choice(legal_moves)

        # Exploitation : Utiliser le modèle pour choisir le meilleur coup
        board_vector = board_to_vector(board)
        move_scores = self.model.predict(board_vector.reshape(1, -1))[0]

        legal_moves = list(board.legal_moves)
        best_move = None
        best_score = -np.inf
        for move in legal_moves:
            index = move_to_index(move)
            score = move_scores[index]
            if score > best_score:
                best_score = score
                best_move = move
        return best_move
