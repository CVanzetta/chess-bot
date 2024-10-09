import random
import chess
import numpy as np
from tensorflow.keras import layers

class ChessBot:
    def __init__(self, model, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon

    def make_move(self, board):
        if np.random.rand() < self.epsilon:
            # Choose a random move
            legal_moves = list(board.legal_moves)
            return np.random.choice(legal_moves)
        else:
            # Choose the best move predicted by the model
            state_vector = board_to_vector(board)
            predictions = self.model.predict(state_vector[np.newaxis])[0]
            best_move = self._vector_to_move(board, predictions)
            return best_move

    def _vector_to_move(self, board, predictions):
        # Convert the model's predictions into a valid move
        legal_moves = list(board.legal_moves)
        move_values = [(move, predictions[move_to_index(move)]) for move in legal_moves]
        move_values.sort(key=lambda x: x[1], reverse=True)
        return move_values[0][0]

def board_to_vector(board):
    # Convert the board state to a vector usable by the model
    vector = np.zeros(64)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            vector[square] = piece_to_value(piece)
    return vector

def piece_to_value(piece):
    # Assign a numeric value to each piece
    piece_values = {
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 10,
        'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': -10
    }
    return piece_values.get(piece.symbol(), 0)

def move_to_index(move):
    # Convert a move into an index for the output vector
    from_square = move.from_square
    to_square = move.to_square
    return from_square * 64 + to_square