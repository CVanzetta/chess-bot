import chess
import random

class ChessBot:
    def __init__(self):
        pass

    def make_move(self, board):
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves)

if __name__ == "__main__":
    board = chess.Board()
    bot = ChessBot()
    move = bot.make_move(board)
    board.push(move)
    print(board)
