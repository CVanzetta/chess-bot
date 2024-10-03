import chess

def play_chess():
    board = chess.Board()
    print(board)

    # Exemple de mouvements basiques
    board.push_san("e4")
    board.push_san("e5")
    print(board)

if __name__ == "__main__":
    play_chess()
