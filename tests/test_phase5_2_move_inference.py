import chess

def infer_moves_from_fens(fen_history):
    moves = []
    for i in range(len(fen_history) - 1):
        board = chess.Board(fen_history[i])
        next_fen = fen_history[i + 1].split(" ")[0]

        found_move = None
        for move in board.legal_moves:
            board.push(move)
            if board.board_fen() == next_fen:
                found_move = move.uci()
                board.pop()
                break
            board.pop()

        if found_move:
            moves.append(found_move)

    return moves


def test_simple_opening_move():
    fen_history = [
        chess.STARTING_FEN,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    ]

    moves = infer_moves_from_fens(fen_history)

    print("Moves inferred:", moves)
    assert moves == ["e2e4"], "Failed to infer e2e4"


if __name__ == "__main__":
    test_simple_opening_move()
