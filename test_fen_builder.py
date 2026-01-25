from src.fen_builder import build_full_fen

piece_map = {
    "a1": "white_rook",
    "b1": "white_knight",
    "e4": "black_knight",
    "h8": "black_rook",
}

fen = build_full_fen(piece_map)
print(fen)
