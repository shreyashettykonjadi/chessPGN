from src.move_inference import infer_moves_from_fens

fens = [
    "8/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "8/8/8/8/8/4P3/8/4K3 b - - 0 1",
]

moves = infer_moves_from_fens(fens, debug=True)

print("Moves:", [m.uci() for m in moves])
