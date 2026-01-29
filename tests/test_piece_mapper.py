if __name__ != "__main__":
    raise RuntimeError("Test file imported at runtime!")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.piece_mapper import map_pieces_to_squares

detections = [
    {"label": "white_pawn", "bbox": (50, 650, 100, 730)},     # a2
    {"label": "black_rook", "bbox": (700, 0, 790, 90)},      # h8
    {"label": "black_knight", "bbox": (420, 350, 480, 420)}  # e4
]

result = map_pieces_to_squares(detections)
print(result)
