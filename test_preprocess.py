import cv2
import numpy as np

from src.detectors.piece_detector import preprocess_board

# Load a warped board image you already saved
board_img = cv2.imread("warped_board.png")

if board_img is None:
    raise RuntimeError("warped_board.png not found")

tensor = preprocess_board(board_img)

print("Tensor shape:", tensor.shape)
print("Tensor dtype:", tensor.dtype)
print("Min value:", tensor.min())
print("Max value:", tensor.max())
