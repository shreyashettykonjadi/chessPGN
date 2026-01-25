import cv2  # type: ignore
# pyright: reportAttributeAccessIssue=false

from src.grid import build_board_grid, extract_square

# Load warped board
board = cv2.imread("warped_board.png")
if board is None:
    raise RuntimeError("warped_board.png not found")

# Build grid
grid = build_board_grid(800)

print("Total squares:", len(grid))
print("a1:", grid["a1"])
print("e4:", grid["e4"])
print("h8:", grid["h8"])

# Use at least 10 algebraic squares to match grid's string keys.
test_squares = [
    "a8",
    "h8",
    "a1",
    "h1",
    "e5",
    "e4",
    "a5",
    "e8",
    "h4",
    "e1",
]

# If a previous test_squares list exists, replace it with the above.
# Ensure any iteration over test_squares or grid sampling uses this list.
# Example usage:
# for sq in test_squares:
#     # ...existing code...
#     # compute square bounds from (file_idx, rank_idx)
#     # ...existing code...

# Visual verification
for sq in test_squares:
    img = extract_square(board, grid[sq])
    cv2.imshow(sq, img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
