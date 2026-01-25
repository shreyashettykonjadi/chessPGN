def build_board_grid(board_size: int = 800) -> dict:
    square_size = board_size // 8
    grid = {}

    for file_idx in range(8):  # a (0) to h (7)
        file_char = chr(ord('a') + file_idx)
        x1 = file_idx * square_size
        x2 = (file_idx + 1) * square_size

        for rank_idx in range(8):  # top (0) to bottom (7)
            rank = 8 - rank_idx    # chess rank
            y1 = rank_idx * square_size
            y2 = (rank_idx + 1) * square_size

            square_name = f"{file_char}{rank}"
            grid[square_name] = (x1, y1, x2, y2)

    return grid


def extract_square(board_img, square_coords):
    """
    Extract the cropped square image using array slicing.
    board_img: NumPy array of the board image.
    square_coords: tuple (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = square_coords
    return board_img[y1:y2, x1:x2]
