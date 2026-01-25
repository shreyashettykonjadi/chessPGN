from typing import Dict

# Mapping from internal labels to FEN characters
PIECE_TO_FEN = {
    "white_pawn": "P",
    "white_rook": "R",
    "white_knight": "N",
    "white_bishop": "B",
    "white_queen": "Q",
    "white_king": "K",
    "black_pawn": "p",
    "black_rook": "r",
    "black_knight": "n",
    "black_bishop": "b",
    "black_queen": "q",
    "black_king": "k",
}

FILES = ["a", "b", "c", "d", "e", "f", "g", "h"]


def build_fen_board(piece_map: Dict[str, str]) -> str:
    """
    Build the board portion of FEN from a mapping:
    { "a1": "white_rook", "e4": "black_knight", ... }
    """
    rows = []

    # FEN ranks go from 8 -> 1
    for rank in range(8, 0, -1):
        empty = 0
        row = ""

        for file in FILES:
            square = f"{file}{rank}"
            if square in piece_map:
                if empty > 0:
                    row += str(empty)
                    empty = 0
                label = piece_map[square]
                row += PIECE_TO_FEN[label]
            else:
                empty += 1

        if empty > 0:
            row += str(empty)

        rows.append(row)

    return "/".join(rows)


def build_full_fen(piece_map: Dict[str, str]) -> str:
    """
    Build a FULL FEN string (Phase 3.5).
    Side to move, castling, en-passant, clocks are defaulted.
    """
    board_fen = build_fen_board(piece_map)

    side_to_move = "w"
    castling = "KQkq"
    en_passant = "-"
    halfmove = "0"
    fullmove = "1"

    return f"{board_fen} {side_to_move} {castling} {en_passant} {halfmove} {fullmove}"
