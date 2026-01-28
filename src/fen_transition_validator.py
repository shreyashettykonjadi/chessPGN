"""
FEN Transition Validator - Phase 5.1b

Validates that FEN transitions are chess-legal before adding to timeline.
Rejects noisy/impossible transitions while allowing valid move sequences.
"""

from typing import Optional, Tuple, List, Dict, Any
import chess


def get_board_fen(full_fen: str) -> str:
    """Extract only the board portion from a full FEN string."""
    return full_fen.split()[0]


def count_square_differences(fen1: str, fen2: str) -> int:
    """
    Count how many squares differ between two board positions.
    
    Normal move: 2 squares change (from and to)
    Castling: 4 squares change (king and rook, from and to)
    En passant: 3 squares change (pawn from, pawn to, captured pawn)
    """
    try:
        board1 = chess.Board(fen1)
        board2 = chess.Board(fen2)
    except ValueError:
        return 999  # Invalid FEN, reject
    
    diff_count = 0
    for square in chess.SQUARES:
        piece1 = board1.piece_at(square)
        piece2 = board2.piece_at(square)
        if piece1 != piece2:
            diff_count += 1
    
    return diff_count


def validate_kings(fen: str) -> Tuple[bool, str]:
    """
    Ensure exactly one white king and one black king.
    Returns (is_valid, error_message).
    """
    try:
        board = chess.Board(fen)
    except ValueError as e:
        return False, f"Invalid FEN: {e}"
    
    white_kings = len(board.pieces(chess.KING, chess.WHITE))
    black_kings = len(board.pieces(chess.KING, chess.BLACK))
    
    if white_kings != 1:
        return False, f"Expected 1 white king, found {white_kings}"
    if black_kings != 1:
        return False, f"Expected 1 black king, found {black_kings}"
    
    return True, ""


class FENTransitionValidator:
    """
    Validate FEN transitions by requiring exactly one legal move explains the change.
    """
    
    def __init__(self, max_square_diff: int = 8, debug: bool = False):
        self.max_square_diff = max_square_diff
        self.debug = debug
        self._last_valid_fen: Optional[str] = None
    
    def reset(self):
        """Reset validator state."""
        self._last_valid_fen = None
    
    @property
    def last_valid_fen(self) -> Optional[str]:
        return self._last_valid_fen

    def validate(self, fen: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Validate current FEN against last valid FEN.

        Returns:
            (is_valid, reason, move_meta)
        move_meta is a dict with keys: 'uci', 'san', 'ply_index' when accepted (None for first entry).
        """
        # Basic king validation on current fen
        ok, err = validate_kings(fen)
        if not ok:
            if self.debug:
                print(f"[validator] Rejected current FEN: {err}")
            return False, err, None

        # If no previous valid FEN, accept as initial position (no move_meta)
        if self._last_valid_fen is None:
            self._last_valid_fen = fen
            if self.debug:
                print("[validator] Accepted initial FEN")
            return True, "", None

        prev_fen = self._last_valid_fen

        # Quick square-difference heuristic
        diff = count_square_differences(prev_fen, fen)
        if self.debug:
            print(f"[validator] Square differences: {diff}")

        # If too many differences and no unique legal move explains, reject
        try:
            prev_board = chess.Board(prev_fen)
            curr_board = chess.Board(fen)
        except Exception as e:
            if self.debug:
                print(f"[validator] Invalid FEN(s): {e}")
            return False, f"Invalid FEN(s): {e}", None

        legal_moves = list(prev_board.legal_moves)
        if self.debug:
            print(f"[validator] Legal moves to try: {len(legal_moves)}")

        matching: List[Tuple[chess.Move, str, int]] = []
        ply_index = len(prev_board.move_stack)

        # Test each legal move - compute SAN before push
        for mv in legal_moves:
            try:
                san = prev_board.san(mv)
            except Exception:
                san = ""
            prev_board.push(mv)
            if prev_board.board_fen() == curr_board.board_fen():
                matching.append((mv, san, ply_index))
            prev_board.pop()

        if self.debug:
            print(f"[validator] Matching moves found: {len(matching)}")

        if len(matching) == 1:
            mv, san, ply = matching[0]
            meta = {"uci": mv.uci(), "san": san, "move_index": ply}
            # Accept transition and update last valid fen
            self._last_valid_fen = fen
            if self.debug:
                print(f"[validator] Accepted move {san} ({mv.uci()}) at ply {ply}")
            return True, "", meta

        # If no unique match, consider heuristic: if diff <= max_square_diff and no match -> reject
        if len(matching) == 0:
            reason = f"No legal move matched ({diff} squares changed)"
            if self.debug:
                print(f"[validator] Rejected: {reason}")
            return False, reason, None

        # Ambiguous (>1 matches)
        reason = f"Ambiguous transition: {len(matching)} matching moves"
        if self.debug:
            print(f"[validator] Rejected: {reason}")
        return False, reason, None
