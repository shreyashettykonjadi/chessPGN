from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import Counter

# Piece limits per side (max allowed)
PIECE_LIMITS = {
    'king': 1,
    'queen': 9,   # 1 original + 8 promoted pawns (theoretical max)
    'rook': 10,   # 2 original + 8 promoted
    'bishop': 10,
    'knight': 10,
    'pawn': 8,
}

# Strict limits for validation (practical game limits)
STRICT_LIMITS = {
    'king': (1, 1),      # exactly 1
    'queen': (0, 9),
    'rook': (0, 2),      # at most 2 (ignoring promotions for stability)
    'bishop': (0, 2),
    'knight': (0, 2),
    'pawn': (0, 8),
}


@dataclass
class Detection:
    """A single piece detection with confidence."""
    square: str
    piece: str  # e.g., 'white_pawn', 'black_king'
    confidence: float = 1.0


@dataclass
class BoardState:
    """Validated chess board state."""
    squares: Dict[str, str] = field(default_factory=dict)  # square -> piece
    is_valid: bool = False
    
    def to_fen_board(self) -> str:
        """Convert board state to FEN board section (first part of FEN)."""
        fen_rows = []
        for rank in range(8, 0, -1):
            row = ""
            empty_count = 0
            for file in "abcdefgh":
                square = f"{file}{rank}"
                piece = self.squares.get(square)
                if piece:
                    if empty_count > 0:
                        row += str(empty_count)
                        empty_count = 0
                    row += self._piece_to_fen_char(piece)
                else:
                    empty_count += 1
            if empty_count > 0:
                row += str(empty_count)
            fen_rows.append(row)
        return "/".join(fen_rows)
    
    def _piece_to_fen_char(self, piece: str) -> str:
        """Convert piece name to FEN character."""
        piece_map = {
            'king': 'k', 'queen': 'q', 'rook': 'r',
            'bishop': 'b', 'knight': 'n', 'pawn': 'p'
        }
        parts = piece.split('_')
        if len(parts) != 2:
            return '?'
        color, piece_type = parts[0], parts[1]
        char = piece_map.get(piece_type, '?')
        return char.upper() if color == 'white' else char


class BoardValidator:
    """Validates chess board states for legality."""
    
    def __init__(self, strict_limits: bool = True):
        if strict_limits:
            self.limits = STRICT_LIMITS
        else:
            # Convert PIECE_LIMITS (max-only ints) into (min, max) pairs with min=0
            self.limits = {k: (0, v) for k, v in PIECE_LIMITS.items()}
    
    def validate(self, squares: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Validate board state.
        Returns (is_valid, list_of_errors).
        """
        errors = []
        
        # Count pieces by color and type
        white_pieces: Counter = Counter()
        black_pieces: Counter = Counter()
        
        for square, piece in squares.items():
            parts = piece.split('_')
            if len(parts) != 2:
                errors.append(f"Invalid piece format: {piece}")
                continue
            color, piece_type = parts[0], parts[1]
            if color == 'white':
                white_pieces[piece_type] += 1
            elif color == 'black':
                black_pieces[piece_type] += 1
            else:
                errors.append(f"Invalid color: {color}")
        
        # Check king constraints (must have exactly 1 each)
        if white_pieces['king'] != 1:
            errors.append(f"White must have exactly 1 king, found {white_pieces['king']}")
        if black_pieces['king'] != 1:
            errors.append(f"Black must have exactly 1 king, found {black_pieces['king']}")
        
        # Check other piece limits
        for piece_type, (min_count, max_count) in self.limits.items():
            if piece_type == 'king':
                continue  # Already checked
            if white_pieces[piece_type] > max_count:
                errors.append(f"White has too many {piece_type}s: {white_pieces[piece_type]} > {max_count}")
            if black_pieces[piece_type] > max_count:
                errors.append(f"Black has too many {piece_type}s: {black_pieces[piece_type]} > {max_count}")
        
        return len(errors) == 0, errors
    
    def resolve_conflicts(self, detections: List[Detection]) -> Dict[str, str]:
        """
        Resolve multiple detections per square by keeping highest confidence.
        Returns clean square -> piece mapping.
        """
        # Group by square, keep highest confidence
        best_per_square: Dict[str, Detection] = {}
        
        for det in detections:
            if det.square not in best_per_square:
                best_per_square[det.square] = det
            elif det.confidence > best_per_square[det.square].confidence:
                best_per_square[det.square] = det
        
        return {sq: det.piece for sq, det in best_per_square.items()}
