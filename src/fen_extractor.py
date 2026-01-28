from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .board_state import BoardState, BoardValidator, Detection
from .temporal_smoother import TemporalSmoother, SmoothedBoard


@dataclass
class ExtractionResult:
    """Result of FEN extraction with metadata."""
    fen: str
    is_valid: bool
    used_fallback: bool
    errors: List[str]


class FENExtractor:
    """
    FEN extractor with validation and temporal stabilization.
    
    Pipeline:
    1. Raw detections -> Conflict resolution (one piece per square)
    2. Temporal smoothing (majority vote over sliding window)
    3. Validation (chess legality constraints)
    4. FEN generation (only from valid, stabilized boards)
    """

    # Default FEN for empty/invalid boards
    EMPTY_FEN = "8/8/8/8/8/8/8/8 w - - 0 1"
    STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def __init__(self, window_size: int = 5, strict_validation: bool = True):
        self.validator = BoardValidator(strict_limits=strict_validation)
        self.smoother = TemporalSmoother(window_size=window_size)
        self.last_valid_state: Optional[BoardState] = None
    
    def extract_fen(self, board_image: Any) -> str:
        """
        Extract FEN string from a board image.

        WARNING:
        This method requires `_detect_pieces()` to be implemented.
        If you already have detections, use `process_detections()`.
        """
        if self._detect_pieces.__func__ is FENExtractor._detect_pieces:
            raise RuntimeError(
                "extract_fen(board_image) requires _detect_pieces() to be implemented. "
                "Use process_detections() when detections are already available."
            )

        result = self.extract_fen_with_metadata(board_image)
        return result.fen

    
    def extract_fen_with_metadata(self, board_image: Any) -> ExtractionResult:
        """Extract FEN with full metadata about the extraction process."""
        # Step 1: Get raw detections
        raw_detections = self._detect_pieces(board_image)
        
        # Step 2: Resolve conflicts (one piece per square)
        resolved = self.validator.resolve_conflicts(raw_detections)
        
        # Step 3: Temporal smoothing
        smoothed = self.smoother.add_frame(resolved)
        
        # Step 4: Validate
        is_valid, errors = self.validator.validate(smoothed.squares)
        
        # Step 5: Build board state and FEN
        if is_valid:
            board_state = BoardState(squares=smoothed.squares, is_valid=True)
            self.last_valid_state = board_state
            return ExtractionResult(
                fen=self._build_full_fen(board_state),
                is_valid=True,
                used_fallback=False,
                errors=[]
            )
        else:
            # Use last valid state as fallback
            if self.last_valid_state:
                return ExtractionResult(
                    fen=self._build_full_fen(self.last_valid_state),
                    is_valid=False,
                    used_fallback=True,
                    errors=errors
                )
            else:
                return ExtractionResult(
                    fen=self.EMPTY_FEN,
                    is_valid=False,
                    used_fallback=True,
                    errors=errors
                )
    
    def process_detections(self, detections: Dict[str, str], 
                           confidences: Optional[Dict[str, float]] = None) -> str:
        """
        Process raw detections dict directly (for external integration).
        
        Args:
            detections: Dict mapping square -> piece (e.g., {'a2': 'white_pawn'})
            confidences: Optional dict mapping square -> confidence score
        
        Returns:
            Validated, stabilized FEN string.
        """
        # Convert to Detection objects
        detection_list = []
        for square, piece in detections.items():
            conf = confidences.get(square, 1.0) if confidences else 1.0
            detection_list.append(Detection(square=square, piece=piece, confidence=conf))
        
        # Resolve conflicts
        resolved = self.validator.resolve_conflicts(detection_list)
        
        # Temporal smoothing
        smoothed = self.smoother.add_frame(resolved)
        
        # Validate
        is_valid, errors = self.validator.validate(smoothed.squares)
        
        # Build FEN
        if is_valid:
            board_state = BoardState(squares=smoothed.squares, is_valid=True)
            self.last_valid_state = board_state
            return self._build_full_fen(board_state)
        elif self.last_valid_state:
            return self._build_full_fen(self.last_valid_state)
        else:
            return self.EMPTY_FEN
    
    def _detect_pieces(self, board_image: Any) -> List[Detection]:
        """
        Detect pieces from board image.
        
        Override this method to integrate with actual detection models.
        Default returns empty list (placeholder).
        """
        # Placeholder - override in subclass or replace with actual detection
        return []
    
    def _build_full_fen(self, board_state: BoardState) -> str:
        """Build complete FEN string from board state."""
        board_fen = board_state.to_fen_board()
        # Default: white to move, no castling info, no en passant
        return f"{board_fen} w - - 0 1"
    
    def reset(self):
        """Reset the extractor state (clear history and last valid state)."""
        self.smoother.reset()
        self.last_valid_state = None
