from typing import Dict, Optional, List
from collections import deque, Counter
from dataclasses import dataclass


@dataclass
class SmoothedBoard:
    """Result of temporal smoothing."""
    squares: Dict[str, str]
    confidence_scores: Dict[str, float]  # Per-square confidence (vote ratio)


class TemporalSmoother:
    """
    Temporal smoothing using sliding window with weak stability:
    Accept a piece on a square if it appears at least `min_hits` times
    within the window (not strict majority).
    """
    
    def __init__(self, window_size: int = 5, min_hits: int = 2):
        self.window_size = window_size
        self.min_hits = max(2, int(min_hits))  # ensure not single-frame
        self.history: deque = deque(maxlen=window_size)
    
    def add_frame(self, squares: Dict[str, str]) -> SmoothedBoard:
        """
        Add a new frame and return smoothed board state.
        Uses weak stability (min_hits) across the sliding window.
        """
        self.history.append(squares.copy())
        return self._compute_majority_vote()
    
    def _compute_majority_vote(self) -> SmoothedBoard:
        """Compute weakly-stable selection for each square across window."""
        if not self.history:
            return SmoothedBoard(squares={}, confidence_scores={})
        
        # Collect all squares seen in window
        all_squares = set()
        for frame in self.history:
            all_squares.update(frame.keys())
        
        result_squares: Dict[str, str] = {}
        confidence_scores: Dict[str, float] = {}
        total_frames = len(self.history)
        
        for square in all_squares:
            # Count occurrences of each piece on this square
            piece_counts: Counter = Counter()
            for frame in self.history:
                piece = frame.get(square)
                if piece:
                    piece_counts[piece] += 1
            
            if piece_counts:
                most_common_piece, count = piece_counts.most_common(1)[0]
                # Weak stability: accept if seen at least min_hits times
                if count >= self.min_hits:
                    result_squares[square] = most_common_piece
                    confidence_scores[square] = count / total_frames
                # Otherwise square is considered empty (not added)
            # If no pieces detected, square remains empty
        
        return SmoothedBoard(squares=result_squares, confidence_scores=confidence_scores)
    
    def reset(self):
        """Clear the history buffer."""
        self.history.clear()
    
    @property
    def is_stable(self) -> bool:
        """Check if we have enough frames to avoid single-frame outputs."""
        return len(self.history) >= self.min_hits
