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
    Temporal smoothing using sliding window with majority voting.
    Reduces flicker and noise in frame-by-frame detections.
    """
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
    
    def add_frame(self, squares: Dict[str, str]) -> SmoothedBoard:
        """
        Add a new frame and return smoothed board state.
        Uses majority voting across the sliding window.
        """
        self.history.append(squares.copy())
        return self._compute_majority_vote()
    
    def _compute_majority_vote(self) -> SmoothedBoard:
        """Compute majority vote for each square across window."""
        if not self.history:
            return SmoothedBoard(squares={}, confidence_scores={})
        
        # Collect all squares seen in window
        all_squares = set()
        for frame in self.history:
            all_squares.update(frame.keys())
        
        result_squares: Dict[str, str] = {}
        confidence_scores: Dict[str, float] = {}
        
        for square in all_squares:
            # Count occurrences of each piece on this square
            piece_counts: Counter = Counter()
            empty_count = 0
            
            for frame in self.history:
                piece = frame.get(square)
                if piece:
                    piece_counts[piece] += 1
                else:
                    empty_count += 1
            
            # Majority vote
            total_frames = len(self.history)
            
            if piece_counts:
                most_common_piece, count = piece_counts.most_common(1)[0]
                # Piece wins if it appears more than empty
                if count > empty_count:
                    result_squares[square] = most_common_piece
                    confidence_scores[square] = count / total_frames
                # Otherwise square is empty (don't add to result)
            # If no pieces detected, square remains empty
        
        return SmoothedBoard(squares=result_squares, confidence_scores=confidence_scores)
    
    def reset(self):
        """Clear the history buffer."""
        self.history.clear()
    
    @property
    def is_stable(self) -> bool:
        """Check if we have enough frames for reliable smoothing."""
        return len(self.history) >= self.window_size // 2 + 1
