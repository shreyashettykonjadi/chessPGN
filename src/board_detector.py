from typing import Any, Tuple

class BoardDetector:
    """Placeholder board detector."""

    def detect_board(self, frame: Any) -> Tuple[Any, Tuple[int, int, int, int]]:
        """
        Detect chessboard in a frame.
        Returns (board_image, (x, y, w, h)).
        Placeholder returns the input as board_image and zeros for coords.
        """
        return frame, (0, 0, 0, 0)
