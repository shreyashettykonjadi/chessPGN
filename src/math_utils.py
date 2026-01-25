from typing import List


def clamp(x: float, min_val: float, max_val: float) -> float:
    """Clamp x s.t. min_val <= x <= max_val"""
    return max(min_val, min(x, max_val))


def zeros(rows: int, columns: int) -> List[List[float]]:
    """Create a 2D list of zeros with given dimensions."""
    return [[0.0 for _ in range(columns)] for _ in range(rows)]
