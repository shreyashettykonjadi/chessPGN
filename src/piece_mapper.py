from typing import List, Dict, Tuple, Any
from .grid import build_board_grid


def map_pieces_to_squares(
    detections: List[Dict[str, Any]],
    board_size: int = 800,
    debug: bool = False
) -> Dict[str, str]:
    """
    Map detection bboxes to algebraic squares using overlap area.
    
    Fixes:
    1. Center-point rounding errors (g8 knight bleeding into e5)
    2. Multi-square bbox assignment (picks square with most overlap)
    3. Edge-case bbox clamping
    """
    
    grid = build_board_grid(board_size)
    square_map: Dict[str, Tuple[str, float]] = {}

    for det in detections:
        label = det.get("label")
        bbox = det.get("bbox")
        
        if label is None or bbox is None:
            continue
        
        try:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        except Exception:
            continue
        
        # Clamp to board bounds
        x1 = max(0, min(x1, board_size))
        y1 = max(0, min(y1, board_size))
        x2 = max(0, min(x2, board_size))
        y2 = max(0, min(y2, board_size))
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Compute overlaps with all squares
        overlaps = _compute_overlaps(bbox=(x1, y1, x2, y2), grid=grid)
        
        if not overlaps:
            continue
        
        # Assign to square with largest overlap
        best_square, best_overlap = overlaps[0]
        
        prev = square_map.get(best_square)
        if prev is None or best_overlap > prev[1]:
            square_map[best_square] = (label, best_overlap)
    
    return {sq: lbl_area[0] for sq, lbl_area in square_map.items()}


def _compute_overlaps(
    bbox: Tuple[int, int, int, int],
    grid: Dict[str, Tuple[int, int, int, int]]
) -> List[Tuple[str, float]]:
    """
    Compute intersection area between bbox and each grid square.
    Returns list sorted by overlap area (descending).
    """
    x1, y1, x2, y2 = bbox
    overlaps = []
    
    for square, (sx1, sy1, sx2, sy2) in grid.items():
        # Intersection bounds
        ix1 = max(x1, sx1)
        iy1 = max(y1, sy1)
        ix2 = min(x2, sx2)
        iy2 = min(y2, sy2)
        
        # Check for valid intersection
        if ix2 > ix1 and iy2 > iy1:
            overlap_area = float((ix2 - ix1) * (iy2 - iy1))
            overlaps.append((square, overlap_area))
    
    overlaps.sort(key=lambda x: x[1], reverse=True)
    return overlaps