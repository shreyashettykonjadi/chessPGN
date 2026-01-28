from typing import List, Dict, Tuple, Any
from .grid import square_from_point


def map_pieces_to_squares(
    detections: List[Dict[str, Any]],
    board_size: int = 800,
    debug: bool = False
) -> Dict[str, str]:
    """
    Map detection dicts {"label": str, "bbox": (x1,y1,x2,y2)} to algebraic squares.
    If multiple detections map to the same square, keep the one with larger bbox area.
    Returns { square: label }.
    """

    square_map: Dict[str, Tuple[str, int]] = {}  # square -> (label, area)

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

        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area = w * h

        # center of bounding box
        cx = x1 + w // 2
        cy = y1 + h // 2

        # map center point to square
        square = square_from_point(cx, cy, board_size=board_size)

        if debug:
            print(f"[DEBUG] label={label}, center=({cx},{cy}), mapped_square={square}")

        prev = square_map.get(square)
        if prev is None or area > prev[1]:
            square_map[square] = (label, area)

    return {sq: lbl_area[0] for sq, lbl_area in square_map.items()}
