import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import argparse
import cv2  # type: ignore
# pyright: reportAttributeAccessIssue=false
import numpy as np

from src.fen_builder import build_fen_board
from src.piece_mapper import map_pieces_to_squares


# Ensure src is importable
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from video_reader import VideoReader  # type: ignore
from preprocess import crop_top_half  # type: ignore
from corner_picker import pick_corners_interactive  # type: ignore
from corner_finder import find_board_corners  # type: ignore

def order_corners_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    # Robustly order corners: top-left, top-right, bottom-right, bottom-left
    # Sort by y (top two first), then by x within each row.
    pts = pts.astype(np.float32)
    idx_by_y = np.argsort(pts[:, 1])
    top = pts[idx_by_y[:2]]
    bottom = pts[idx_by_y[2:]]

    top_sorted = top[np.argsort(top[:, 0])]
    bottom_sorted = bottom[np.argsort(bottom[:, 0])]

    tl, tr = top_sorted[0], top_sorted[1]
    bl, br = bottom_sorted[0], bottom_sorted[1]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def is_square_light(board_img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    """Check if the square region is light (True) or dark (False) based on mean intensity."""
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    margin = 10
    region = board_img[max(0, cy - margin):min(board_img.shape[0], cy + margin), 
                       max(0, cx - margin):min(board_img.shape[1], cx + margin)]
    if region.size == 0:
        return False
    if len(region.shape) == 3:
        region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(region)
    return bool(mean_intensity > 127)


def check_board_orientation(board_img: np.ndarray) -> bool:
    """
    Return True if the board is oriented with a1 dark, a8 light, h1 light, h8 dark.
    Uses relative brightness of the four corner squares (no fixed threshold).
    """
    h, w = board_img.shape[:2]
    board_size = min(h, w)
    sq = board_size // 8

    def mean_at_square(x1, y1, x2, y2):
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        margin = max(3, sq // 16)
        region = board_img[max(0, cy - margin):min(h, cy + margin),
                           max(0, cx - margin):min(w, cx + margin)]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        return float(np.mean(gray)) if gray.size else 0.0

    a1 = mean_at_square(0, board_size - sq, sq, board_size)
    a8 = mean_at_square(0, 0, sq, sq)
    h1 = mean_at_square(board_size - sq, board_size - sq, board_size, board_size)
    h8 = mean_at_square(board_size - sq, 0, board_size, sq)

    vals = {'a1': a1, 'a8': a8, 'h1': h1, 'h8': h8}
    # Two darkest should be a1 and h8; two brightest should be a8 and h1
    sorted_keys = sorted(vals, key=lambda k: vals[k])
    darkest = set(sorted_keys[:2])
    brightest = set(sorted_keys[2:])
    ok = darkest == {'a1', 'h8'} and brightest == {'a8', 'h1'}

    # Require a minimal contrast to avoid noise (adaptive)
    contrast = (np.mean([vals[k] for k in brightest]) - np.mean([vals[k] for k in darkest]))
    return ok


def normalize_board_orientation(board_img: np.ndarray) -> np.ndarray:
    """
    Rotate by 0/90/180/270 to achieve typical orientation (a1 dark, a8 light, h1 light, h8 dark).
    """
    candidates = [
        board_img,
        cv2.rotate(board_img, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(board_img, cv2.ROTATE_180),
        cv2.rotate(board_img, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    for img in candidates:
        try:
            if check_board_orientation(img):
                return img
        except Exception:
            continue
    # Fallback: keep original if no confident orientation found
    return board_img


def warp_board(frame: np.ndarray, corners: np.ndarray, output_size: Tuple[int, int] = (800, 800)) -> Optional[np.ndarray]:
    """Warp frame using provided corners to a square output. Returns None on failure."""
    # Validate corners
    if corners is None:
        print("Error: Corners are None")
        return None
    if corners.shape != (4, 2):
        print(f"Error: Invalid corners shape: expected (4, 2), got {corners.shape}")
        return None
    
    # Additional validation: check if corners are within frame bounds and not collinear
    h, w = frame.shape[:2]
    corners_flat = corners.reshape(-1, 2)
    if np.any(corners_flat < 0) or np.any(corners_flat[:, 0] >= w) or np.any(corners_flat[:, 1] >= h):
        print("Error: Corners are out of frame bounds")
        return None
    
    # Check if points are approximately collinear (area of triangle formed by first 3 points is near zero)
    pts = corners_flat[:3]
    area = 0.5 * abs((pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1]) - (pts[2][0] - pts[0][0]) * (pts[1][1] - pts[0][1]))
    if area < 1e-3:
        print("Error: Corners are collinear or nearly collinear")
        return None
    
    # Ensure corners are properly ordered and valid
    src_pts = order_corners_tl_tr_br_bl(corners.astype(np.float32))

    # Define destination corners for square output
    width, height = output_size
    dst_pts = np.array([
        [0.0, 0.0],                    # top-left
        [width - 1.0, 0.0],            # top-right
        [width - 1.0, height - 1.0],   # bottom-right
        [0.0, height - 1.0],           # bottom-left
    ], dtype=np.float32)

    try:
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(frame, M, output_size)
        return warped
    except Exception as e:
        print(f"Error during perspective warping: {e}")
        return None
    

def get_board_corners(
    frame: np.ndarray, 
    cached_corners: Optional[np.ndarray] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Get board corners using:
    1. Cached corners (if provided)
    2. Automatic detection (CameraChessWeb-inspired)
    3. Manual selection (fallback)

    Returns (corners, debug_img)
    """
    # Use cached corners if available
    if cached_corners is not None:
        return cached_corners, None

    # Try automatic detection
    corners, debug_img = find_board_corners(frame, debug=True)

    if corners is not None:
        print("Automatic board detection successful.")
        return corners, debug_img

    # Do not show UI here; main() controls all visualization
    print("Automatic board detection failed. Please select board corners manually.")
    print("Click corners in order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
    manual_corners = pick_corners_interactive(frame)
    return manual_corners, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chess PGN from video: Extract and warp chessboard from video frames."
    )
    parser.add_argument("--video", "-v", required=True, help="Path to input video file")
    parser.add_argument("--interval", type=int, default=30, help="Sample every N frames (default: 30)")
    parser.add_argument("--max-frames", type=int, default=10, help="Maximum frames to process (default: 10)")
    parser.add_argument("--no-debug", action="store_true", help="Skip debug windows and visualization")
    # removed --rotate temporary flag
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Initialize video reader
    reader = VideoReader(
        path=args.video,
        sample_interval_frames=15,
        max_frames=None
    )

    frames = reader.read()
    print(f"Read {len(frames)} frames")

    cached_corners: Optional[np.ndarray] = None

    for i, frame in enumerate(frames):
        cropped_frame = crop_top_half(frame)

        corners, debug_img = get_board_corners(cropped_frame, cached_corners)

        if corners is None:
            print(f"Frame {i}: Corner selection cancelled. Exiting.")
            break

        if cached_corners is None:
            cached_corners = corners
            print(f"Corners cached for subsequent frames:\n{corners}")

        board_img = warp_board(cropped_frame, corners)

        if board_img is None:
            print(f"Frame {i}: Warping failed. Skipping frame.")
            continue

        # Rotate 90° counter-clockwise to match standard chess orientation
        board_img = cv2.rotate(board_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        print(f"Frame {i}: Warped board shape = {board_img.shape}")

        if i == 0:
            cv2.imwrite("warped_board.png", board_img)  # temp verification

        if not args.no_debug:
            if i == 0:
                if debug_img is not None:
                    cv2.imshow("Board Detection Debug", debug_img)
                    cv2.waitKey(0)

                cv2.imshow("Warped Board", board_img)
                cv2.waitKey(0)
            else:
                cv2.imshow("Warped Board", board_img)
                key = cv2.waitKey(500)
                if key == ord('q'):
                    break
    if not args.no_debug:
        cv2.destroyAllWindows()
    print("Processing complete.")


if __name__ == "__main__":
    main(parse_args())
