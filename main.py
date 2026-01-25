import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import argparse
import cv2  # type: ignore
# pyright: reportAttributeAccessIssue=false
import numpy as np

# Ensure src is importable
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from video_reader import VideoReader
from preprocess import crop_top_half
from corner_picker import pick_corners_interactive
from corner_finder import find_board_corners


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
    src_pts = corners.reshape(4, 2).astype(np.float32)

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

    # Show debug image if available
    if debug_img is not None:
        cv2.imshow("Auto-detection Debug", debug_img)
        cv2.waitKey(1000)
        cv2.destroyWindow("Auto-detection Debug")

    # Fall back to manual selection
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
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Initialize video reader
    reader = VideoReader(
        path=args.video,
        sample_interval_frames=args.interval,
        max_frames=args.max_frames
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

        print(f"Frame {i}: Warped board shape = {board_img.shape}")

        if not args.no_debug:
            if i == 0:
                # Show debug and warped for first frame
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
