# pyright: reportMissingImports=false
# pyright: reportAttributeAccessIssue=false

import os
import sys
import traceback
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

# now safe to import everything
from typing import Optional, Tuple, List
import argparse
import cv2  # type: ignore
import numpy as np
import chess  # python-chess library

from src.fen_builder import build_fen_board
from src.piece_mapper import map_pieces_to_squares
from piece_decoder import decode_leyolo_outputs  # type: ignore
from detectors.piece_detector import preprocess_board, run_inference  # type: ignore
from src.fen_extractor import FENExtractor
from src.fen_timeline import FENTimeline
from src.grid import square_from_point

# Ensure src is importable
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from video_reader import VideoReader  # type: ignore
from preprocess import crop_top_half  # type: ignore
from corner_picker import pick_corners_interactive  # type: ignore
from corner_finder import find_board_corners  # type: ignore

# Insert warm-up and filtering parameters near top-level (just after imports / constants)
WARMUP_FRAMES = 8            # reduced from 12 since we sample more densely now
CONF_THRESHOLD = 0.5         # minimum confidence to accept a raw detection
ENABLE_CHESS_PRIOR = True    # apply light gating during warm-up only
CENTRAL_SQUARES = {"e4", "d4", "e5", "d5"}

# Temporary debug toggle for move inference diagnostics
debug_move_inference = True
debug_fen_timeline = True  # Enable FEN transition validation debug


def get_board_fen(full_fen: str) -> str:
    """Extract only the board portion from a full FEN string."""
    return full_fen.split()[0]


def infer_moves_from_fens(fen_history: List[str], debug: bool = False) -> List[str]:
    """
    Infer chess moves from consecutive FEN positions.

    When debug=True, print diagnostics per FEN pair:
      - previous FEN
      - current FEN
      - number of legal moves tried
      - message when no legal move matched
    """
    if len(fen_history) < 2:
        return []
    
    move_list: List[str] = []
    
    for i in range(len(fen_history) - 1):
        fen_before = fen_history[i]
        fen_after = fen_history[i + 1]
        
        # Extract board-only portions for comparison
        target_board = get_board_fen(fen_after)
        
        try:
            board = chess.Board(fen_before)
        except ValueError:
            # Invalid FEN, skip this pair
            if debug:
                print(f"[move-debug] Invalid fen_before at index {i}: {fen_before}")
            continue
        
        # Materialize legal moves to count and iterate safely
        legal_moves = list(board.legal_moves)
        if debug:
            print(f"[move-debug] Pair {i} -> {i+1}")
            print(f"[move-debug] prev_fen: {fen_before}")
            print(f"[move-debug] curr_fen: {fen_after}")
            print(f"[move-debug] legal moves to try: {len(legal_moves)}")

        matching_moves: List[chess.Move] = []

        for move in legal_moves:
            board.push(move)
            result_board = get_board_fen(board.fen())
            board.pop()

            if result_board == target_board:
                matching_moves.append(move)

        if len(matching_moves) == 1:
            move_list.append(matching_moves[0].uci())
        else:
            if debug:
                if len(matching_moves) == 0:
                    print("[move-debug] No legal move matched this FEN transition")
                else:
                    print(f"[move-debug] Ambiguous: {len(matching_moves)} matching moves (skipping)")

    return move_list


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
        # print("Automatic board detection successful.")
        return corners, debug_img

    # Do not show UI here; main() controls all visualization
    # print("Automatic board detection failed. Please select board corners manually.")
    # print("Click corners in order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
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
    # Initialize video reader with denser sampling for better move detection
    reader = VideoReader(
        path=args.video,
        sample_interval_frames=3,  # changed from 15 -> 3 for denser sampling
        max_frames=None
    )

    frames = reader.read()

    cached_corners: Optional[np.ndarray] = None
    fen_timeline = FENTimeline(validate_transitions=True, debug=debug_fen_timeline)
    fen_extractor = FENExtractor(window_size=5, strict_validation=True)

    for i, frame in enumerate(frames):
        cropped_frame = crop_top_half(frame)

        corners, debug_img = get_board_corners(cropped_frame, cached_corners)

        if corners is None:
            break

        if cached_corners is None:
            cached_corners = corners

        board_img = warp_board(cropped_frame, corners)

        if board_img is None:
            continue

        # Rotate 90° counter-clockwise to match standard chess orientation
        board_img = cv2.rotate(board_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # --- Phase 4.5: Detect pieces, map to squares, and build FEN ---
        try:
            input_tensor = preprocess_board(board_img)
            outputs = run_inference(input_tensor)
            raw_output = outputs[0] if isinstance(outputs, list) else outputs

            # Pass debug=False to silence per-frame decode logs
            detections = decode_leyolo_outputs(raw_output, debug=False)

            def _conf(d):
                return float(d.get("score", 1.0)) if isinstance(d, dict) else 1.0
            filtered = [d for d in detections if _conf(d) >= CONF_THRESHOLD]

            # Map filtered detections to squares
            piece_map_candidate = map_pieces_to_squares(filtered, board_size=800, debug=False)

            # Warm-up: apply chess prior gating, skip FEN logging
            if i < WARMUP_FRAMES:
                if ENABLE_CHESS_PRIOR and piece_map_candidate:
                    for sq in list(piece_map_candidate.keys()):
                        piece = piece_map_candidate.get(sq)
                        if not piece:
                            continue
                        parts = piece.split("_")
                        piece_type = parts[-1] if len(parts) >= 2 else ""
                        rank = int(sq[1]) if len(sq) >= 2 and sq[1].isdigit() else None
                        if piece_type != "pawn" and (rank in (3, 4, 5, 6) or sq in CENTRAL_SQUARES):
                            del piece_map_candidate[sq]
                # Feed to extractor to fill temporal buffer, but don't log
                fen_extractor.process_detections(piece_map_candidate)
            else:
                # Post warm-up: collect stabilized FENs via timeline
                fen = fen_extractor.process_detections(piece_map_candidate)
                new_fen = fen_timeline.collect(fen)
                if new_fen:
                    print(f"[STATE CHANGE] {new_fen}")

        except Exception as e:
            print(f"[main] ERROR in Phase 4.5: {e}")
            traceback.print_exc()
            break
        # --- End Phase 4.5 ---

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
    
    # Phase 5.1: Print FEN timeline summary
    print(f"Total stable FENs collected: {len(fen_timeline)}")
    
    # Phase 5.3: Extract moves inferred during validation
    entries = fen_timeline.entries()
    move_list: List[str] = []
    for fen, meta in entries:
        if meta and isinstance(meta, dict) and "uci" in meta:
            move_list.append(meta["uci"])

    print(f"Total moves inferred: {len(move_list)}")
    # Optionally print moves in order (commented out)
    # for m in move_list:
    #     print(m)

if __name__ == "__main__":
    main(parse_args())
