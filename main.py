# pyright: reportMissingImports=false
# pyright: reportAttributeAccessIssue=false

import os
import sys
import traceback
from pathlib import Path
import hashlib  # Add at top with other imports

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
from src.pgn_writer import write_pgn
from src.stable_snapshot_detector import StableSnapshotDetector

# Ensure src is importable
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from video_reader import VideoReader  # type: ignore
from preprocess import crop_top_half  # type: ignore
from corner_picker import pick_corners_interactive  # type: ignore
from corner_finder import find_board_corners  # type: ignore

# Insert warm-up and filtering parameters near top-level (just after imports / constants)
WARMUP_FRAMES = 12           # ignore first N frames for FEN output/logging
CONF_THRESHOLD = 0.5         # minimum confidence to accept a raw detection
ENABLE_CHESS_PRIOR = True    # apply light gating during warm-up only
CENTRAL_SQUARES = {"e4", "d4", "e5", "d5"}
MAX_FRAMES = 100             # hard early-stop to prevent process from being killed

# Temporary debug toggle for move inference diagnostics
debug_move_inference = True
debug_fen_timeline = True  # Enable FEN transition validation debug
debug_pipeline_trace = True  # Trace raw model output variation


def get_board_fen(full_fen: str) -> str:
    """Extract only the board portion from a full FEN string."""
    return full_fen.split()[0]


def count_square_differences(board_fen1: str, board_fen2: str) -> int:
    """
    Count how many squares differ between two board FEN strings.
    Returns the number of differing squares (0-64).
    """
    def expand_fen(fen: str) -> List[str]:
        """Expand a board FEN to a list of 64 square contents."""
        squares = []
        for char in fen:
            if char == '/':
                continue
            elif char.isdigit():
                squares.extend(['1'] * int(char))  # '1' represents empty
            else:
                squares.append(char)
        return squares
    
    squares1 = expand_fen(board_fen1)
    squares2 = expand_fen(board_fen2)
    
    if len(squares1) != 64 or len(squares2) != 64:
        return 64  # Invalid FEN, max difference
    
    return sum(1 for a, b in zip(squares1, squares2) if a != b)


def get_piece_at_square(board_fen: str, square: str) -> str:
    """
    Get the piece at a given square from a board FEN string.
    Returns the piece character or '1' for empty.
    """
    def expand_fen(fen: str) -> List[str]:
        squares = []
        for char in fen:
            if char == '/':
                continue
            elif char.isdigit():
                squares.extend(['1'] * int(char))
            else:
                squares.append(char)
        return squares
    
    squares = expand_fen(board_fen)
    if len(squares) != 64:
        return '?'
    
    file_idx = ord(square[0]) - ord('a')  # 0-7
    rank_idx = 8 - int(square[1])  # 0-7 (rank 8 = index 0)
    idx = rank_idx * 8 + file_idx
    return squares[idx] if 0 <= idx < 64 else '?'


def get_king_squares(board_fen: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Get the squares of white and black kings from a board FEN.
    Returns (white_king_square, black_king_square).
    """
    def expand_fen(fen: str) -> List[str]:
        squares = []
        for char in fen:
            if char == '/':
                continue
            elif char.isdigit():
                squares.extend(['1'] * int(char))
            else:
                squares.append(char)
        return squares
    
    squares = expand_fen(board_fen)
    white_king = None
    black_king = None
    
    for idx, piece in enumerate(squares):
        file_char = chr(ord('a') + (idx % 8))
        rank_char = str(8 - (idx // 8))
        sq = file_char + rank_char
        if piece == 'K':
            white_king = sq
        elif piece == 'k':
            black_king = sq
    
    return white_king, black_king


def infer_moves_from_fens(fen_history: List[str], debug: bool = False) -> List[str]:
    """
    Infer chess moves from consecutive FEN positions.

    Noise-tolerant matching:
      - Allow up to 2 mismatched squares between expected and detected boards.
      - Require the move's from-square and to-square to match exactly.
      - Require king squares to match exactly.

    When debug=True, print diagnostics per FEN pair.
    """
    MAX_SQUARE_TOLERANCE = 2  # Allow up to 2 mismatched squares
    
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

        matching_moves: List[Tuple[chess.Move, int]] = []  # (move, diff_count)
        target_wk, target_bk = get_king_squares(target_board)

        for move in legal_moves:
            board.push(move)
            result_board = get_board_fen(board.fen())
            board.pop()

            # Check king squares match exactly
            result_wk, result_bk = get_king_squares(result_board)
            if result_wk != target_wk or result_bk != target_bk:
                continue

            # Check move's from and to squares match exactly
            from_sq = move.uci()[:2]
            to_sq = move.uci()[2:4]
            
            # The from-square should be empty in target (piece moved away)
            # The to-square should have the moved piece in target
            result_from = get_piece_at_square(result_board, from_sq)
            target_from = get_piece_at_square(target_board, from_sq)
            result_to = get_piece_at_square(result_board, to_sq)
            target_to = get_piece_at_square(target_board, to_sq)
            
            if result_from != target_from or result_to != target_to:
                continue

            # Count total square differences
            diff_count = count_square_differences(result_board, target_board)
            
            if diff_count <= MAX_SQUARE_TOLERANCE:
                matching_moves.append((move, diff_count))

        if debug:
            print(f"[move-debug] Matching moves found: {len(matching_moves)}")

        if len(matching_moves) == 1:
            move_list.append(matching_moves[0][0].uci())
            if debug:
                print(f"[move-debug] Accepted move: {matching_moves[0][0].uci()} (diff={matching_moves[0][1]})")
        elif len(matching_moves) > 1:
            # Pick the one with fewest differences
            matching_moves.sort(key=lambda x: x[1])
            if matching_moves[0][1] < matching_moves[1][1]:
                # Clear winner
                move_list.append(matching_moves[0][0].uci())
                if debug:
                    print(f"[move-debug] Accepted best match: {matching_moves[0][0].uci()} (diff={matching_moves[0][1]})")
            else:
                if debug:
                    print(f"[move-debug] Ambiguous: {len(matching_moves)} matching moves with same diff (skipping)")
        else:
            if debug:
                print("[move-debug] No legal move matched this FEN transition")

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
    2. Manual selection (automatic detection skipped for speed)

    Returns (corners, debug_img)
    """
    # Use cached corners if available
    if cached_corners is not None:
        return cached_corners, None

    # Skip automatic detection - go straight to manual selection
    print("Please select board corners manually.")
    print("Click corners in this order: 1) Top-Left, 2) Top-Right, 3) Bottom-Right, 4) Bottom-Left")
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
    parser.add_argument("--output", "-o", type=str, default=None, help="Output PGN file path (default: <video_name>.pgn)")
    # removed --rotate temporary flag
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Initialize video reader
    reader = VideoReader(
        path=args.video,
        sample_interval_frames=3,
        max_frames=None
    )

    frames = reader.read()

    cached_corners: Optional[np.ndarray] = None
    
    # NEW: Stable snapshot detector (replaces FENExtractor/FENTimeline approach)
    # Reduced threshold to 3 for more responsive detection in noisy video
    stable_detector = StableSnapshotDetector(stability_threshold=3, debug=True)

    for i, frame in enumerate(frames):
        # Hard early-stop to prevent process from being killed
        if i >= MAX_FRAMES:
            print(f"[main] Early stop: frame limit ({MAX_FRAMES}) reached")
            break

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

            # DIAGNOSTIC: hash raw ONNX output (disabled for cleaner output)
            # try:
            #     raw_arr = raw_output if isinstance(raw_output, np.ndarray) else np.array(raw_output)
            #     raw_hash = hashlib.md5(raw_arr.tobytes()).hexdigest()[:8]
            # except Exception:
            #     raw_hash = "err"
            # print(f"[trace] frame={i} raw_output_hash={raw_hash}")

            detections = decode_leyolo_outputs(raw_output)

            def _conf(d):
                return float(d.get("score", 1.0)) if isinstance(d, dict) else 1.0
            filtered = [d for d in detections if _conf(d) >= CONF_THRESHOLD]

            # Map filtered detections to squares
            piece_map_candidate = map_pieces_to_squares(filtered, board_size=800, debug=False)

            # --- NEW: Stable snapshot detection approach ---
            # Process all frames (warm-up just skips logging, but we need frames for stability)
            detected_move = stable_detector.process_frame(i, piece_map_candidate)
            if detected_move:
                print(f"[main] ✓ Move detected at frame {i}: {detected_move}")

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
    
    # NEW: Get moves from stable snapshot detector
    move_list = stable_detector.get_detected_moves()
    
    print(f"\n=== STABLE SNAPSHOT DETECTION RESULTS ===")
    print(f"Total moves detected: {len(move_list)}")
    
    # Print moves for debugging
    if move_list:
        print("Moves detected:")
        for i, m in enumerate(move_list, 1):
            print(f"  {i}. {m}")
    
    # Write PGN file if moves were detected
    if move_list:
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            video_path = Path(args.video)
            output_path = video_path.with_suffix('.pgn')
        
        # Extract video name for PGN headers
        video_name = Path(args.video).stem
        
        # Write PGN using detected moves
        write_pgn(
            move_list,
            str(output_path),
            event=f"Chess Game",
            site=video_name,
            white="White",
            black="Black",
            result="*"
        )
        print(f"\nPGN file written to: {output_path}")
    else:
        print("\nNo moves detected. PGN file not generated.")

if __name__ == "__main__":
    main(parse_args())
