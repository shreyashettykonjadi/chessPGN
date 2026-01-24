import os
import sys
from pathlib import Path
from typing import List, Any
import argparse

# ensure src is importable
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from video_reader import VideoReader
from preprocess import Preprocess

# downstream modules may be stubs; import safely
try:
    from board_detector import BoardDetector  # type: ignore
    from fen_extractor import FENExtractor  # type: ignore
    from move_inference import MoveInference  # type: ignore
    from pgn_writer import PGNWriter  # type: ignore
except Exception:
    # downstream stubs missing or broken; provide minimal placeholders
    class BoardDetector:
        def detect_board(self, frame: Any):
            return frame, (0, 0, 0, 0)

    class FENExtractor:
        def extract_fen(self, board_image: Any) -> str:
            return "8/8/8/8/8/8/8/8 w - - 0 1"

    class MoveInference:
        def infer_moves(self, fens: List[str]) -> List[str]:
            return []

    class PGNWriter:
        def write_pgn(self, moves: List[str], output_path: str) -> None:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("[Event \"Placeholder\"]\n\n")


def parse_args():
    p = argparse.ArgumentParser(description="Chess PGN from video (Phase 1: video input & preprocessing)")
    p.add_argument("--video", "-v", required=True, help="Path to input video file")
    p.add_argument("--interval-frames", type=int, default=None, help="Sample every N frames")
    p.add_argument("--interval-seconds", type=float, default=None, help="Sample every N seconds")
    p.add_argument("--max-frames", type=int, default=50, help="Max frames to read (for demo)")
    p.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=None, help="Resize frames to W H")
    p.add_argument("--gray", action="store_true", help="Convert frames to grayscale")
    p.add_argument("--normalize", action="store_true", help="Normalize frames to [0,1]")
    return p.parse_args()


def main():
    args = parse_args()

    reader = VideoReader(
        path=args.video,
        sample_interval_frames=args.interval_frames,
        sample_interval_seconds=args.interval_seconds,
        max_frames=args.max_frames,
    )
    frames = reader.read()  # List of NumPy arrays

    pre = Preprocess(resize=tuple(args.resize) if args.resize else None, to_gray=args.gray, normalize=args.normalize)

    print(f"Read {len(frames)} frames from {args.video}")
    for i, frame in enumerate(frames):
        try:
            proc = pre.process_frame(frame)
        except Exception as e:
            print(f"Skipping frame {i} due to error: {e}")
            continue
        # Demonstration: print basic info about processed frame
        if hasattr(proc, "shape"):
            print(f"Frame {i}: shape={proc.shape}, dtype={proc.dtype}")
        else:
            print(f"Frame {i}: processed (no shape)")

    # leave rest of pipeline intact (stubs or real implementations)
    print("Preprocessing demo complete.")


if __name__ == "__main__":
    main()