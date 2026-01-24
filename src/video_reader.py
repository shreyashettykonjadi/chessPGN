import cv2
import numpy as np
from typing import List, Optional


class VideoReader:
    """
    Video reader that returns sampled frames as NumPy arrays.

    Arguments:
    - path: path to video file
    - sample_interval_frames: sample every N frames (preferred if provided)
    - sample_interval_seconds: alternative sampling by seconds (used if sample_interval_frames is None)
    - max_frames: optional cap on number of frames to return
    """

    def __init__(
        self,
        path: str,
        sample_interval_frames: Optional[int] = None,
        sample_interval_seconds: Optional[float] = None,
        max_frames: Optional[int] = None,
    ):
        self.path = path
        self.sample_interval_frames = sample_interval_frames
        self.sample_interval_seconds = sample_interval_seconds
        self.max_frames = max_frames

    def read(self) -> List[np.ndarray]:
        """
        Read video and return a list of sampled frames as NumPy arrays (BGR color as returned by OpenCV).

        If both sample_interval_frames and sample_interval_seconds are None, defaults to every 30 frames.
        """
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {self.path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.sample_interval_frames is None:
            if self.sample_interval_seconds is not None:
                interval_frames = max(1, int(round(self.sample_interval_seconds * fps)))
            else:
                interval_frames = 30
        else:
            interval_frames = max(1, int(self.sample_interval_frames))

        frames: List[np.ndarray] = []
        idx = 0
        collected = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval_frames == 0:
                # ensure we store a copy to avoid accidental OpenCV memory reuse
                frames.append(frame.copy())
                collected += 1
                if self.max_frames is not None and collected >= self.max_frames:
                    break
            idx += 1

        cap.release()
        return frames

if __name__ == "__main__":
    vr = VideoReader(
        path="data/videos/game_1.mp4",
        sample_interval_frames=30,
        max_frames=5,
    )

    frames = vr.read()
    print(f"Collected {len(frames)} frames")

    for i, f in enumerate(frames):
        print(i, f.shape, f.dtype)
