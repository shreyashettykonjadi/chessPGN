import cv2
import numpy as np
from typing import Any, Optional, Tuple


class Preprocess:
    """
    Simple preprocessing for video frames with optional top-crop.

    Usage:
      pre = Preprocess(resize=(W,H), to_gray=False, normalize=False, crop_top_ratio=None)
      img = pre.process_frame(frame)
      # or call pre.crop_top(frame, ratio) directly
    """

    def __init__(
        self,
        resize: Optional[Tuple[int, int]] = None,
        to_gray: bool = False,
        normalize: bool = False,
        crop_top_ratio: Optional[float] = None,
    ):
        self.resize = resize
        self.to_gray = to_gray
        self.normalize = normalize
        self.crop_top_ratio = None if crop_top_ratio is None else float(crop_top_ratio)

    def crop_top(self, frame: Any, ratio: Optional[float] = None) -> np.ndarray:
        """
        Crop the top portion of the frame.

        Args:
            frame: input BGR image as NumPy array
            ratio: fraction of image height to keep from the top (0 < ratio <= 1).
                   If None, uses the instance's crop_top_ratio. If that is None, returns original frame.
        Returns:
            Cropped image (top portion).
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy ndarray")

        r = self.crop_top_ratio if ratio is None else float(ratio)
        if r is None:
            return frame
        if not (0.0 < r <= 1.0):
            raise ValueError("ratio must be in (0, 1]")

        h = frame.shape[0]
        cut = max(1, int(round(h * r)))
        return frame[:cut, :, :].copy()

    def process_frame(self, frame: Any) -> np.ndarray:
        """
        Process a single frame and return a NumPy array.
        Applies top-crop (if configured), then resizing / gray / normalization as configured.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy ndarray")

        img = frame
        if self.crop_top_ratio is not None:
            img = self.crop_top(img)

        if self.resize is not None:
            img = cv2.resize(img, (self.resize[0], self.resize[1]), interpolation=cv2.INTER_AREA)

        if self.to_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.normalize:
            img = img.astype("float32") / 255.0

        return img


def crop_top_half(frame: Any) -> np.ndarray:
    """
    Return the top 50% of the input frame.

    Args:
        frame: input BGR image as NumPy array

    Returns:
        Cropped image containing the top half of the frame.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a NumPy ndarray")
    h = frame.shape[0]
    cut = max(1, h // 2)
    if frame.ndim == 3:
        return frame[:cut, :, :].copy()
    else:
        return frame[:cut, :].copy()
