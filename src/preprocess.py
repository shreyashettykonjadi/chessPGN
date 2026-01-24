import cv2
import numpy as np
from typing import Any, Optional, Tuple


class Preprocess:
    """
    Simple preprocessing for video frames.

    Methods:
    - process_frame: accepts a NumPy BGR image and returns a processed NumPy image.
      Options: resize, convert to grayscale, normalize to [0,1].
    """

    def __init__(self, resize: Optional[Tuple[int, int]] = None, to_gray: bool = False, normalize: bool = False):
        self.resize = resize
        self.to_gray = to_gray
        self.normalize = normalize

    def process_frame(self, frame: Any) -> np.ndarray:
        """
        Process a single frame and return a NumPy array.

        Raises:
            TypeError if frame is not a NumPy array.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy ndarray")

        img = frame
        if self.resize is not None:
            img = cv2.resize(img, (self.resize[0], self.resize[1]), interpolation=cv2.INTER_AREA)

        if self.to_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.normalize:
            img = img.astype("float32") / 255.0

        return img
