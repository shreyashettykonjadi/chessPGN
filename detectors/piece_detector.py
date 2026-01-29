# pyright: reportMissingImports=false
# pyright: reportAttributeAccessIssue=false
"""
Minimal piece detector module.
Exposes:
  - preprocess_board(board_img) -> NCHW float32 numpy tensor
  - run_inference(input_tensor) -> list of model outputs
"""

import sys
from pathlib import Path
import numpy as np
import cv2  # type: ignore
import onnxruntime as ort

# Compute project root locally (do NOT import test modules)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ONNX model path
ONNX_MODEL_PATH = PROJECT_ROOT / "models" / "pieces" / "480M_leyolo_pieces.onnx"
print(f"[infer] Using model path: {ONNX_MODEL_PATH}")

# Model input size expected by the detector
MODEL_WIDTH = 480
MODEL_HEIGHT = 288

# Module-level session cache
_session = None
_input_name = None
_output_names = None


def preprocess_board(board_img: np.ndarray) -> np.ndarray:
    """
    Preprocess warped board image for ONNX model.
    - Resize to MODEL_WIDTH x MODEL_HEIGHT
    - BGR -> RGB
    - Scale to [0,1]
    - Return NCHW float32 tensor (1, 3, H, W)
    """
    if board_img is None:
        raise ValueError("board_img is None")
    # Resize
    img = cv2.resize(board_img, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_LINEAR)
    # BGR to RGB
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize
    img = img.astype(np.float32) / 255.0
    # HWC -> CHW
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    chw = np.transpose(img, (2, 0, 1))
    tensor = np.expand_dims(chw, axis=0).astype(np.float32)
    return tensor


def run_inference(input_tensor: np.ndarray) -> list:
    """
    Run ONNX inference on input tensor.
    Lazily loads model on first call.
    Returns list of output arrays.
    """
    global _session, _input_name, _output_names
    if _session is None:
        model_path = ONNX_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        _session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        _input_name = _session.get_inputs()[0].name
        _output_names = [o.name for o in _session.get_outputs()]
        print(f"[infer] Loaded ONNX model: {model_path}")
        print(f"[infer] Input name: {_input_name}, output(s): {_output_names}")
    # Ensure input dtype matches FP16 model expectation
    try:
        if input_tensor.dtype != np.float16:
            input_tensor = input_tensor.astype(np.float16)
    except Exception:
        # Fallback: attempt to cast via numpy
        input_tensor = np.asarray(input_tensor).astype(np.float16)
    return _session.run(_output_names, {_input_name: input_tensor})