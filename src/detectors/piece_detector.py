# pyright: reportAttributeAccessIssue=false
import os
import cv2  # type: ignore
import numpy as np
import onnxruntime as ort
from pathlib import Path

from test_piece_mapper import PROJECT_ROOT

# Cache ONNX Runtime session (created once)
_session = None
_input_name = None
_output_names = None

def load_piece_detector(model_path: str):
    """
    Load LeYOLO chess piece detector from ONNX file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    net = cv2.dnn.readNetFromONNX(
        "models/pieces/480M_leyolo_pieces_fp32.onnx"
    )

    return net

def preprocess_board(board_img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB) if len(board_img.shape) == 3 else board_img
    img = cv2.resize(img, (480, 288), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float16) / np.float16(255.0)

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    assert img.shape == (1, 3, 288, 480), f"Unexpected tensor shape: {img.shape}"
    return np.ascontiguousarray(img)

def run_inference(input_tensor: np.ndarray) -> list:
    global _session, _input_name, _output_names
    if _session is None:
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        model_path = PROJECT_ROOT / "models" / "pieces" / "480M_leyolo_pieces.onnx"
        _session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        _input_name = _session.get_inputs()[0].name
        _output_names = [o.name for o in _session.get_outputs()]
    return _session.run(_output_names, {_input_name: input_tensor})
