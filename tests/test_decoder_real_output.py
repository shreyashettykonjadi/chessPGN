import onnxruntime as ort
import numpy as np
from src.piece_decoder import decode_leyolo_outputs

MODEL_PATH = "models/pieces/480M_leyolo_pieces.onnx"

def test_real_model_decode():
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name

    dummy = np.random.rand(1, 3, 288, 480).astype(np.float16)

    outputs = session.run(None, {input_name: dummy})
    raw = outputs[0]

    detections = decode_leyolo_outputs(raw, conf_threshold=0.5)

    # Should not crash and should return a list
    assert isinstance(detections, list)
