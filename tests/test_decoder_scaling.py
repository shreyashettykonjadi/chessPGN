import numpy as np
from src.piece_decoder import decode_leyolo_outputs

def test_decoder_bbox_scaling():
    raw = np.zeros((1, 16, 2835), dtype=np.float32)

    # One confident detection
    raw[0, 0, 0] = 0.5   # cx
    raw[0, 1, 0] = 0.5   # cy
    raw[0, 2, 0] = 0.2   # w
    raw[0, 3, 0] = 0.2   # h
    raw[0, 4, 0] = 10.0  # objectness (high)
    raw[0, 5, 0] = 10.0  # class logit (high)

    detections = decode_leyolo_outputs(raw, conf_threshold=0.1)
    assert len(detections) == 1

    x1, y1, x2, y2 = detections[0]["bbox"]

    # Must be inside 800x800 board
    assert 0 <= x1 < 800
    assert 0 <= x2 <= 800
    assert 0 <= y1 < 800
    assert 0 <= y2 <= 800

    # Box must have positive area
    assert x2 > x1
    assert y2 > y1
