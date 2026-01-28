import numpy as np
from src.piece_decoder import decode_leyolo_outputs, LABELS

def test_decoder_shape_and_classes():
    # Fake model output: (1, 16, 2835)
    raw = np.random.randn(1, 16, 2835).astype(np.float32)

    detections = decode_leyolo_outputs(raw, conf_threshold=0.0)

    # Check labels count (MUST be 11)
    assert len(LABELS) == 11, f"Expected 11 labels, got {len(LABELS)}"

    # Each detection must have required keys
    for det in detections:
        assert "label" in det
        assert "bbox" in det
        assert "score" in det
        assert len(det["bbox"]) == 4
