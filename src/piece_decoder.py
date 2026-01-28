import numpy as np
from typing import List, Dict, Any

# ChessCam uses 12 classes: ["b", "k", "n", "p", "q", "r", "B", "K", "N", "P", "Q", "R"]
# Mapped to readable names
LABELS = [
    "black_bishop", "black_king", "black_knight", "black_pawn",
    "black_queen", "black_rook",
    "white_bishop", "white_king", "white_knight", "white_pawn",
    "white_queen", "white_rook"
]

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)  # Prevent overflow
    return 1.0 / (1.0 + np.exp(-x))

def decode_leyolo_outputs(raw_output: np.ndarray, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
    # Validate input shape
    if raw_output.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got ndim={raw_output.ndim}")
    if raw_output.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {raw_output.shape[0]}")
    
    num_channels = raw_output.shape[1]
    num_anchors = raw_output.shape[2]
    
    print(f"[decode] Input shape: {raw_output.shape} (batch=1, channels={num_channels}, anchors={num_anchors})")
    
    # Layout: [cx, cy, w, h, class_logits...] — NO separate objectness for this model
    # ChessCam: channels = 4 (box) + 12 (classes) = 16
    num_classes = num_channels - 4
    
    if num_classes != len(LABELS):
        raise ValueError(f"Model has {num_classes} classes but LABELS has {len(LABELS)} entries")
    
    print(f"[decode] Detected {num_classes} classes")
    
    # Transpose from (1, 16, 2835) to (2835, 16)
    arr = raw_output.squeeze(0).astype(np.float32).T  # shape (2835, 16)
    
    # Extract box coordinates (normalized to model input 480x288)
    boxes_raw = arr[:, 0:4]  # cx, cy, w, h
    
    # Extract class logits and apply sigmoid
    class_logits = arr[:, 4:4 + num_classes]
    class_probs = _sigmoid(class_logits)
    
    # Confidence = max class probability (no separate objectness)
    max_class_probs = class_probs.max(axis=1)
    class_indices = class_probs.argmax(axis=1)
    
    confidences = max_class_probs
    keep = confidences >= conf_threshold
    
    print(f"[decode] Anchors above threshold ({conf_threshold}): {np.sum(keep)} / {len(keep)}")
    
    detections: List[Dict[str, Any]] = []
    if not np.any(keep):
        return detections
    
    boxes_raw = boxes_raw[keep]
    confidences = confidences[keep]
    class_indices = class_indices[keep]
    
    # Convert from model input coords (480x288) to board image coords (800x800)
    # Model input: width=480, height=288
    # Output board: 800x800
    model_w, model_h = 480.0, 288.0
    board_size = 800.0
    
    cx = boxes_raw[:, 0] * (board_size / model_w)
    cy = boxes_raw[:, 1] * (board_size / model_h)
    bw = boxes_raw[:, 2] * (board_size / model_w)
    bh = boxes_raw[:, 3] * (board_size / model_h)
    
    x1 = np.clip(cx - bw / 2.0, 0.0, board_size).astype(np.int32)
    y1 = np.clip(cy - bh / 2.0, 0.0, board_size).astype(np.int32)
    x2 = np.clip(cx + bw / 2.0, 0.0, board_size).astype(np.int32)
    y2 = np.clip(cy + bh / 2.0, 0.0, board_size).astype(np.int32)
    
    for i in range(len(x1)):
        cls_idx = int(class_indices[i])
        label = LABELS[cls_idx] if 0 <= cls_idx < len(LABELS) else "unknown"
        detections.append({
            "label": label,
            "bbox": (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
            "score": float(confidences[i])
        })
    
    return detections
