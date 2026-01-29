import numpy as np
from typing import List, Dict, Any, Optional

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

def decode_leyolo_outputs(
    raw_output: np.ndarray,
    conf_threshold: float = 0.25,
    top_k: Optional[int] = 100
) -> List[Dict[str, Any]]:
    # Validate input shape
    if raw_output.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got ndim={raw_output.ndim}")
    if raw_output.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {raw_output.shape[0]}")
    
    num_channels = raw_output.shape[1]
    num_anchors = raw_output.shape[2]
    
    # Layout: [cx, cy, w, h, class_logits...] — NO separate objectness for this model
    # ChessCam: channels = 4 (box) + 12 (classes) = 16
    num_classes = num_channels - 4
    
    if num_classes != len(LABELS):
        raise ValueError(f"Model has {num_classes} classes but LABELS has {len(LABELS)} entries")
    
    # Transpose from (1, 16, 2835) to (2835, 16)
    arr = raw_output.squeeze(0).astype(np.float32).T  # shape (2835, 16)
    
    # Extract box coordinates (normalized to model input 480x288)
    boxes_raw_all = arr[:, 0:4]  # cx, cy, w, h
    
    # Extract class logits and apply sigmoid
    class_logits_all = arr[:, 4:4 + num_classes]
    class_probs_all = _sigmoid(class_logits_all)
    
    # Confidence = max class probability (no separate objectness)
    max_class_probs_all = class_probs_all.max(axis=1)
    class_indices_all = class_probs_all.argmax(axis=1)
    
    confidences_all = max_class_probs_all
    keep_mask = confidences_all >= conf_threshold
    num_kept = int(np.sum(keep_mask))
    # print(f"[decode] Anchors above threshold ({conf_threshold}): {num_kept} / {len(keep_mask)}")  # Disabled for cleaner output
    
    detections: List[Dict[str, Any]] = []
    if num_kept == 0:
        return detections
    
    # Convert mask to indices and select top-k by confidence
    kept_indices = np.nonzero(keep_mask)[0]
    kept_scores = confidences_all[kept_indices]
    order = np.argsort(-kept_scores)  # descending
    if top_k is not None and top_k > 0:
        selected = kept_indices[order[:min(top_k, len(order))]]
    else:
        selected = kept_indices[order]

    # Single concise log per frame (after applying top_k)
    # print(f"[decode] anchors above threshold: {num_kept}, after top_k: {len(selected)}")  # Disabled for cleaner output

    # Slice arrays by selected indices
    boxes_raw = boxes_raw_all[selected]
    confidences = confidences_all[selected]
    class_indices = class_indices_all[selected]
    
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
