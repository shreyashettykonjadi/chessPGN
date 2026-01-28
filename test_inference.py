import cv2
import numpy as np

# ---- Load model ----
MODEL_PATH = "models/pieces/480M_leyolo_pieces.onnx"
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

print("Model loaded")

# ---- Fake input (same as Phase 4.2 output) ----
# 1 image, 3 channels, 480x480
dummy = np.random.rand(1, 3, 480, 480).astype(np.float32)

# ---- Run inference ----
net.setInput(dummy)
outputs = net.forward()

# ---- Inspect output ----
print("Type:", type(outputs))
print("Shape:", outputs.shape if hasattr(outputs, "shape") else "no shape")
print("Dtype:", outputs.dtype if hasattr(outputs, "dtype") else "unknown")
print("Min:", outputs.min())
print("Max:", outputs.max())
