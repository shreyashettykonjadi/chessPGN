import numpy as np
import onnxruntime as ort

MODEL_PATH = "models/pieces/480M_leyolo_pieces.onnx"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("Model loaded with onnxruntime")

inputs = session.get_inputs()
print("Inputs:", [(i.name, i.shape, i.type) for i in inputs])

input_name = inputs[0].name

dummy = np.random.rand(1, 3, 288, 480).astype(np.float16)

outputs = session.run(None, {input_name: dummy})

print("Number of outputs:", len(outputs))
for i, out in enumerate(outputs):
    print(f"Output {i}: shape={out.shape}, dtype={out.dtype}")
