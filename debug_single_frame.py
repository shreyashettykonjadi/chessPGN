import cv2
from src.detect import process_frame

cap = cv2.VideoCapture("data/videos/game_1.mp4")
ret, frame = cap.read()
cap.release()

assert ret, "Could not read frame"

result = process_frame(frame)
print(result)
print(f"Detected FEN: {result.fen}")