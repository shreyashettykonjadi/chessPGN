from src.video_reader import VideoReader
from src.preprocess import crop_top_half
from src.corner_picker import pick_corners_interactive
from src.corner_finder import find_board_corners
import cv2
import numpy as np
from typing import Optional, Tuple

def warp_board(frame: np.ndarray, corners: np.ndarray, output_size: tuple = (800, 800)) -> np.ndarray:
	"""Warp frame using provided corners to a square output."""
	dst = np.array([
		[0.0, 0.0],
		[output_size[0] - 1.0, 0.0],
		[output_size[0] - 1.0, output_size[1] - 1.0],
		[0.0, output_size[1] - 1.0],
	], dtype="float32")
	M = cv2.getPerspectiveTransform(corners, dst)
	return cv2.warpPerspective(frame, M, output_size)

def get_board_corners(frame: np.ndarray, cached_corners: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
	"""
	Get board corners using:
	1. Cached corners (if provided)
	2. Automatic detection (CameraChessWeb-inspired)
	3. Manual selection (fallback)
	
	Returns (corners, debug_img)
	"""
	# Use cached corners if available
	if cached_corners is not None:
		return cached_corners, None
	
	# Try automatic detection
	corners, debug_img = find_board_corners(frame, debug=True)
	
	if corners is not None:
		print("Automatic board detection successful.")
		return corners, debug_img
	
	# Show debug image if available
	if debug_img is not None:
		cv2.imshow("Auto-detection Debug", debug_img)
		cv2.waitKey(1000)
		cv2.destroyWindow("Auto-detection Debug")
	
	# Fall back to manual selection
	print("Automatic board detection failed. Please select board corners manually.")
	print("Click corners in order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
	manual_corners = pick_corners_interactive(frame)
	return manual_corners, None

def main():
	reader = VideoReader(
		path="data/videos/game_1.mp4",
		sample_interval_frames=30,
		max_frames=10
	)

	frames = reader.read()
	print(f"Read {len(frames)} frames")

	cached_corners: Optional[np.ndarray] = None

	for i, frame in enumerate(frames):
		cropped_frame = crop_top_half(frame)
		
		corners, debug_img = get_board_corners(cropped_frame, cached_corners)
		
		if corners is None:
			print(f"Frame {i}: Corner selection cancelled. Exiting.")
			return
		
		if cached_corners is None:
			cached_corners = corners
			print(f"Corners cached for subsequent frames:\n{corners}")
		
		board_img = warp_board(cropped_frame, corners)
		
		print(f"Frame {i}: Warped board shape = {board_img.shape}")
		
		if i == 0:
			# Show debug and warped for first frame
			if debug_img is not None:
				cv2.imshow("Board Detection Debug", debug_img)
				cv2.waitKey(0)
			
			cv2.imshow("Warped Board", board_img)
			cv2.waitKey(0)
		else:
			cv2.imshow("Warped Board", board_img)
			key = cv2.waitKey(500)
			if key == ord('q'):
				break

	cv2.destroyAllWindows()
	print("Processing complete.")

if __name__ == "__main__":
	main()
