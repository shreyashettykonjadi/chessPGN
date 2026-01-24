import cv2
import numpy as np
from typing import List, Tuple, Optional


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _validate_points(points: List[Tuple[int, int]], img_shape: Tuple[int, int]) -> Tuple[bool, str]:
    """
    Validate 4 corner points.
    Returns (is_valid, error_message).
    """
    if len(points) != 4:
        return False, "Must select exactly 4 corners."
    
    # Check uniqueness (min distance > 10 pixels)
    pts_arr = np.array(points, dtype="float32")
    for i in range(4):
        for j in range(i + 1, 4):
            dist = np.linalg.norm(pts_arr[i] - pts_arr[j])
            if dist < 10.0:
                return False, f"Points {i+1} and {j+1} are too close. Please reselect."
    
    # Check polygon area (must be > 1% of image area)
    img_area = float(img_shape[0] * img_shape[1])
    poly_area = cv2.contourArea(pts_arr)
    if poly_area < 0.01 * img_area:
        return False, "Selected area is too small. Please reselect."
    
    return True, ""


def pick_corners_interactive(img: np.ndarray, window_name: str = "Select Board Corners") -> Optional[np.ndarray]:
    """
    Display image and guide user to click 4 corners.
    After 4 clicks, validate and auto-order points.
    
    Keys: 'r' to reset, ESC or 'q' to cancel.
    
    Args:
        img: Input image (BGR NumPy array)
        window_name: Name of the OpenCV window
    
    Returns:
        4x2 float32 array of ordered corners (TL, TR, BR, BL) or None on cancel.
    """
    corner_labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    points: List[Tuple[int, int]] = []
    validation_msg = ""
    
    def redraw():
        nonlocal validation_msg
        disp = img.copy()
        
        # Draw instruction text
        next_idx = len(points)
        if next_idx < 4:
            instruction = f"Click corner {next_idx + 1}/4"
            cv2.putText(disp, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(disp, "Validating selection...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show validation message if any
        if validation_msg:
            cv2.putText(disp, validation_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw existing points with numbers
        for idx, (x, y) in enumerate(points):
            color = (0, 0, 255) if idx == len(points) - 1 else (255, 0, 0)
            cv2.circle(disp, (x, y), 5, color, -1)
            cv2.putText(disp, str(idx + 1), (x + 8, y - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw lines between consecutive points
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(disp, points[i], points[i + 1], (255, 0, 0), 2)
        # Close the quadrilateral if all 4 points selected
        if len(points) == 4:
            cv2.line(disp, points[3], points[0], (255, 0, 0), 2)
        
        # Draw reset and cancel instructions
        cv2.putText(disp, "Press 'r' to reset | ESC/q to cancel", (10, img.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return disp

    def _mouse(event, x, y, flags, param):
        nonlocal points, validation_msg
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                validation_msg = ""
                cv2.imshow(window_name, redraw())

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, redraw())
    cv2.setMouseCallback(window_name, _mouse)

    while True:
        key = cv2.waitKey(50) & 0xFF
        
        # Validate when 4 points collected
        if len(points) == 4:
            is_valid, error_msg = _validate_points(points, img.shape[:2])
            if is_valid:
                # validation passed, wait for confirmation
                validation_msg = "Selection valid. Press any key to confirm."
                cv2.imshow(window_name, redraw())
                key = cv2.waitKey(0) & 0xFF
                if key not in [ord('r'), ord('q'), 27]:
                    break
                elif key == ord('r'):
                    points = []
                    validation_msg = ""
                    cv2.imshow(window_name, redraw())
                elif key == ord('q') or key == 27:
                    points = []
                    break
            else:
                # validation failed, show message and reset
                validation_msg = error_msg
                cv2.imshow(window_name, redraw())
                cv2.waitKey(2000)  # show error for 2 seconds
                points = []
                validation_msg = ""
                cv2.imshow(window_name, redraw())
        
        if key == ord("r"):
            points = []
            validation_msg = ""
            cv2.imshow(window_name, redraw())
        if key == ord("q") or key == 27:  # ESC
            points = []
            break

    cv2.destroyWindow(window_name)
    if len(points) != 4:
        return None
    
    # Auto-order points using sum/diff method
    pts_arr = np.array(points, dtype="float32")
    ordered = _order_points(pts_arr)
    return ordered
