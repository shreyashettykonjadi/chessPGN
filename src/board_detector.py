import cv2
import numpy as np
from typing import Optional, Tuple, Any


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four points in the order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _is_valid_quad(pts: np.ndarray, min_area_ratio: float = 0.01, frame_area: float = 1.0) -> bool:
    """Check if the quad is valid (unique points, sufficient area)."""
    if pts is None or len(pts) != 4:
        return False
    # Check all points are unique (min distance > 10 pixels)
    for i in range(4):
        for j in range(i + 1, 4):
            dist = np.linalg.norm(pts[i] - pts[j])
            if dist < 10.0:
                return False
    # Check polygon area
    area = cv2.contourArea(pts)
    if area < min_area_ratio * frame_area:
        return False
    return True


class BoardDetector:
    """
    Minimal contour-based board detector with optional manual corner selection fallback.
    detect_board(frame, output_size=(800,800), debug=False, allow_manual=False) -> (warped, ordered_corners, debug_img)
    """

    def __init__(self):
        pass

    def detect_board(
        self,
        frame: Any,
        output_size: Tuple[int, int] = (800, 800),
        debug: bool = False,
        allow_manual: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy ndarray (BGR image)")

        # Visualization frame (ensure BGR)
        vis_frame = frame.copy()
        if vis_frame.ndim == 2:
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR)

        # Preprocessing: grayscale -> blur -> Canny
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        debug_img = vis_frame.copy() if debug else None
        if contours and debug_img is not None:
            cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 1)  # all contours green

        # Attempt automatic detection
        board_quad = None
        if contours:
            img_h, img_w = frame.shape[:2]
            frame_area = float(img_w * img_h)
            area_threshold = 0.03 * frame_area

            # collect centroids for all contours
            centroids = {}
            for idx, cnt in enumerate(contours):
                M = cv2.moments(cnt)
                if M.get("m00", 0) != 0:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])
                    centroids[idx] = (cx, cy)
                else:
                    centroids[idx] = None

            # collect all quadrilateral candidates
            quad_candidates = []
            for idx, cnt in enumerate(contours):
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4:
                    area = cv2.contourArea(approx)
                    quad_candidates.append((idx, area, approx.reshape(4, 2).astype("float32")))

            if quad_candidates:
                # For each candidate, count how many other contour centroids lie inside it
                scores = []
                for orig_idx, area, quad in quad_candidates:
                    count_inside = 0
                    contour_pts = quad.reshape(-1, 2).astype(np.float32)
                    for other_idx, centroid in centroids.items():
                        if centroid is None or other_idx == orig_idx:
                            continue
                        pt = (float(centroid[0]), float(centroid[1]))
                        val = cv2.pointPolygonTest(contour_pts, pt, False)
                        if val > 0:
                            count_inside += 1
                    scores.append((orig_idx, area, quad, count_inside))

                # choose candidate with area > threshold and highest count_inside
                filtered = [s for s in scores if s[1] > area_threshold]
                if filtered:
                    filtered.sort(key=lambda x: (x[3], x[1]), reverse=True)
                    board_quad = filtered[0][2]
                else:
                    # fallback: best by count_inside then area
                    scores.sort(key=lambda x: (x[3], x[1]), reverse=True)
                    board_quad = scores[0][2]

                if debug_img is not None:
                    # draw quad candidates in yellow
                    for orig_idx, area, quad in quad_candidates:
                        pts = quad.reshape(-1, 1, 2).astype(int)
                        cv2.polylines(debug_img, [pts], isClosed=True, color=(0, 255, 255), thickness=1)

        # If automatic detection failed and manual is allowed, trigger manual corner selection
        if board_quad is None and allow_manual:
            print("Automatic board detection failed. Please select board corners manually.")
            print("Click corners in order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
            try:
                from corner_picker import pick_corners_interactive
                manual_corners = pick_corners_interactive(vis_frame)
                if manual_corners is not None:
                    board_quad = manual_corners
            except Exception as e:
                print(f"Manual corner selection failed: {e}")

        if board_quad is None:
            return None, None, debug_img

        rect = _order_points(board_quad)

        # Validate the ordered quad before proceeding
        if not _is_valid_quad(rect, min_area_ratio=0.01, frame_area=frame_area):
            # Ordered quad is degenerate; treat as detection failure
            if allow_manual:
                print("Automatic detection produced invalid quad. Please select corners manually.")
                try:
                    from corner_picker import pick_corners_interactive
                    manual_corners = pick_corners_interactive(vis_frame)
                    if manual_corners is not None:
                        rect = manual_corners
                    else:
                        return None, None, debug_img
                except Exception as e:
                    print(f"Manual corner selection failed: {e}")
                    return None, None, debug_img
            else:
                return None, None, debug_img

        dst = np.array([
            [0.0, 0.0],
            [output_size[0] - 1.0, 0.0],
            [output_size[0] - 1.0, output_size[1] - 1.0],
            [0.0, output_size[1] - 1.0],
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, output_size)

        if debug_img is not None:
            # highlight selected contour in blue (thicker)
            pts_best = rect.reshape(-1, 1, 2).astype(int)
            cv2.polylines(debug_img, [pts_best], isClosed=True, color=(255, 0, 0), thickness=3)
            # draw ordered corner points as red dots
            for (x, y) in rect.astype(int):
                cv2.circle(debug_img, (int(x), int(y)), 6, (0, 0, 255), -1)

        return warped, rect, debug_img
