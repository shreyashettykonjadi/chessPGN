"""
Automatic board corner detection inspired by CameraChessWeb.
Uses OpenCV corner detection + Delaunay triangulation to find the chessboard.
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple

# 7x7 grid of internal corners for an 8x8 chessboard
GRID = np.array([[x, y] for y in range(7) for x in range(7)], dtype=np.float32)
IDEAL_QUAD = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)


def detect_xcorners(frame: np.ndarray, max_corners: int = 100) -> np.ndarray:
    """
    Detect potential grid intersection points (X-corners) in the frame.
    Uses Shi-Tomasi corner detection followed by subpixel refinement.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Detect corners using Shi-Tomasi
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=15,
        blockSize=5,
        useHarrisDetector=False
    )
    
    if corners is None or len(corners) < 5:
        return np.array([])
    
    # Subpixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    
    return corners.reshape(-1, 2)


def get_perspective_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute 3x3 perspective transform matrix from src to dst points."""
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))


def perspective_transform(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply perspective transform M to points."""
    if len(points) == 0:
        return np.array([])
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    transformed = cv2.perspectiveTransform(pts, M)
    return transformed.reshape(-1, 2)


def cdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance between two sets of points."""
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def calculate_offset_score(warped_xcorners: np.ndarray, shift: Tuple[int, int]) -> float:
    """Calculate how well warped corners align with shifted grid."""
    grid = GRID + np.array(shift)
    dist = cdist(grid, warped_xcorners)
    assignment_cost = np.sum(np.min(dist, axis=1))
    score = 1.0 / (1.0 + assignment_cost)
    return score


def find_offset(warped_xcorners: np.ndarray) -> Tuple[int, int]:
    """Find the best grid offset for warped corners using binary search."""
    best_offset = [0, 0]
    
    for i in range(2):
        low, high = -7, 1
        scores = {}
        
        while (high - low) > 1:
            mid = (high + low) // 2
            for x in [mid, mid + 1]:
                if x not in scores:
                    shift = [0, 0]
                    shift[i] = x
                    scores[x] = calculate_offset_score(warped_xcorners, tuple(shift))
            
            if scores[mid] > scores[mid + 1]:
                high = mid
            else:
                low = mid
        
        best_offset[i] = low + 1
    
    return tuple(best_offset)


def _build_delaunay_indices(points: np.ndarray, tolerance: float = 5.0) -> List[Tuple[int, int, int]]:
    if len(points) < 3:
        return []
    margin = 10.0
    min_xy = np.min(points, axis=0) - margin
    max_xy = np.max(points, axis=0) + margin
    rect = (int(min_xy[0]), int(min_xy[1]), int(max_xy[0]), int(max_xy[1]))
    subdiv = cv2.Subdiv2D(rect)
    for (x, y) in points:
        subdiv.insert((float(x), float(y)))
    triangles = subdiv.getTriangleList()
    indices: List[Tuple[int, int, int]] = []
    seen: set[Tuple[int, int, int]] = set()
    for tri in triangles:
        tri_pts = np.array([(tri[0], tri[1]), (tri[2], tri[3]), (tri[4], tri[5])], dtype=np.float32)
        idx: List[int] = []
        for pt in tri_pts:
            dists = np.linalg.norm(points - pt, axis=1)
            nearest = int(np.argmin(dists))
            if dists[nearest] > tolerance:
                idx = []
                break
            idx.append(nearest)
        if len(idx) == 3 and len(set(idx)) == 3:
            key = tuple(sorted(idx))
            if key not in seen:
                seen.add(key)
                indices.append(tuple(idx))
    return indices


def get_quads_from_xcorners(xcorners: np.ndarray) -> List[np.ndarray]:
    """
    Use Delaunay triangulation to find quadrilateral candidates from X-corners.
    Each quad is formed by two adjacent triangles sharing an edge.
    """
    if len(xcorners) < 4:
        return []
    
    triangle_indices = _build_delaunay_indices(xcorners)
    if not triangle_indices:
        return []
    
    quads: List[np.ndarray] = []
    for i, tri in enumerate(triangle_indices):
        t1, t2, t3 = tri
        quad = [t1, t2, t3, -1]
        for j, other in enumerate(triangle_indices):
            if i == j:
                continue
            shared = set(tri) & set(other)
            if len(shared) == 2:
                fourth_candidates = set(other) - shared
                if fourth_candidates:
                    quad[3] = fourth_candidates.pop()
                    break
        if quad[3] == -1:
            continue
        quad_pts = xcorners[np.array(quad)]
        center = quad_pts.mean(axis=0)
        angles = np.arctan2(quad_pts[:, 1] - center[1], quad_pts[:, 0] - center[0])
        order = np.argsort(angles)
        quads.append(quad_pts[order])
    return quads


def score_quad(quad: np.ndarray, xcorners: np.ndarray) -> Tuple[float, np.ndarray, Tuple[int, int]]:
    """
    Score how well a quad represents the chessboard by:
    1. Computing perspective transform from ideal quad to candidate
    2. Warping all X-corners
    3. Finding best grid alignment offset
    4. Scoring based on alignment quality
    """
    M = get_perspective_transform(IDEAL_QUAD, quad)
    M_inv = np.linalg.inv(M)
    warped_xcorners = perspective_transform(xcorners, M_inv)
    offset = find_offset(warped_xcorners)
    score = calculate_offset_score(warped_xcorners, offset)
    return score, M, offset


def find_corners_from_xcorners(xcorners: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the 4 board corners from detected X-corners.
    Uses Delaunay triangulation to form quad candidates and scores each.
    """
    quads = get_quads_from_xcorners(xcorners)
    
    if len(quads) == 0:
        return None
    
    best_score = -1
    best_M = None
    best_offset = None
    
    for quad in quads:
        try:
            score, M, offset = score_quad(quad, xcorners)
            if score > best_score:
                best_score = score
                best_M = M
                best_offset = offset
        except:
            continue
    
    if best_M is None or best_offset is None:
        return None
    
    # Compute board corners from best transform and offset
    try:
        M_inv = np.linalg.inv(best_M)
    except:
        return None
    
    # Warped corner positions (outer corners of 8x8 board)
    warped_corners = np.array([
        [best_offset[0] - 1, best_offset[1] - 1],
        [best_offset[0] - 1, best_offset[1] + 7],
        [best_offset[0] + 7, best_offset[1] + 7],
        [best_offset[0] + 7, best_offset[1] - 1]
    ], dtype=np.float32)
    
    corners = perspective_transform(warped_corners, M_inv)
    return corners


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Order corners as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]  # top-left
    rect[2] = corners[np.argmax(s)]  # bottom-right
    diff = np.diff(corners, axis=1).flatten()
    rect[1] = corners[np.argmin(diff)]  # top-right
    rect[3] = corners[np.argmax(diff)]  # bottom-left
    return rect


def find_board_corners(frame: np.ndarray, debug: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Main entry point: detect chessboard corners automatically.
    
    Args:
        frame: BGR image
        debug: If True, return debug visualization
    
    Returns:
        (corners, debug_img) - corners is 4x2 float32 (TL, TR, BR, BL) or None
    """
    debug_img = frame.copy() if debug else None
    
    # Detect X-corners (grid intersection points)
    xcorners = detect_xcorners(frame)
    
    if debug_img is not None and len(xcorners) > 0:
        for pt in xcorners:
            cv2.circle(debug_img, tuple(pt.astype(int)), 3, (0, 255, 0), -1)
    
    if len(xcorners) < 5:
        if debug_img is not None:
            cv2.putText(debug_img, f"Need >= 5 X-corners, found {len(xcorners)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, debug_img
    
    # Find corners from X-corners
    corners = find_corners_from_xcorners(xcorners)
    
    if corners is None:
        if debug_img is not None:
            cv2.putText(debug_img, "Failed to find board corners", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, debug_img
    
    # Order corners consistently
    corners = order_corners(corners)
    
    # Clip to image bounds
    h, w = frame.shape[:2]
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
    
    # Validate corners
    area = cv2.contourArea(corners)
    if area < 0.01 * w * h:
        if debug_img is not None:
            cv2.putText(debug_img, "Detected area too small", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, debug_img
    
    if debug_img is not None:
        pts = corners.reshape(-1, 1, 2).astype(int)
        cv2.polylines(debug_img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        for i, (x, y) in enumerate(corners.astype(int)):
            cv2.circle(debug_img, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(debug_img, str(i + 1), (x + 8, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(debug_img, f"Found {len(xcorners)} X-corners", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return corners, debug_img
