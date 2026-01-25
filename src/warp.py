import numpy as np
from typing import List, Tuple
from constants import BOARD_SIZE, SQUARE_SIZE


def perspective_transform(src: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply perspective transform to source points.
    
    Args:
        src: Nx2 or Nx3 array of points
        transform: 3x3 transformation matrix
    
    Returns:
        Nx2 array of transformed points
    """
    if src.shape[1] == 2:
        ones = np.ones((src.shape[0], 1))
        src = np.hstack([src, ones])
    
    warped = src @ transform.T
    
    # Normalize by w coordinate
    warped[:, 0] /= warped[:, 2]
    warped[:, 1] /= warped[:, 2]
    
    return warped[:, :2]


def get_perspective_transform(target: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """
    Compute perspective transform matrix from keypoints to target.
    
    Args:
        target: 4x2 array of target points
        keypoints: 4x2 array of source keypoints
    
    Returns:
        3x3 transformation matrix
    """
    A = np.zeros((8, 8), dtype=np.float64)
    B = np.zeros((8, 1), dtype=np.float64)
    
    for i in range(4):
        x, y = keypoints[i]
        u, v = target[i]
        
        A[i * 2, 0] = x
        A[i * 2, 1] = y
        A[i * 2, 2] = 1
        A[i * 2, 6] = -u * x
        A[i * 2, 7] = -u * y
        
        A[i * 2 + 1, 3] = x
        A[i * 2 + 1, 4] = y
        A[i * 2 + 1, 5] = 1
        A[i * 2 + 1, 6] = -v * x
        A[i * 2 + 1, 7] = -v * y
        
        B[i * 2, 0] = u
        B[i * 2 + 1, 0] = v
    
    solution = np.linalg.solve(A, B).flatten()
    transform = np.array([*solution, 1.0]).reshape(3, 3)
    
    return transform


def get_inv_transform(keypoints: np.ndarray) -> np.ndarray:
    """
    Get inverse perspective transform from board corners to image coordinates.
    
    Args:
        keypoints: 4x2 array of corner points (ordered: BR, BL, TL, TR)
    
    Returns:
        3x3 inverse transformation matrix
    """
    target = np.array([
        [BOARD_SIZE, BOARD_SIZE],
        [0, BOARD_SIZE],
        [0, 0],
        [BOARD_SIZE, 0]
    ], dtype=np.float64)
    
    transform = get_perspective_transform(target, keypoints)
    inv_transform = np.linalg.inv(transform)
    
    return inv_transform


def transform_centers(inv_transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform square center points from board space to image space.
    
    Args:
        inv_transform: 3x3 inverse transformation matrix
    
    Returns:
        Tuple of (centers_2d, centers_3d) where centers_3d has batch dimension
    """
    x = np.array([0.5 + i for i in range(8)])
    y = np.array([7.5 - i for i in range(8)])
    
    warped_centers = []
    for yy in y:
        for xx in x:
            warped_centers.append([xx * SQUARE_SIZE, yy * SQUARE_SIZE, 1])
    
    warped_centers = np.array(warped_centers, dtype=np.float64)
    centers = perspective_transform(warped_centers, inv_transform)
    centers_3d = np.expand_dims(centers, axis=0)
    
    return centers, centers_3d


def transform_boundary(inv_transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform board boundary points from board space to image space.
    
    Args:
        inv_transform: 3x3 inverse transformation matrix
    
    Returns:
        Tuple of (boundary_2d, boundary_3d) where boundary_3d has batch dimension
    """
    warped_boundary = np.array([
        [-0.5 * SQUARE_SIZE, -0.5 * SQUARE_SIZE, 1],
        [-0.5 * SQUARE_SIZE, 8.5 * SQUARE_SIZE, 1],
        [8.5 * SQUARE_SIZE, 8.5 * SQUARE_SIZE, 1],
        [8.5 * SQUARE_SIZE, -0.5 * SQUARE_SIZE, 1]
    ], dtype=np.float64)
    
    boundary = perspective_transform(warped_boundary, inv_transform)
    boundary_3d = np.expand_dims(boundary, axis=0)
    
    return boundary, boundary_3d
