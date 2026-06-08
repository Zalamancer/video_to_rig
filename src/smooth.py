"""Smooth landmark trajectories to reduce MediaPipe jitter."""
import numpy as np
from scipy.signal import savgol_filter


def smooth_landmarks(landmarks, factor=0.5):
    """
    Apply Savitzky-Golay smoothing to landmark trajectories.

    Args:
        landmarks: np.array of shape (num_frames, 33, 3)
        factor: 0-1, higher = smoother (0 = no smoothing)

    Returns:
        Smoothed landmarks array of same shape.
    """
    if factor <= 0 or len(landmarks) < 5:
        return landmarks

    # Window size scales with smoothing factor (must be odd, >= 3)
    window = max(3, int(factor * 15) | 1)
    if window >= len(landmarks):
        window = len(landmarks) if len(landmarks) % 2 == 1 else len(landmarks) - 1
        window = max(3, window)

    poly_order = min(2, window - 1)

    smoothed = landmarks.copy()
    num_frames, num_landmarks, num_dims = landmarks.shape

    for i in range(num_landmarks):
        for d in range(num_dims):
            smoothed[:, i, d] = savgol_filter(
                landmarks[:, i, d], window, poly_order
            )

    return smoothed
