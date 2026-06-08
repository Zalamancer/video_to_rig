"""
Recover global body orientation using Perspective-n-Point (PnP).

MediaPipe world landmarks are body-relative (centred at hips), losing global
rotation. By matching 2D image landmarks to the 3D body model with solvePnP,
we recover the camera-to-body rotation — i.e. the global body orientation.

This rotation is then applied to the body-relative 3D positions, giving us
properly oriented world-space coordinates where the skeleton turns and faces
in the direction the person actually faces in the video.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# Key torso landmarks for stable PnP (arms/legs move too much)
_PNP_INDICES = [0, 11, 12, 23, 24, 25, 26, 27, 28]  # head, shoulders, hips, knees, ankles


def estimate_global_rotations(landmarks_3d, landmarks_2d, image_dims):
    """
    For each frame, estimate the global body rotation via PnP.

    Args:
        landmarks_3d: (N, 33, 3) body-relative world coords in metres
        landmarks_2d: (N, 33, 2) normalised image coords [0..1]
        image_dims:   (width, height) in pixels

    Returns:
        rotations: list of N scipy Rotation objects (global body orientation)
    """
    w, h = image_dims
    # Approximate camera intrinsics (focal length ≈ image width for typical video)
    fx = fy = float(w)
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros(4)

    rotations = []
    prev_rvec = None

    for fi in range(len(landmarks_3d)):
        pts_3d = landmarks_3d[fi][_PNP_INDICES].astype(np.float64)
        pts_2d_norm = landmarks_2d[fi][_PNP_INDICES].astype(np.float64)

        # Convert normalised coords → pixels
        pts_2d = pts_2d_norm.copy()
        pts_2d[:, 0] *= w
        pts_2d[:, 1] *= h

        try:
            if prev_rvec is not None:
                ok, rvec, tvec = cv2.solvePnP(
                    pts_3d, pts_2d, camera_matrix, dist_coeffs,
                    rvec=prev_rvec.copy(), tvec=np.zeros((3, 1)),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            else:
                ok, rvec, tvec = cv2.solvePnP(
                    pts_3d, pts_2d, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_SQPNP,
                )

            if ok:
                prev_rvec = rvec
                rot = Rotation.from_rotvec(rvec.flatten())
                rotations.append(rot)
            else:
                rotations.append(rotations[-1] if rotations else Rotation.identity())
        except Exception:
            rotations.append(rotations[-1] if rotations else Rotation.identity())

    return rotations


def apply_global_rotations(landmarks_3d, rotations):
    """
    Rotate body-relative landmarks into global orientation.

    Args:
        landmarks_3d: (N, 33, 3) body-relative world coords
        rotations:    list of N Rotation objects

    Returns:
        (N, 33, 3) globally oriented landmarks
    """
    oriented = np.empty_like(landmarks_3d)

    for fi in range(len(landmarks_3d)):
        rot = rotations[fi]
        # Rotate all landmarks by the global orientation
        oriented[fi] = rot.apply(landmarks_3d[fi])

    return oriented


def smooth_rotations(rotations, window=5):
    """Smooth rotation sequence using SLERP-like averaging."""
    if len(rotations) < window:
        return rotations

    # Convert to rotation vectors for averaging
    rvecs = np.array([r.as_rotvec() for r in rotations])

    smoothed = rvecs.copy()
    half = window // 2
    for i in range(len(rvecs)):
        lo = max(0, i - half)
        hi = min(len(rvecs), i + half + 1)
        smoothed[i] = rvecs[lo:hi].mean(axis=0)

    return [Rotation.from_rotvec(rv) for rv in smoothed]
