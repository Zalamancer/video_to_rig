"""Extract 3D pose landmarks from video using MediaPipe Tasks API."""
import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "pose_landmarker_heavy.task")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def extract_poses(video_path, target_fps=30, start_time=0, duration=0):
    """
    Extract 3D + 2D pose landmarks from a video file.

    Returns:
        landmarks_3d: np.array (num_frames, 33, 3) — world landmarks in metres
        landmarks_2d: np.array (num_frames, 33, 2) — normalised image coords
        image_dims:   (width, height) of video frames
        actual_fps:   effective FPS after sampling
        frame_count:  number of extracted frames
    """
    if not os.path.isfile(_MODEL_PATH):
        raise FileNotFoundError(
            f"Pose model not found at {_MODEL_PATH}\n"
            "Download with:\n"
            '  curl -L -o pose_landmarker_heavy.task '
            '"https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
            'pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"'
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_interval = max(1, round(source_fps / target_fps))
    actual_fps = source_fps / frame_interval

    start_frame = int(start_time * source_fps)
    end_frame = total_frames
    if duration > 0:
        end_frame = min(total_frames, start_frame + int(duration * source_fps))

    expected_frames = max(1, (end_frame - start_frame) // frame_interval)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_3d = []
    all_2d = []
    missed = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        pbar = tqdm(total=expected_frames, desc="Extracting poses", unit="frame")
        frame_idx = start_frame

        while cap.isOpened() and frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % frame_interval != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / source_fps)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if (result.pose_world_landmarks and len(result.pose_world_landmarks) > 0
                    and result.pose_landmarks and len(result.pose_landmarks) > 0):
                wl = result.pose_world_landmarks[0]
                nl = result.pose_landmarks[0]
                frame_3d = np.array([[lm.x, lm.y, lm.z] for lm in wl])
                frame_2d = np.array([[lm.x, lm.y] for lm in nl])
                all_3d.append(frame_3d)
                all_2d.append(frame_2d)
            elif all_3d:
                all_3d.append(all_3d[-1].copy())
                all_2d.append(all_2d[-1].copy())
                missed += 1
            else:
                missed += 1

            frame_idx += 1
            pbar.update(1)

        pbar.close()

    cap.release()

    if missed > 0:
        total = missed + len(all_3d)
        pct = missed / max(1, total) * 100
        print(f"  Warning: pose detection missed {missed} frames ({pct:.0f}%)")

    if not all_3d:
        return np.array([]), np.array([]), (img_w, img_h), actual_fps, 0

    return (np.array(all_3d), np.array(all_2d),
            (img_w, img_h), actual_fps, len(all_3d))
