"""Export 3D landmark positions as a JSON animation file.

This preserves the full 3D data without lossy rotation conversion.
The viewer can animate joint positions directly and use IK for retargeting.
"""
import json
import numpy as np


# MediaPipe landmark indices → joint name
JOINT_MAP = {
    "Hips":         (23, 24),    # midpoint
    "Spine":        None,         # interpolated
    "Chest":        (11, 12),    # midpoint
    "Neck":         None,         # interpolated
    "Head":         (0,),

    "LeftShoulder":  (11,),
    "LeftElbow":     (13,),
    "LeftWrist":     (15,),

    "RightShoulder": (12,),
    "RightElbow":    (14,),
    "RightWrist":    (16,),

    "LeftHip":       (23,),
    "LeftKnee":      (25,),
    "LeftAnkle":     (27,),

    "RightHip":      (24,),
    "RightKnee":     (26,),
    "RightAnkle":    (28,),
}

# Bones: pairs of joints to draw lines between
BONES = [
    ("Hips", "Spine"), ("Spine", "Chest"), ("Chest", "Neck"), ("Neck", "Head"),
    ("Chest", "LeftShoulder"), ("LeftShoulder", "LeftElbow"), ("LeftElbow", "LeftWrist"),
    ("Chest", "RightShoulder"), ("RightShoulder", "RightElbow"), ("RightElbow", "RightWrist"),
    ("Hips", "LeftHip"), ("LeftHip", "LeftKnee"), ("LeftKnee", "LeftAnkle"),
    ("Hips", "RightHip"), ("RightHip", "RightKnee"), ("RightKnee", "RightAnkle"),
]


def _mp_to_bvh(pos, scale):
    """MediaPipe → Y-up, cm."""
    return [float(pos[0] * scale), float(-pos[1] * scale), float(-pos[2] * scale)]


def _get_pos(frame_lm, indices, scale):
    if len(indices) == 1:
        return _mp_to_bvh(frame_lm[indices[0]], scale)
    elif len(indices) == 2:
        mid = (frame_lm[indices[0]] + frame_lm[indices[1]]) / 2
        return _mp_to_bvh(mid, scale)


def landmarks_to_json(all_landmarks, output_path, fps=30, scale=100):
    """Export landmark positions as JSON for direct-position animation."""
    joint_names = list(JOINT_MAP.keys())
    num_frames = len(all_landmarks)

    # Compute all frames
    frames = []
    for fi in range(num_frames):
        lm = all_landmarks[fi]
        positions = {}

        for name, indices in JOINT_MAP.items():
            if indices is not None:
                positions[name] = _get_pos(lm, indices, scale)

        # Interpolated joints
        hips = np.array(positions["Hips"])
        chest = np.array(positions["Chest"])
        positions["Spine"] = ((hips + chest) / 2).tolist()
        positions["Neck"] = (chest + (chest - hips) * 0.15).tolist()

        frames.append(positions)

    # Vertical offset so feet touch y=0
    min_y = min(
        min(frames[0]["LeftAnkle"][1], frames[0]["RightAnkle"][1]),
        min(frames[0]["LeftKnee"][1], frames[0]["RightKnee"][1]),
    )
    for frame in frames:
        for name in frame:
            frame[name][1] -= min_y

    data = {
        "fps": fps,
        "joints": joint_names,
        "bones": BONES,
        "numFrames": num_frames,
        "frames": frames,
    }

    with open(output_path, "w") as f:
        json.dump(data, f)

    return output_path
