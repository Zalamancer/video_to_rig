"""Convert MediaPipe 3D landmarks to BVH animation format."""
import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Skeleton definition
# ---------------------------------------------------------------------------
# Each entry: (name, parent, rest_offset, primary_child, landmark_index)
#   rest_offset: T-pose offset from parent in centimetres (Y-up)
#   primary_child: which child determines this joint's rotation direction
#   landmark_index: MediaPipe landmark index (or None for computed joints)

SKELETON = [
    ("Hips",         None,     (0, 0, 0),      "Spine",        None),
    ("Spine",        "Hips",   (0, 12, 0),     "Spine1",       None),
    ("Spine1",       "Spine",  (0, 12, 0),     "Neck",         None),
    ("Neck",         "Spine1", (0, 6, 0),      "Head",         None),
    ("Head",         "Neck",   (0, 8, 0),      None,           0),

    ("LeftArm",      "Spine1", (15, 0, 0),     "LeftForeArm",  11),
    ("LeftForeArm",  "LeftArm",(25, 0, 0),     "LeftHand",     13),
    ("LeftHand",     "LeftForeArm", (22, 0, 0), None,          15),

    ("RightArm",     "Spine1", (-15, 0, 0),    "RightForeArm", 12),
    ("RightForeArm", "RightArm",(-25, 0, 0),   "RightHand",    14),
    ("RightHand",    "RightForeArm",(-22, 0, 0), None,         16),

    ("LeftUpLeg",    "Hips",   (10, 0, 0),     "LeftLeg",      23),
    ("LeftLeg",      "LeftUpLeg",(0, -43, 0),  "LeftFoot",     25),
    ("LeftFoot",     "LeftLeg",(0, -40, 0),    None,           27),

    ("RightUpLeg",   "Hips",   (-10, 0, 0),    "RightLeg",     24),
    ("RightLeg",     "RightUpLeg",(0, -43, 0), "RightFoot",    26),
    ("RightFoot",    "RightLeg",(0, -40, 0),   None,           28),
]

END_SITES = {
    "Head":      (0, 10, 0),
    "LeftHand":  (10, 0, 0),
    "RightHand": (-10, 0, 0),
    "LeftFoot":  (0, -5, 5),
    "RightFoot": (0, -5, 5),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.array([0.0, 1.0, 0.0])


def _rot_between(a, b):
    """Rotation that maps unit vector *a* onto unit vector *b*."""
    a, b = _norm(a), _norm(b)
    dot = float(np.dot(a, b))
    if dot > 0.99999:
        return Rotation.identity()
    if dot < -0.99999:
        perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        axis = _norm(np.cross(a, perp))
        return Rotation.from_rotvec(np.pi * axis)
    cross = np.cross(a, b)
    s = np.linalg.norm(cross)
    angle = np.arctan2(s, dot)
    return Rotation.from_rotvec(angle * cross / s)


def _mp_to_bvh(pos, scale):
    """MediaPipe world coords -> BVH Y-up coords, scaled to cm."""
    # MediaPipe: x right, y DOWN, z toward camera
    # BVH:      x right, y UP,   z forward (away from camera)
    return np.array([pos[0], -pos[1], -pos[2]]) * scale

# ---------------------------------------------------------------------------
# Joint target positions from landmarks
# ---------------------------------------------------------------------------

def _target_positions(frame_lm, scale):
    """Return dict  joint_name -> world position (cm, Y-up)."""
    p = {}

    lhip = _mp_to_bvh(frame_lm[23], scale)
    rhip = _mp_to_bvh(frame_lm[24], scale)
    lsho = _mp_to_bvh(frame_lm[11], scale)
    rsho = _mp_to_bvh(frame_lm[12], scale)

    hips  = (lhip + rhip) / 2
    chest = (lsho + rsho) / 2

    p["Hips"]   = hips
    p["Spine"]  = hips * 0.65 + chest * 0.35
    p["Spine1"] = hips * 0.30 + chest * 0.70
    p["Neck"]   = chest + (chest - hips) * 0.15
    p["Head"]   = _mp_to_bvh(frame_lm[0], scale)

    p["LeftArm"]      = _mp_to_bvh(frame_lm[11], scale)
    p["LeftForeArm"]  = _mp_to_bvh(frame_lm[13], scale)
    p["LeftHand"]     = _mp_to_bvh(frame_lm[15], scale)

    p["RightArm"]     = _mp_to_bvh(frame_lm[12], scale)
    p["RightForeArm"] = _mp_to_bvh(frame_lm[14], scale)
    p["RightHand"]    = _mp_to_bvh(frame_lm[16], scale)

    p["LeftUpLeg"]  = _mp_to_bvh(frame_lm[23], scale)
    p["LeftLeg"]    = _mp_to_bvh(frame_lm[25], scale)
    p["LeftFoot"]   = _mp_to_bvh(frame_lm[27], scale)

    p["RightUpLeg"] = _mp_to_bvh(frame_lm[24], scale)
    p["RightLeg"]   = _mp_to_bvh(frame_lm[26], scale)
    p["RightFoot"]  = _mp_to_bvh(frame_lm[28], scale)

    return p

# ---------------------------------------------------------------------------
# Root (Hips) orientation
# ---------------------------------------------------------------------------

def _root_rotation(frame_lm, scale):
    lhip = _mp_to_bvh(frame_lm[23], scale)
    rhip = _mp_to_bvh(frame_lm[24], scale)
    lsho = _mp_to_bvh(frame_lm[11], scale)
    rsho = _mp_to_bvh(frame_lm[12], scale)

    hip_c  = (lhip + rhip) / 2
    chest  = (lsho + rsho) / 2

    y = _norm(chest - hip_c)                       # up
    x_raw = _norm(lhip - rhip)                     # side-to-side
    x = _norm(x_raw - np.dot(x_raw, y) * y)        # orthogonalise
    z = np.cross(x, y)                              # forward

    mat = np.column_stack([x, y, z])
    det = np.linalg.det(mat)
    if abs(det) < 0.01:
        return Rotation.identity()
    if det < 0:
        mat[:, 2] *= -1  # fix handedness
    return Rotation.from_matrix(mat)

# ---------------------------------------------------------------------------
# Per-frame rotation solve
# ---------------------------------------------------------------------------

_child_offset_cache = {}

def _child_offset(child_name):
    if child_name not in _child_offset_cache:
        for name, _, off, _, _ in SKELETON:
            if name == child_name:
                _child_offset_cache[child_name] = np.array(off, dtype=float)
                break
    return _child_offset_cache.get(child_name)


def _solve_frame(frame_lm, scale):
    """Return (root_position, {joint_name: euler_ZXY_degrees})."""
    tgt = _target_positions(frame_lm, scale)
    root_rot = _root_rotation(frame_lm, scale)

    w_pos = {}   # world position (FK)
    w_rot = {}   # world rotation
    euler = {}   # local Euler angles ZXY (degrees)

    # Root
    w_pos["Hips"] = tgt["Hips"]
    w_rot["Hips"] = root_rot
    euler["Hips"] = root_rot.as_euler("ZXY", degrees=True)

    for name, parent, offset_tup, prim_child, _ in SKELETON[1:]:
        offset = np.array(offset_tup, dtype=float)
        p_rot = w_rot[parent]
        p_pos = w_pos[parent]

        # FK position of this joint
        j_pos = p_pos + p_rot.apply(offset)
        w_pos[name] = j_pos

        if prim_child and prim_child in tgt:
            desired = tgt[prim_child] - j_pos
            if np.linalg.norm(desired) < 1e-6:
                loc = Rotation.identity()
            else:
                c_off = _child_offset(prim_child)
                if c_off is not None and np.linalg.norm(c_off) > 0.01:
                    local_desired = p_rot.inv().apply(desired)
                    loc = _rot_between(c_off, local_desired)
                else:
                    loc = Rotation.identity()
        else:
            loc = Rotation.identity()

        euler[name] = loc.as_euler("ZXY", degrees=True)
        w_rot[name] = p_rot * loc

    return tgt["Hips"], euler

# ---------------------------------------------------------------------------
# BVH writer
# ---------------------------------------------------------------------------

def _write_hierarchy(f):
    """Write HIERARCHY section; return joint_order list."""
    joint_order = []

    def _write(name, offset, is_root, depth):
        ind = "  " * depth
        tag = "ROOT" if is_root else "JOINT"
        f.write(f"{ind}{tag} {name}\n{ind}{{\n")
        f.write(f"{ind}  OFFSET {offset[0]:.4f} {offset[1]:.4f} {offset[2]:.4f}\n")
        if is_root:
            f.write(f"{ind}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
        else:
            f.write(f"{ind}  CHANNELS 3 Zrotation Xrotation Yrotation\n")
        joint_order.append(name)

        # children
        for cn, cp, co, _, _ in SKELETON:
            if cp == name:
                _write(cn, co, False, depth + 1)

        # end site for leaves
        if name in END_SITES:
            es = END_SITES[name]
            f.write(f"{ind}  End Site\n{ind}  {{\n")
            f.write(f"{ind}    OFFSET {es[0]:.4f} {es[1]:.4f} {es[2]:.4f}\n")
            f.write(f"{ind}  }}\n")

        f.write(f"{ind}}}\n")

    f.write("HIERARCHY\n")
    root = SKELETON[0]
    _write(root[0], root[2], True, 0)
    return joint_order

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def landmarks_to_bvh(all_landmarks, output_path, fps=30, scale=100):
    """
    Convert MediaPipe world landmarks to a BVH file.

    Args:
        all_landmarks: np.array (num_frames, 33, 3) in metres
        output_path:   where to write the .bvh
        fps:           frames per second
        scale:         metres -> output units (100 = centimetres)
    """
    num_frames = len(all_landmarks)
    root_positions = []
    frame_eulers = []

    for i in range(num_frames):
        rp, eu = _solve_frame(all_landmarks[i], scale)
        root_positions.append(rp)
        frame_eulers.append(eu)

    # Vertical offset so feet touch y=0
    all_tgt = _target_positions(all_landmarks[0], scale)
    min_y = min(all_tgt[k][1] for k in ("LeftFoot", "RightFoot"))
    y_off = -min_y
    for rp in root_positions:
        rp[1] += y_off

    with open(output_path, "w") as f:
        joint_order = _write_hierarchy(f)

        f.write("MOTION\n")
        f.write(f"Frames: {num_frames}\n")
        f.write(f"Frame Time: {1.0 / fps:.6f}\n")

        for i in range(num_frames):
            vals = []
            for jn in joint_order:
                if jn == "Hips":
                    p = root_positions[i]
                    r = frame_eulers[i][jn]
                    vals.extend([p[0], p[1], p[2], r[0], r[1], r[2]])
                else:
                    r = frame_eulers[i][jn]
                    vals.extend([r[0], r[1], r[2]])
            f.write(" ".join(f"{v:.4f}" for v in vals) + "\n")
