import math
import numpy as np

# 4 locked Phase 1 equipment classes
EQ_CLASSES = ['barbell', 'dumbbell', 'trap_bar', 'weightplate']
MAX_NORM_DIST = 5.0

def _angle_deg(a, b, c):
    """Return interior angle at vertex b (in degrees, 0-180) normalized to [0, 1]."""
    ba = a - b
    bc = c - b
    n_ba = np.linalg.norm(ba)
    n_bc = np.linalg.norm(bc)
    if n_ba < 1e-6 or n_bc < 1e-6:
        return 0.0
    cos_val = np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)
    return float(math.degrees(math.acos(cos_val)) / 180.0)

def select_primary_person(pose_result, frame_w, frame_h):
    """Pick the most prominent/central lifter when background gym-goers appear."""
    if pose_result.keypoints is None or len(pose_result.keypoints.xy) == 0:
        return None, None
    boxes = pose_result.boxes.xyxy.cpu().numpy()
    kps_xy = pose_result.keypoints.xy.cpu().numpy()
    kps_cf = pose_result.keypoints.conf.cpu().numpy() if pose_result.keypoints.conf is not None else np.ones((len(kps_xy), 17))

    best_idx, best_score = 0, -1.0
    cx_img, cy_img = frame_w / 2.0, frame_h / 2.0
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        area = max(1.0, (x2 - x1) * (y2 - y1))
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dist_center = math.hypot((cx - cx_img) / frame_w, (cy - cy_img) / frame_h)
        score = area * (1.0 / (1.0 + 1.5 * dist_center))
        if score > best_score:
            best_score = score
            best_idx = i
    return kps_xy[best_idx], kps_cf[best_idx]

def build_frame_vector(pose_result, eq_result, frame_w, frame_h):
    """
    Build a 54-dim camera-invariant feature vector:
      [0:34]  -> 17 (x, y) COCO keypoints centered at mid-hip & scaled by torso length
      [34:42] -> 8 bilateral joint angles (L/R elbow, shoulder, hip, knee) in [0, 1]
      [42:46] -> 4 equipment max-confidence scores (barbell, dumbbell, trap_bar, weightplate)
      [46:54] -> 8 normalized L/R wrist-to-equipment distances (clipped to [0, 5.0])
    """
    vec = np.zeros(54, dtype=np.float32)
    vec[46:54] = MAX_NORM_DIST  # default when equipment is absent

    kps_xy, kps_cf = select_primary_person(pose_result, frame_w, frame_h)
    has_pose = kps_xy is not None and np.mean(kps_cf) > 0.15

    # 1. Pose geometry (42 dims)
    if has_pose:
        mid_hip = (kps_xy[11] + kps_xy[12]) / 2.0
        mid_sh  = (kps_xy[5]  + kps_xy[6])  / 2.0
        torso_len = float(np.linalg.norm(mid_sh - mid_hip))
        scale = max(torso_len, 0.05 * max(frame_w, frame_h))

        norm_kps = (kps_xy - mid_hip) / scale
        # Zero out low-confidence keypoints so noise doesn't explode
        norm_kps[kps_cf < 0.25] = 0.0
        vec[0:34] = np.clip(norm_kps.flatten(), -5.0, 5.0)

        # 8 anatomical joint angles
        vec[34] = _angle_deg(kps_xy[5],  kps_xy[7],  kps_xy[9])   # L elbow
        vec[35] = _angle_deg(kps_xy[6],  kps_xy[8],  kps_xy[10])  # R elbow
        vec[36] = _angle_deg(kps_xy[11], kps_xy[5],  kps_xy[7])   # L shoulder
        vec[37] = _angle_deg(kps_xy[12], kps_xy[6],  kps_xy[8])   # R shoulder
        vec[38] = _angle_deg(kps_xy[5],  kps_xy[11], kps_xy[13])  # L hip
        vec[39] = _angle_deg(kps_xy[6],  kps_xy[12], kps_xy[14])  # R hip
        vec[40] = _angle_deg(kps_xy[11], kps_xy[13], kps_xy[15])  # L knee
        vec[41] = _angle_deg(kps_xy[12], kps_xy[14], kps_xy[16])  # R knee

        l_wrist, r_wrist = kps_xy[9], kps_xy[10]
        l_w_ok, r_w_ok = kps_cf[9] >= 0.25, kps_cf[10] >= 0.25
    else:
        scale = 0.25 * max(frame_w, frame_h)
        l_wrist = r_wrist = None
        l_w_ok = r_w_ok = False

    # 2. Equipment priors (12 dims)
    if eq_result.boxes is not None and len(eq_result.boxes) > 0:
        cls_ids = eq_result.boxes.cls.cpu().numpy().astype(int)
        confs   = eq_result.boxes.conf.cpu().numpy()
        xywh    = eq_result.boxes.xywh.cpu().numpy()

        for c in range(4):
            mask = (cls_ids == c)
            if not np.any(mask):
                continue
            vec[42 + c] = float(np.max(confs[mask]))
            centers = xywh[mask, :2]

            if l_w_ok:
                d_l = float(np.min(np.linalg.norm(centers - l_wrist, axis=1)) / scale)
                vec[46 + 2 * c] = min(d_l, MAX_NORM_DIST)
            if r_w_ok:
                d_r = float(np.min(np.linalg.norm(centers - r_wrist, axis=1)) / scale)
                vec[46 + 2 * c + 1] = min(d_r, MAX_NORM_DIST)

    pose_conf = float(np.mean(kps_cf)) if has_pose else 0.0
    return vec, pose_conf
