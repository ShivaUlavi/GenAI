# src/phase3/kinematics.py
import base64
import cv2
import numpy as np
from scipy.signal import find_peaks

EQ_CLASSES = ["barbell", "dumbbell", "trap_bar", "weightplate"]

# Indices in vec[0:54] for bilateral joint angles (stored normalized in [0, 1])
ANGLE_MAP = {
    "elbow": (34, 35),
    "shoulder": (36, 37),
    "hip": (38, 39),
    "knee": (40, 41),
}

# Primary biomechanical signal per movement family: (joint_group, track_min_for_bottom)
EXERCISE_SIGNAL_SPEC = {
    "squat": ("knee", True),
    "lunge": ("knee", True),
    "deadlift": ("hip", True),
    "row": ("elbow", True),
    "curl": ("elbow", True),
    "pull_up": ("elbow", True),
    "dip": ("elbow", True),
    "press": ("elbow", True),
    "lateral_raise": ("shoulder", False),  # Peak abduction = max shoulder angle
    "core": ("hip", True),
    "shrug": ("shoulder", False),
    "farmers_walk": ("knee", True),
    "idle": ("hip", True),
}


def add_velocity_96(seq_54: np.ndarray) -> np.ndarray:
    """Append 42-dim frame-to-frame velocity (34 pose coords + 8 angles) -> [T, 96]."""
    vel = np.diff(seq_54[:, :42], axis=0, prepend=seq_54[:1, :42])
    return np.concatenate([seq_54, vel], axis=1).astype(np.float32)


def build_clip_summary_vector(seq_54: np.ndarray) -> np.ndarray:
    """Exact clip-level kinematic + motion-energy vector expected by phase2_rf_clipstats."""
    seq96 = add_velocity_96(seq_54)
    pose_ang = seq96[:, :42]
    eq_conf = seq96[:, 42:46]
    eq_dist = seq96[:, 46:54]
    vel_abs = np.abs(seq96[:, 54:96])

    return np.concatenate([
        pose_ang.mean(axis=0),
        pose_ang.std(axis=0),
        np.percentile(pose_ang, 90, axis=0) - np.percentile(pose_ang, 10, axis=0),
        eq_conf.mean(axis=0),
        (eq_conf > 0.25).mean(axis=0),
        eq_dist.min(axis=0),
        eq_dist.mean(axis=0),
        vel_abs.mean(axis=0),
        np.percentile(vel_abs, 90, axis=0),
    ]).astype(np.float32)


def _smooth_1d(signal: np.ndarray, k: int = 5) -> np.ndarray:
    if len(signal) < k:
        return signal.copy()
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(np.pad(signal, (k // 2, k // 2), mode="edge"), kernel, mode="valid")[: len(signal)]


def _bilateral_angle_deg(seq_54: np.ndarray, joint_group: str) -> np.ndarray:
    """Average valid L/R joint angles in degrees, ignoring zeroed-out occluded joints."""
    l_idx, r_idx = ANGLE_MAP[joint_group]
    l_deg = seq_54[:, l_idx] * 180.0
    r_deg = seq_54[:, r_idx] * 180.0

    valid_l = l_deg > 1.0
    valid_r = r_deg > 1.0
    out = np.zeros(len(seq_54), dtype=np.float32)
    both = valid_l & valid_r
    out[both] = 0.5 * (l_deg[both] + r_deg[both])
    out[valid_l & ~valid_r] = l_deg[valid_l & ~valid_r]
    out[~valid_l & valid_r] = r_deg[~valid_l & valid_r]
    return out


def _vertical_depth_ratio(seq_54: np.ndarray) -> np.ndarray:
    """Front-view-safe vertical motion proxy: mean wrist_y divided by mean ankle_y."""
    # COCO normalized y indices in seq_54: wrists = (19, 21), ankles = (31, 33)
    wrist_y = 0.5 * (seq_54[:, 19] + seq_54[:, 21])
    ankle_y = 0.5 * (seq_54[:, 31] + seq_54[:, 33])
    denom = np.where(np.abs(ankle_y) > 0.2, np.abs(ankle_y), 1.5)
    return np.clip(wrist_y / denom, -2.0, 2.0).astype(np.float32)


def analyze_biomechanics(seq_54: np.ndarray, frame_indices: list[int], top1_label: str, eff_fps: float, top1_prob: float = 1.0) -> dict:
    """Compute detected equipment, joint ROMs, rep count, and 3 representative frame indices."""
    T = len(seq_54)

    # 1. Equipment summary
    eq_summary = {}
    for c, name in enumerate(EQ_CLASSES):
        confs = seq_54[:, 42 + c]
        present_ratio = float((confs > 0.25).mean())
        if present_ratio >= 0.15:
            min_wrist_dist = float(np.min(seq_54[:, 46 + 2 * c : 46 + 2 * c + 2]))
            eq_summary[name] = {
                "frame_presence_pct": round(present_ratio * 100.0, 1),
                "max_conf": round(float(confs.max()), 3),
                "min_wrist_dist_torso": round(min_wrist_dist, 2),
                "in_hands_likely": bool(min_wrist_dist < 1.5),
            }

    # 2. Bilateral joint range-of-motion (p2/p98 preserves brief bottom-of-rep inflections)
    rom_stats = {}
    for jname in ANGLE_MAP:
        ang = _bilateral_angle_deg(seq_54, jname)
        valid = ang[ang > 1.0]
        if len(valid) >= 3:
            p02, p98 = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
            rom_stats[jname] = {
                "min_deg": round(p02, 1),
                "max_deg": round(p98, 1),
                "rom_deg": round(p98 - p02, 1),
            }

    # 3. Rep counting & peak-inflection frame index (2D angle + front-view depth fallback)
    joint_group, track_min = EXERCISE_SIGNAL_SPEC.get(top1_label, ("hip", True))
    
    # If ensemble confidence is low, track whichever joint actually moved the most
    if top1_prob < 0.45 and rom_stats:
        joint_group = max(rom_stats, key=lambda j: rom_stats[j]["rom_deg"])
        track_min = joint_group != "shoulder"


    raw_sig = _bilateral_angle_deg(seq_54, joint_group)
    smooth_sig = _smooth_1d(raw_sig, k=5)

    min_dist_frames = max(4, int(round(eff_fps * 0.8)))
    rep_count = 0
    peak_t = T // 2
    tracking_mode = f"2d_angle:{joint_group}"

    if top1_label not in ("idle", "farmers_walk") and T >= 6:
        angle_rom = rom_stats.get(joint_group, {}).get("rom_deg", 0.0)

            #if angle_rom >= 20.0:
            #target = -smooth_sig if track_min else smooth_sig
            #prom = 15.0 if joint_group in ("knee", "elbow") else 12.0
            #peaks, _ = find_peaks(target, distance=min_dist_frames, prominence=prom)
        
        
        #if angle_rom >= 20.0:
         #   target = -smooth_sig if track_min else smooth_sig
          #  # Adaptive prominence stops high-ROM noisy joints (e.g., cropped curls) from double-counting
           # prom = max(15.0 if joint_group in ("knee", "elbow") else 12.0, 0.30 * angle_rom)
            #peaks, _ = find_peaks(target, distance=min_dist_frames, prominence=prom)
        if angle_rom >= 20.0:
            target = -smooth_sig if track_min else smooth_sig
            prom = max(15.0 if joint_group in ("knee", "elbow") else 12.0, 0.30 * angle_rom)
            peaks, _ = find_peaks(target, distance=min_dist_frames, prominence=prom)
            # Single-rep boundary safeguard: if ROM >= 50 deg and clip ends near bottom/top, count 1 rep
            if len(peaks) == 0 and angle_rom >= 50.0:
                peaks = np.array([int(np.argmax(target))])
        
        else:
            # Front-view foreshortening fallback: track vertical wrist-to-ankle descent
            tracking_mode = "vertical_depth_fallback"
            depth_sig = _smooth_1d(_vertical_depth_ratio(seq_54), k=5)
            target = depth_sig  # Higher wrist_y/ankle_y = deeper bottom position
            peaks, _ = find_peaks(target, distance=min_dist_frames, prominence=0.12)

        rep_count = int(len(peaks))
        if len(peaks) > 0:
            peak_t = int(peaks[int(np.argmax(target[peaks]))])
        else:
            peak_t = int(np.argmax(target))

    # 4. Select 3 keyframes: setup, peak inflection, finish/lockout
    #start_t = min(max(0, int(T * 0.12)), max(0, peak_t - 1))
    #end_t = max(min(T - 1, int(T * 0.88)), min(T - 1, peak_t + 1))

    #keyframe_slots = {
     #   "setup_frame_idx": int(frame_indices[start_t]),
      #  "peak_inflection_frame_idx": int(frame_indices[peak_t]),
       # "finish_frame_idx": int(frame_indices[end_t]),
    #}

    # 4. Select 3 well-separated keyframes: prefer a mid-clip peak when multiple reps exist
    if top1_label not in ("idle", "farmers_walk") and T >= 6 and len(peaks) > 1:
        mid_peaks = [p for p in peaks if int(T * 0.22) <= p <= int(T * 0.78)]
        if mid_peaks:
            peak_t = int(mid_peaks[int(np.argmax(target[mid_peaks]))])

    min_gap = max(2, int(T * 0.12))
    start_t = int(T * 0.12)
    end_t = int(T * 0.88)
    if abs(peak_t - start_t) < min_gap:
        start_t = max(0, min(peak_t - min_gap, int(T * 0.05)))
    if abs(end_t - peak_t) < min_gap:
        end_t = min(T - 1, max(peak_t + min_gap, int(T * 0.95)))

    keyframe_slots = {
        "setup_frame_idx": int(frame_indices[start_t]),
        "peak_inflection_frame_idx": int(frame_indices[peak_t]),
        "finish_frame_idx": int(frame_indices[end_t]),
    }

    return {
        "primary_joint_tracked": joint_group,
        "rep_tracking_mode": tracking_mode,
        "estimated_reps": rep_count,
        "detected_equipment": eq_summary,
        "joint_angles_deg": rom_stats,
        "keyframes": keyframe_slots,
        "camera_projection_note": (
            "Front/oblique 2D camera view detected: 2D hip/knee angles are foreshortened and "
            "understate true sagittal flexion. Judge depth from the keyframes and vertical bar travel, "
            "NOT from the raw 2D hip/knee degree numbers."
            if tracking_mode == "vertical_depth_fallback"
            else "Side/sagittal view: 2D joint angles reliably reflect sagittal flexion (180 deg = straight/locked out, smaller angle = deeper bend)."
        ),
    }

def extract_keyframes_b64(video_path: str, keyframe_slots: dict, max_width: int = 512) -> dict[str, str]:
    """Read only the 3 chosen keyframes from the MP4 and encode as resized base64 JPEGs."""
    cap = cv2.VideoCapture(video_path)
    encoded = {}
    try:
        for label, f_idx in keyframe_slots.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                frame = cv2.resize(frame, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
            ok_jpg, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok_jpg:
                encoded[label] = base64.b64encode(buf.tobytes()).decode("utf-8")
    finally:
        cap.release()
    return encoded