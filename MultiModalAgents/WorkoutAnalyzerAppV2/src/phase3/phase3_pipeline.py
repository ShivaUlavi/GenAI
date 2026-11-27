# src/phase3/phase3_pipeline.py
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import joblib
import numpy as np
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src" / "phase2"))
sys.path.insert(0, str(ROOT_DIR / "src" / "phase3"))

from phase2_features import build_frame_vector
from phase2_models import TemporalConvNet
from kinematics import add_velocity_96, analyze_biomechanics, build_clip_summary_vector, extract_keyframes_b64
from vlm_grounding import run_grounded_vlm_critique


class _H264VideoWriter:
    """Streams annotated BGR frames to a browser-playable H.264 MP4 on disk."""

    def __init__(self, out_path: str, fps: float, width: int, height: int):
        self.out_path = out_path
        self.width = width - (width % 2)
        self.height = height - (height % 2)
        self.backend = None
        self.writer = None

        try:
            import imageio.v2 as imageio

            self.writer = imageio.get_writer(
                out_path,
                fps=max(1.0, float(fps)),
                codec="libx264",
                pixelformat="yuv420p",
                quality=8,
                macro_block_size=1,
            )
            self.backend = "imageio"
        except Exception:
            for fourcc_str in ("avc1", "H264", "mp4v"):
                w = cv2.VideoWriter(
                    out_path,
                    cv2.VideoWriter_fourcc(*fourcc_str),
                    max(1.0, float(fps)),
                    (self.width, self.height),
                )
                if w.isOpened():
                    self.writer = w
                    self.backend = f"cv2:{fourcc_str}"
                    break

    def write_bgr(self, frame_bgr: np.ndarray):
        if self.writer is None:
            return
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.backend == "imageio":
            self.writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        else:
            self.writer.write(frame_bgr)

    def close(self):
        if self.writer is None:
            return
        if self.backend == "imageio":
            self.writer.close()
        else:
            self.writer.release()
        self.writer = None


def _compose_dual_overlay(
    pose_res,
    eq_res,
    frame_idx: int,
    max_dim: int = 720,
) -> np.ndarray:
    """Render COCO-17 skeleton + custom gym equipment bounding boxes on one BGR frame."""
    # Draw skeleton first (hide noisy person bbox), then overlay equipment detections on top
    vis = pose_res.plot(boxes=False, kpt_line=True, kpt_radius=4, line_width=2)
    vis = eq_res.plot(img=vis, line_width=2)

    h, w = vis.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        nw = max(2, int(round(w * scale)) // 2 * 2)
        nh = max(2, int(round(h * scale)) // 2 * 2)
        vis = cv2.resize(vis, (nw, nh), interpolation=cv2.INTER_AREA)

    # Subtle translucent top HUD bar for clean demo / LinkedIn presentation
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (vis.shape[1], 34), (15, 18, 25), -1)
    vis = cv2.addWeighted(overlay, 0.65, vis, 0.35, 0)
    cv2.putText(
        vis,
        f"YOLO11s_v3.1 (Equip) + YOLO11n-pose | Frame #{frame_idx}",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (240, 248, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


def vram_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "cuda": True,
        "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 1),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
        "device_used_mb": round((total_b - free_b) / (1024**2), 1),
        "device_total_mb": round(total_b / (1024**2), 1),
    }


def flush_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def extract_video_features(
    video_path: str,
    eq_weights_path: str,
    pose_weights_path: str = "yolo11n-pose.pt",
    target_fps: float = 10.0,
    imgsz: int = 640,
    device: str = "cuda:0",
    annotated_video_out: Optional[str] = None,
    frame_callback: Optional[Callable[[np.ndarray, int, int], None]] = None,
):
    """Stage 1A: Dual-YOLO perception -> [T, 54] features + optional live/saved dual-overlay video."""
    pose_model = YOLO(pose_weights_path)
    eq_model = YOLO(eq_weights_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(orig_fps / target_fps)))
    eff_fps = orig_fps / stride
    est_sampled = max(1, total_frames // stride) if total_frames > 0 else 1

    feats, frame_indices, pose_confs = [], [], []
    video_writer = None
    f_idx = 0
    sampled_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if f_idx % stride == 0:
                h, w = frame.shape[:2]
                pose_res = pose_model.predict(frame, imgsz=imgsz, device=device, verbose=False)[0]
                eq_res = eq_model.predict(frame, imgsz=imgsz, device=device, verbose=False)[0]
                vec, p_conf = build_frame_vector(pose_res, eq_res, w, h)
                feats.append(vec)
                frame_indices.append(f_idx)
                pose_confs.append(p_conf)
                sampled_idx += 1

                if annotated_video_out or frame_callback:
                    vis_bgr = _compose_dual_overlay(pose_res, eq_res, f_idx, max_dim=720)
                    if annotated_video_out:
                        if video_writer is None:
                            vh, vw = vis_bgr.shape[:2]
                            video_writer = _H264VideoWriter(annotated_video_out, eff_fps, vw, vh)
                        video_writer.write_bgr(vis_bgr)
                    if frame_callback:
                        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
                        frame_callback(vis_rgb, sampled_idx, est_sampled)
            f_idx += 1
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.close()

    del pose_model, eq_model
    flush_vram()

    if len(feats) == 0:
        raise RuntimeError(f"No frames extracted from {video_path}")

    return {
        "seq_54": np.stack(feats, axis=0).astype(np.float32),
        "frame_indices": frame_indices,
        "eff_fps": round(eff_fps, 2),
        "orig_fps": round(orig_fps, 2),
        "duration_sec": round(f_idx / orig_fps, 2),
        "mean_pose_conf": round(float(np.mean(pose_confs)), 3),
    }


def score_phase2_ensemble(
    seq_54: np.ndarray,
    tcn_ckpt_path: str,
    rf_joblib_path: str,
    w_tcn: float = 0.3,
    w_rf: float = 0.7,
    device: str = "cuda:0",
) -> dict:
    """Stage 1B: Soft-voting blend of TCN_96_c64 (0.34 MB) + RandomForest_ClipStats."""
    ckpt = torch.load(tcn_ckpt_path, map_location=device, weights_only=False)
    tcn_classes = ckpt["classes"]
    win_len = int(ckpt.get("win_len", 32))
    in_dim = int(ckpt.get("in_dim", 96))
    channels = int(ckpt.get("channels", 64))

    tcn = TemporalConvNet(in_dim=in_dim, channels=channels, num_classes=len(tcn_classes)).to(device)
    tcn.load_state_dict(ckpt["state_dict"])
    tcn.eval()

    T = len(seq_54)
    windows = []
    if T <= win_len:
        pad = np.repeat(seq_54[-1:], win_len - T, axis=0)
        windows.append(add_velocity_96(np.concatenate([seq_54, pad], axis=0)))
    else:
        starts = list(range(0, T - win_len + 1, 8))
        if starts[-1] != T - win_len:
            starts.append(T - win_len)
        for s in starts:
            windows.append(add_velocity_96(seq_54[s : s + win_len]))

    xb = torch.from_numpy(np.stack(windows, axis=0)).to(device)
    with torch.no_grad():
        probs_tcn = torch.softmax(tcn(xb), dim=-1).mean(dim=0).cpu().numpy()

    del tcn, xb, ckpt
    flush_vram()

    rf_bundle = joblib.load(rf_joblib_path)
    rf_model = rf_bundle["model"]
    rf_classes = rf_bundle["classes"]
    assert list(tcn_classes) == list(rf_classes), "Class ordering mismatch between TCN and RF artifacts!"

    x_clip = build_clip_summary_vector(seq_54).reshape(1, -1)
    probs_rf = rf_model.predict_proba(x_clip)[0]

    blend = w_tcn * probs_tcn + w_rf * probs_rf
    top_idx = np.argsort(blend)[::-1]

    top2 = [
        {
            "rank": rank + 1,
            "movement_family": tcn_classes[i],
            "blend_prob": round(float(blend[i]), 4),
            "tcn_prob": round(float(probs_tcn[i]), 4),
            "rf_prob": round(float(probs_rf[i]), 4),
        }
        for rank, i in enumerate(top_idx[:2])
    ]

    return {
        "blend_weights": {"w_tcn": w_tcn, "w_rf": w_rf},
        "top2_candidates": top2,
        "top1_label": top2[0]["movement_family"],
    }


def run_phase3_pipeline(
    video_path: str,
    eq_weights: str,
    tcn_weights: str,
    rf_weights: str,
    vlm_model: str = "bonsai-2:4b",
    w_tcn: float = 0.3,
    w_rf: float = 0.7,
    skip_vlm: bool = False,
    annotated_video_out: Optional[str] = None,
    frame_callback: Optional[Callable[[np.ndarray, int, int], None]] = None,
    stage_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    telemetry = {"device": device, "stages": {}}
    t_total = time.perf_counter()

    # --- STAGE 1A: Perception (+ Live/Saved Dual-Overlay) ---
    if stage_callback:
        stage_callback("Stage 1A: Running Dual-YOLO perception (equipment + pose)...")
    t0 = time.perf_counter()
    perc = extract_video_features(
        video_path,
        eq_weights,
        device=device,
        annotated_video_out=annotated_video_out,
        frame_callback=frame_callback,
    )
    telemetry["stages"]["stage1a_perception_sec"] = round(time.perf_counter() - t0, 2)
    telemetry["stages"]["vram_after_stage1a_flush"] = vram_snapshot()

    # --- STAGE 1B: Phase 2 Soft-Voting Ensemble ---
    if stage_callback:
        stage_callback("Stage 1B: Scoring TCN-96 + RandomForest soft-voting ensemble...")
    t1 = time.perf_counter()
    ens = score_phase2_ensemble(perc["seq_54"], tcn_weights, rf_weights, w_tcn=w_tcn, w_rf=w_rf, device=device)
    telemetry["stages"]["stage1b_ensemble_sec"] = round(time.perf_counter() - t1, 2)

    # --- STAGE 1C: Kinematics, Rep Count & 3 Keyframes ---
    bio = analyze_biomechanics(
        perc["seq_54"],
        perc["frame_indices"],
        ens["top1_label"],
        perc["eff_fps"],
        top1_prob=ens["top2_candidates"][0]["blend_prob"],
    )
    keyframes_b64 = extract_keyframes_b64(video_path, bio["keyframes"])

    # --- STAGE 1D: Final Pre-VLM VRAM Verification ---
    flush_vram()
    telemetry["stages"]["vram_pre_vlm"] = vram_snapshot()

    prior_payload = {
        "video_file": os.path.basename(video_path),
        "duration_sec": perc["duration_sec"],
        "mean_pose_conf": perc["mean_pose_conf"],
        "activity_priors": ens,
        "biomechanics": bio,
    }

    # --- STAGE 2: Grounded VLM Critique ---
    critique = None
    if not skip_vlm:
        if stage_callback:
            stage_callback(f"Stage 2: VRAM flushed — querying local VLM (`{vlm_model}`)...")
        t2 = time.perf_counter()
        critique = run_grounded_vlm_critique(prior_payload, keyframes_b64, model_name=vlm_model)
        telemetry["stages"]["stage2_vlm_sec"] = round(time.perf_counter() - t2, 2)
        telemetry["stages"]["vram_post_vlm"] = vram_snapshot()

    telemetry["total_sec"] = round(time.perf_counter() - t_total, 2)

    return {
        "prior_payload": prior_payload,
        "vlm_critique": critique,
        "telemetry": telemetry,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 Local GTX 1050 Ti Workout Analytics Orchestrator")
    parser.add_argument("--video", required=True, help="Path to local workout .mp4 clip")
    parser.add_argument("--eq-weights", default=str(ROOT_DIR / "models" / "yolo11s_v3_1_640_best.pt"))
    parser.add_argument("--tcn-weights", default=str(ROOT_DIR / "models" / "phase2_tcn96_c64_best.pt"))
    parser.add_argument("--rf-weights", default=str(ROOT_DIR / "models" / "phase2_rf_clipstats.joblib"))
    parser.add_argument("--model", default="bonsai-2:4b", help="Local Ollama model tag")
    parser.add_argument("--w-tcn", type=float, default=0.3)
    parser.add_argument("--w-rf", type=float, default=0.7)
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--save-video", default="", help="Optional path to save annotated dual-YOLO MP4")
    parser.add_argument("--out-json", default="", help="Optional path to save full JSON output")
    args = parser.parse_args()

    result = run_phase3_pipeline(
        video_path=args.video,
        eq_weights=args.eq_weights,
        tcn_weights=args.tcn_weights,
        rf_weights=args.rf_weights,
        vlm_model=args.model,
        w_tcn=args.w_tcn,
        w_rf=args.w_rf,
        skip_vlm=args.skip_vlm,
        annotated_video_out=args.save_video or None,
    )

    print("\n=== STAGE 1 STRUCTURED PRIOR ===")
    print(json.dumps(result["prior_payload"], indent=2))
    print("\n=== EDGE TELEMETRY (GTX 1050 Ti) ===")
    print(json.dumps(result["telemetry"], indent=2))
    if result["vlm_critique"]:
        print("\n=== STAGE 2 GROUNDED VLM CRITIQUE ===")
        print(result["vlm_critique"])

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved full report -> {args.out_json}")