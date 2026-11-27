# WorkoutAnalyzerAppV2/app.py
import base64
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src" / "phase3"))

from kinematics import extract_keyframes_b64
from phase3_pipeline import run_phase3_pipeline

st.set_page_config(
    page_title="WorkoutAnalyzerApp V2 — Grounded Edge CV",
    page_icon="🏋️",
    layout="wide",
)

# --- Sidebar Controls ---
st.sidebar.title("⚙️ Edge Pipeline Controls")
st.sidebar.caption("Sequential 4 GB VRAM Orchestrator (GTX 1050 Ti)")

vlm_model = st.sidebar.text_input("Local Ollama VLM Tag", value="bonsai-2:4b")
skip_vlm = st.sidebar.checkbox("Stage 1 Only (Skip VLM Critique)", value=False)
live_preview = st.sidebar.checkbox("Stream Live Frame-by-Frame Overlay", value=True)

preset = st.sidebar.radio(
    "Phase 2 Ensemble Operating Point",
    options=[
        "0.3 TCN / 0.7 RF (Best Top-2 VLM Prior — 82.5% OOF)",
        "0.7 TCN / 0.3 RF (Best Standalone Top-1 — 68.3% OOF)",
        "Custom Blend",
    ],
    index=0,
)

if preset.startswith("0.3"):
    w_tcn, w_rf = 0.3, 0.7
elif preset.startswith("0.7"):
    w_tcn, w_rf = 0.7, 0.3
else:
    w_tcn = st.sidebar.slider("TCN Weight (w_tcn)", 0.0, 1.0, 0.3, 0.05)
    w_rf = round(1.0 - w_tcn, 2)
    st.sidebar.caption(f"Random Forest Weight (w_rf): **{w_rf:.2f}**")

with st.sidebar.expander("Locked Model Artifacts", expanded=False):
    eq_weights = st.text_input(
        "YOLO11s Equipment Weights",
        value=str(ROOT_DIR / "models" / "yolo11s_v3_1_640_best.pt"),
    )
    tcn_weights = st.text_input(
        "TCN-96 Checkpoint",
        value=str(ROOT_DIR / "models" / "phase2_tcn96_c64_best.pt"),
    )
    rf_weights = st.text_input(
        "Random Forest Bundle",
        value=str(ROOT_DIR / "models" / "phase2_rf_clipstats.joblib"),
    )

# --- Header ---
st.title("🏋️ WorkoutAnalyzerApp V2")
st.markdown(
    "Dual-YOLO perception (`YOLO11s_v3.1` + `YOLO11n-pose`) $\\rightarrow$ "
    "`TCN-96 + RF` soft-voting ensemble $\\rightarrow$ CUDA VRAM flush $\\rightarrow$ "
    "grounded local VLM coaching."
)

uploaded = st.file_uploader("Upload a gym workout clip", type=["mp4", "mov", "avi"])

if uploaded is not None:
    # Compact top action bar so both videos align cleanly side-by-side below
    bar_info, bar_btn = st.columns([2.2, 1.0], vertical_alignment="center")
    with bar_info:
        st.markdown(
            f"**Clip:** `{uploaded.name}` ({uploaded.size / (1024**2):.2f} MB) &nbsp;|&nbsp; "
            f"**Ensemble:** `{w_tcn:.2f} TCN + {w_rf:.2f} RF` &nbsp;|&nbsp; "
            f"**VLM:** `{'Disabled' if skip_vlm else vlm_model}`"
        )
    with bar_btn:
        run_btn = st.button("🚀 Analyze Workout Clip", type="primary", use_container_width=True)

    # Side-by-side video panels
    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        st.subheader("🎥 Raw Input Clip")
        st.video(uploaded)

    with col_right:
        st.subheader("🎯 Dual-YOLO Perception (Equipment + Pose)")
        right_video_slot = st.empty()
        right_progress_slot = st.empty()

        # If we already analyzed this exact uploaded clip, keep showing the annotated MP4
        same_clip = st.session_state.get("last_filename") == uploaded.name
        if not run_btn:
            if same_clip and st.session_state.get("last_annotated_video"):
                right_video_slot.video(st.session_state["last_annotated_video"])
            else:
                right_video_slot.info(
                    "Click **Analyze Workout Clip** to stream frame-by-frame "
                    "`YOLO11s_v3.1` equipment boxes + `YOLO11n-pose` skeletons here."
                )

    if run_btn:
        suffix = Path(uploaded.name).suffix or ".mp4"
        tmp_in_path = None
        tmp_out_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
                tmp_in.write(uploaded.getbuffer())
                tmp_in_path = tmp_in.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                tmp_out_path = tmp_out.name

            progress_bar = right_progress_slot.progress(0.0, text="Initializing Dual-YOLO on GTX 1050 Ti...")

            def on_frame(vis_rgb, idx, total):
                pct = min(1.0, idx / max(1, total))
                if live_preview:
                    right_video_slot.image(
                        vis_rgb,
                        caption=f"Live Perception Stream — Sampled Frame {idx}/{total}",
                        use_container_width=True,
                    )
                progress_bar.progress(pct, text=f"Stage 1A Perception: {int(pct * 100)}% ({idx}/{total} frames)")

            with st.status("Running sequential GTX 1050 Ti pipeline...", expanded=True) as status:
                def on_stage(msg):
                    st.write(msg)
                    if "Stage 2" in msg:
                        progress_bar.progress(1.0, text=msg)

                result = run_phase3_pipeline(
                    video_path=tmp_in_path,
                    eq_weights=eq_weights,
                    tcn_weights=tcn_weights,
                    rf_weights=rf_weights,
                    vlm_model=vlm_model,
                    w_tcn=w_tcn,
                    w_rf=w_rf,
                    skip_vlm=skip_vlm,
                    annotated_video_out=tmp_out_path,
                    frame_callback=on_frame,
                    stage_callback=on_stage,
                )
                result["prior_payload"]["video_file"] = uploaded.name

                st.write("Extracting 3 grounding keyframes...")
                kf_slots = result["prior_payload"]["biomechanics"]["keyframes"]
                keyframes_b64 = extract_keyframes_b64(tmp_in_path, kf_slots, max_width=640)

                annotated_bytes = None
                if os.path.exists(tmp_out_path) and os.path.getsize(tmp_out_path) > 0:
                    with open(tmp_out_path, "rb") as vf:
                        annotated_bytes = vf.read()

                status.update(
                    label=f"Analysis complete in {result['telemetry']['total_sec']}s",
                    state="complete",
                    expanded=False,
                )

            right_progress_slot.empty()
            if annotated_bytes:
                right_video_slot.video(annotated_bytes)

            st.session_state["last_result"] = result
            st.session_state["last_keyframes"] = keyframes_b64
            st.session_state["last_annotated_video"] = annotated_bytes
            st.session_state["last_filename"] = uploaded.name
        finally:
            if tmp_in_path and os.path.exists(tmp_in_path):
                os.remove(tmp_in_path)
            if tmp_out_path and os.path.exists(tmp_out_path):
                os.remove(tmp_out_path)

# --- Render Results from Session State ---
if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    kf_b64 = st.session_state.get("last_keyframes", {})
    annotated_bytes = st.session_state.get("last_annotated_video")
    prior = res["prior_payload"]
    bio = prior["biomechanics"]
    priors_ens = prior["activity_priors"]
    tel = res["telemetry"]
    stages = tel["stages"]

    st.divider()

    # 1. Edge Telemetry Strip
    st.subheader("⚡ Local Edge Telemetry (GTX 1050 Ti)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Latency", f"{tel['total_sec']:.2f}s")
    s1_sec = stages.get("stage1a_perception_sec", 0.0) + stages.get("stage1b_ensemble_sec", 0.0)
    m2.metric("Stage 1 (YOLO + Ensemble)", f"{s1_sec:.2f}s")
    s2_sec = stages.get("stage2_vlm_sec")
    m3.metric("Stage 2 (Local VLM)", f"{s2_sec:.2f}s" if s2_sec is not None else "Skipped")
    vram_pre = stages.get("vram_pre_vlm", {})
    if vram_pre.get("cuda"):
        m4.metric(
            "Peak PyTorch VRAM",
            f"{vram_pre['max_allocated_mb']:.1f} MB",
            delta=f"{vram_pre['device_used_mb']:.0f} / {vram_pre['device_total_mb']:.0f} MB post-flush",
            delta_color="off",
        )
    else:
        m4.metric("Device", "CPU")

    # 2. Grounded VLM Critique
    if res.get("vlm_critique"):
        st.subheader(f"🧠 Grounded Coaching Critique (`{vlm_model}`)")
        with st.container(border=True):
            st.markdown(res["vlm_critique"])

    # 3. Three Extracted Keyframes
    if kf_b64:
        st.subheader("🎞️ Grounding Keyframes Passed to VLM")
        kf_cols = st.columns(3)
        labels = [
            ("setup_frame_idx", "1. Setup"),
            ("peak_inflection_frame_idx", "2. Peak Inflection"),
            ("finish_frame_idx", "3. Finish / Lockout"),
        ]
        for col, (key, title) in zip(kf_cols, labels):
            if key in kf_b64:
                f_idx = bio["keyframes"].get(key, "?")
                col.image(
                    base64.b64decode(kf_b64[key]),
                    caption=f"{title} (Frame #{f_idx})",
                    use_container_width=True,
                )

    # 4. Two-Column Deep Dive: Activity Priors & Biomechanics
    st.divider()
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("🎯 Phase 2 Ensemble Priors (Top-2)")
        top2 = priors_ens["top2_candidates"]
        df_top2 = pd.DataFrame(top2)
        st.dataframe(
            df_top2[["rank", "movement_family", "blend_prob", "tcn_prob", "rf_prob"]],
            use_container_width=True,
            hide_index=True,
        )
        chart_df = df_top2.set_index("movement_family")[["blend_prob", "tcn_prob", "rf_prob"]]
        st.bar_chart(chart_df, stack=False)

        st.subheader("🏋️ Detected Equipment")
        eq = bio.get("detected_equipment", {})
        if eq:
            eq_rows = [
                {
                    "equipment": name,
                    "presence_%": info["frame_presence_pct"],
                    "max_conf": info["max_conf"],
                    "min_wrist_dist": info["min_wrist_dist_torso"],
                    "in_hands": "✅" if info["in_hands_likely"] else "—",
                }
                for name, info in eq.items()
            ]
            st.dataframe(pd.DataFrame(eq_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No equipment exceeded presence threshold (may be bodyweight or hand-occluded).")

    with right:
        st.subheader("📐 Biomechanical Kinematics")
        k1, k2, k3 = st.columns(3)
        k1.metric("Estimated Reps", bio["estimated_reps"])
        k2.metric("Tracked Joint", bio["primary_joint_tracked"].capitalize())
        k3.metric("Pose Confidence", f"{prior['mean_pose_conf']:.2f}")

        st.caption(f"**Rep Tracking Mode:** `{bio.get('rep_tracking_mode', 'n/a')}`")
        if bio.get("camera_projection_note"):
            if "foreshortened" in bio["camera_projection_note"].lower():
                st.warning(bio["camera_projection_note"])
            else:
                st.info(bio["camera_projection_note"])

        rom = bio.get("joint_angles_deg", {})
        if rom:
            rom_rows = [
                {
                    "joint": j.capitalize(),
                    "min_deg": vals["min_deg"],
                    "max_deg": vals["max_deg"],
                    "rom_deg": vals["rom_deg"],
                }
                for j, vals in rom.items()
            ]
            st.dataframe(pd.DataFrame(rom_rows), use_container_width=True, hide_index=True)

    # 5. Exports & Raw Telemetry
    st.divider()
    stem = Path(st.session_state.get("last_filename", "workout")).stem
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="⬇️ Download Full JSON Report",
            data=json.dumps(res, indent=2),
            file_name=f"{stem}_phase3_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with dl2:
        if annotated_bytes:
            st.download_button(
                label="🎬 Download Annotated Dual-YOLO Video (MP4)",
                data=annotated_bytes,
                file_name=f"{stem}_dual_yolo_overlay.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

    with st.expander("Inspect Raw JSON Payload & Telemetry"):
        st.json(res)