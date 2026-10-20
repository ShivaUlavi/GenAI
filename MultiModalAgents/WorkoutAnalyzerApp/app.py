import streamlit as st
import tempfile
import os
from video_processor import extract_keyframes, cleanup_files
from model_handler import load_vlm, analyze_workout_frames

#page configuration
st.set_page_config(
    page_title="Exercise Analyzer AI",
    page_icon = "🏋️‍♂️",
    layout = "wide"
)

st.title("🏋️‍♂️ AI Gym Form Analyzer")
st.markdown("Upload your workout video clip to get instant form analysis, stability scoring, and safety recommendations using **bonsai-2:4b**.")

##intialize model
llm = load_vlm()
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
sample_interval = st.sidebar.slider("Sampling Interval (seconds)", min_value=1, max_value=5, value=2)

custom_prompt = st.sidebar.text_area(
    "Custom Analysis Prompt",
    value=(
        "Analyze this sequence of frames from a gym workout clip. "
        "explicitly identify the exercise being performed (e.g., weighted dips, farmer's walk, squats, deadlifts), evaluating whether it is a stationary or traveling movement. "
        "evaluate form stability, posture, and range of motion, "
        "and suggest any potential improvements."
    ),
    height=120
)


if uploaded_file is not None:
    #save file temporarily
    tfile = tempfile.NamedTemporaryFile(delete = False, suffix = ".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    tfile.close()
    
    col1, col2 = st.columns([1,2])
    
    with col1:
        st.subheader("video preview")
        st.video(video_path)
    
    extracted_frames = extract_keyframes(video_path, sample_interval_sec=sample_interval)
    
    
    with col1:
        st.subheader(f"🖼️ Extracted Keyframes ({len(extracted_frames)})")
        if extracted_frames:
            cols = st.columns(min(len(extracted_frames), 4))
            for idx, img_path in enumerate(extracted_frames):
                cols[idx % 4].image(img_path, caption=f"Frame {idx+1}")
                
    with col2:
        st.subheader("📊 Form Analysis Feedback")
        if st.button("🚀 Analyze Exercise Form", type="primary"):
            if not extracted_frames:
                st.error("No keyframes were extracted from the video.")
            else:
                with st.spinner("Analyzing keyframes....."):
                    analysis_result = analyze_workout_frames(llm, extracted_frames, custom_prompt)
                    st.markdown(analysis_result)

    # Clean up temporary files after session usage
    cleanup_files([video_path] + extracted_frames)

else:
    st.info("👈 Upload a workout video clip from the sidebar to begin.")
                    