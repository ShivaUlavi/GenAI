import os
import glob
import json
import base64
import uuid
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def encode_image_to_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_keyframes(video_path, sample_interval_sec=2, max_resolution=(640, 360)):
    """Extracts keyframes with unique filenames to prevent race conditions during parallel execution."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[-] Could not open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * sample_interval_sec) if fps > 0 else 30
    
    extracted_paths = []
    frame_count = 0
    unique_session_id = uuid.uuid4().hex[:8]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frame_resized = cv2.resize(frame, max_resolution)
            temp_filename = f"temp_{unique_session_id}_frame_{len(extracted_paths)}.jpg"
            cv2.imwrite(temp_filename, frame_resized)
            extracted_paths.append(temp_filename)
            
        frame_count += 1

    cap.release()
    return extracted_paths

def process_single_video(video_path, llm, prompt, output_dir):
    """Worker function to process a single video end-to-end safely in a thread."""
    filename = os.path.basename(video_path)
    base_name = os.path.splitext(filename)[0]
    md_file = os.path.join(output_dir, f"{base_name}_analysis.md")
    
    # Skip if already processed
    if os.path.exists(md_file):
        print(f"[ℹ] Skipping already processed: {filename}")
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read().replace(f"# Analysis Report: {filename}\n\n", "")
        return filename, content

    print(f"\n[+] Starting: {filename}")
    keyframes = extract_keyframes(video_path, sample_interval_sec=2, max_resolution=(640, 360))
    
    if not keyframes:
        return filename, "Failed to extract keyframes."

    print(f"    -> Extracted {len(keyframes)} keyframes for {filename}.")

    # Build prompt payload
    message_content = [{"type": "text", "text": prompt}]
    for path in keyframes:
        b64_str = encode_image_to_base64(path)
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
        })

    try:
        response = llm.invoke([HumanMessage(content=message_content)])
        analysis_text = response.content
    except Exception as e:
        analysis_text = f"Error during inference: {str(e)}"
    finally:
        # Clean up temporary keyframe image files specific to this session
        for path in keyframes:
            if os.path.exists(path):
                os.remove(path)

    # Save individual Markdown report
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# Analysis Report: {filename}\n\n{analysis_text}")

    print(f"[✓] Completed & Saved: {filename}")
    return filename, analysis_text

def run_batch_pipeline(folder_path, output_dir="analysis_outputs", max_workers=2):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Bonsai 2 (4B) model
    llm = ChatOllama(model="bonsai-2:4b", num_ctx=16384)
    
    prompt = (
        "Analyze this sequence of frames from a gym workout clip. "
        "Explicitly identify the exercise being performed (e.g., weighted dips, pull-ups, squats, "
        "farmer's walk, deadlifts), evaluate form stability, posture, and range of motion, "
        "and suggest any potential improvements."
    )

    video_extensions = ("*.mp4", "*.mov", "*.avi", "*.mkv")
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(folder_path, ext)))

    print(f"[+] Found {len(video_files)} videos in {folder_path}")
    print(f"[+] Running parallel pipeline with max_workers={max_workers} (optimized for 4GB VRAM)")

    batch_results = {}

    # Use ThreadPoolExecutor to handle parallel processing safely
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {
            executor.submit(process_single_video, video_path, llm, prompt, output_dir): video_path
            for video_path in video_files
        }

        for future in as_completed(future_to_video):
            try:
                filename, analysis = future.result()
                batch_results[filename] = analysis
            except Exception as exc:
                video_path = future_to_video[future]
                print(f"[-] Video {os.path.basename(video_path)} generated an exception: {exc}")

    # Save consolidated JSON summary
    json_summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(batch_results, f, indent=4)

    print(f"\n[+] Batch processing complete! All reports saved in '{output_dir}'.")

if __name__ == "__main__":
    TARGET_FOLDER = r"C:\Users\Shivachetan Ulavi\Desktop\workout_test"
    run_batch_pipeline(TARGET_FOLDER, max_workers=2)
