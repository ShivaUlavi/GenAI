import cv2
import os

def extract_keyframes(video_path, sample_interval_sec=2, max_resolution=(640, 360)):
    """Extracts keyframes at specified time intervals and resizes them."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * sample_interval_sec) if fps > 0 else 30
    
    extracted_frames = []
    frame_count = 0
    saved_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frame_resized = cv2.resize(frame, max_resolution)
            temp_filename = f"st_frame_{saved_idx}.jpg"
            cv2.imwrite(temp_filename, frame_resized)
            extracted_frames.append(temp_filename)
            saved_idx += 1
            
        frame_count += 1

    cap.release()
    return extracted_frames

def cleanup_files(file_paths):
    """Safely removes temporary image and video files."""
    for path in file_paths:
        if path and os.path.exists(path):
            os.remove(path)