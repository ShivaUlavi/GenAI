# Workout Analyzer App

A lightweight multimodal AI application for analyzing workout form from short exercise videos. The app extracts representative frames from a user-uploaded video, passes them to a vision-language model, and returns exercise-specific feedback on form, stability, posture, and movement quality.

## Overview

This project is designed to help users evaluate gym movement quality without requiring a manual video review. The workflow is simple:

1. Upload a workout video clip.
2. The app samples frames at a configurable interval.
3. Extracted frames are resized and saved temporarily.
4. A multimodal model inspects the sequence of frames.
5. The system returns a likely exercise identification and form-related suggestions.

## Key Components

### `app.py`
The Streamlit application provides the user interface for loading a video, selecting sampling parameters, and triggering analysis. It renders the uploaded video, displays extracted keyframes, and shows the model-generated feedback in the main analysis panel.

### `video_processor.py`
This module handles frame extraction using OpenCV. It reads the input video, samples frames at defined time intervals, resizes them to a manageable resolution, and stores them temporarily for model inference.

### `model_handler.py`
This module loads a vision-language model through `langchain_ollama` and prepares image prompts for inference. Each extracted frame is converted into a base64-encoded image payload and sent alongside a text prompt that instructs the model to analyze form, stability, posture, and range of motion.

## Tech Stack

- Python
- Streamlit
- OpenCV
- LangChain
- Ollama
- Vision-language model: `bonsai-2:4b`

## Typical Workflow

- User uploads a workout video in MP4/MOV/AVI/MKV format.
- The app extracts keyframes at a chosen sampling interval.
- The selected frames are analyzed by the multimodal model.
- Results include the likely exercise, movement classification, form assessment, and corrective suggestions.

## Use Case

This project is useful for:

- exercise form review
- gym technique feedback
- quick AI-assisted movement analysis
- experimentation with local multimodal prompting for fitness understanding

## Project Documentation

This README is intentionally kept at a high-level technical overview. Deeper findings and evaluation notes are documented in:

- `results.md` — summary of outcomes, architecture comparison, and practical findings
- `project_findings.md` — technical interpretation, prompt engineering, and next steps
- `evaluation_notes.md` — assessment notes, failure cases, and example-image analysis

## Basic Run Instructions

1. Install the required dependencies.
2. Pull the Ollama model used by the app.
3. Run the Streamlit application from the project folder.

Example:

```bash
streamlit run app.py
```

## Notes

- The app uses temporary files for uploaded videos and extracted frames.
- The model call is cached to avoid reloading the LLM unnecessarily.
- The analysis prompt can be customized in the UI for different movement types or evaluation preferences.

This project is a practical example of combining video processing and multimodal AI to create an interactive exercise-analysis tool.
