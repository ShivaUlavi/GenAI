# MultiModalAgents

This folder contains examples for multimodal AI workflows using local vision-capable models with Ollama and LangChain. It focuses on analyzing images and video clips using prompts that combine visual understanding with LLM reasoning.

## What this folder includes

- `VideoAnalysisInBatches.py` — batch-processing script that extracts frames from videos, sends them to a vision-capable Ollama model, and saves Markdown analysis reports.
- `04_Batch_Video_Analysis_ThreadPool.py` — similar batch video analysis pipeline implemented with `ThreadPoolExecutor` for parallel processing.

## Use cases

- Gym workout video analysis
- Exercise detection and form review
- Posture, motion, and range-of-motion evaluation
- Summarizing key visual cues from multiple frames

## Workflow overview

1. Read video files from a target folder.
2. Extract sampled frames (keyframes) from each video.
3. Convert frames to base64 image data.
4. Send the frames and a prompt to a multimodal model via LangChain.
5. Save per-video Markdown analytical reports and a consolidated JSON summary.

## Requirements

Install the necessary dependencies:

```bash
pip install opencv-python langchain langchain-ollama
```

Also ensure Ollama is installed and running locally, and that you have a compatible multimodal model available, such as a vision-capable model pulled into Ollama.

## Example usage

```python
from MultiModalAgents.VideoAnalysisInBatches import run_batch_pipeline

run_batch_pipeline(r"C:\Users\YourName\Desktop\workout_test", output_dir="analysis_outputs", max_workers=2)
```

Or run the script directly:

```bash
python MultiModalAgents/VideoAnalysisInBatches.py
```

## Notes

- The scripts are configured for local execution and rely on an Ollama-hosted model.
- Keyframes are temporarily created during processing and cleaned up afterward.
- The output directory stores one Markdown report per video plus a `batch_summary.json` file.

## Example prompt behavior

The model is prompted to:

- Identify the exercise being performed
- Assess form stability, posture, and range of motion
- Highlight likely issues or improvement areas

This folder is intended as a practical starting point for local multimodal experimentation with AI-driven video analysis.
