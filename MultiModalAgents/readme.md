# MultiModalAgents

This directory presents a focused collection of multimodal AI workflows designed for visual understanding, video analysis, and exercise assessment. The projects collectively examine how language models and vision-capable systems can be integrated to interpret workout footage, summarize movement quality, and generate structured feedback from visual evidence.

## Included components

- `01_sequential_multimodal_narrative_generator.ipynb` — a sequential multimodal pipeline for generating narrative descriptions from visual inputs and task-specific prompts.
- `02_asynchronous_multimodal_portfolio_evaluator.ipynb` — an asynchronous evaluation workflow for assessing multimodal outputs in a portfolio-style analytical setting.
- `03_Local_Edge_Gym_Form_Analyzer_Ollama.ipynb` — a local edge-oriented gym form analysis workflow built around Ollama, emphasizing privacy-preserving and resource-efficient deployment.
- `04_Batch_Video_Analysis_ThreadPool.py` — a scalable batch-processing script that analyzes multiple videos in parallel by extracting frames and evaluating them with a multimodal model.
- `WorkoutAnalyzerApp/` — a Streamlit-based application that allows users to upload a workout video and receive automated feedback on technique and movement quality. Additional implementation details are provided in the app-specific README within this folder.

## Research and application focus

The workflows in this folder are oriented toward several core use cases:

- Exercise recognition and classification
- Gym form assessment and biomechanical review
- Posture, range-of-motion, and movement quality analysis
- Multimodal reasoning over sequential visual frames
- Local deployment of vision-language pipelines for practical experimentation

## Typical workflow

1. Load one or more workout videos from a local directory.
2. Extract representative frames or keyframes from each video.
3. Convert visual data into a format suitable for multimodal inference.
4. Invoke a vision-capable model to reason over the extracted frames.
5. Generate structured assessments, recommendations, and summaries for each video or batch.

## Technical requirements

These examples are designed for local execution and typically depend on:

- Python
- OpenCV
- LangChain
- Ollama
- A vision-capable local model compatible with multimodal inference

## Notes

The scripts in this directory are intended as practical research and prototyping tools for multimodal video analytics in fitness and exercise contexts. They demonstrate how local AI systems can support detailed visual interpretation without requiring a fully cloud-based deployment pipeline.

The `WorkoutAnalyzerApp` extends this work into a user-facing interface, enabling interactive video upload and feedback generation in a streamlined application environment.
