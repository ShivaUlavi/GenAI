WorkoutAnalyzerAppV2
Edge-Deployed Multimodal Workout Analytics & Grounded Biomechanical Coaching on 4 GB VRAM
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c)
![Ultralytics](https://img.shields.io/badge/YOLO11-Perception-00b894)
![Ollama](https://img.shields.io/badge/Ollama-Local_VLM-111111)
![Streamlit](https://img.shields.io/badge/Streamlit-Edge_UI-ff4b4b)
![Hardware](https://img.shields.io/badge/Edge_Target-GTX_1050_Ti_(4GB_VRAM)-76b900)
WorkoutAnalyzerAppV2 is a fully local, privacy-preserving computer vision and multimodal coaching pipeline engineered to run end-to-end on a consumer NVIDIA GTX 1050 Ti (4 GB VRAM, 8 GB system RAM). Instead of relying on cloud APIs or ungrounded end-to-end Vision-Language Model (VLM) prompting, it fuses deterministic dual-YOLO perception, a lightweight temporal classification ensemble, camera-aware kinematic signal processing, and a confidence-gated local VLM (`bonsai-2:4b`) inside a strict sequential VRAM orchestrator.
> **Demo Media:** *Add your side-by-side Streamlit dashboard screenshot and `dual_yolo_overlay.mp4` demo clip here once uploaded to GitHub.*
---
Key Highlights
Zero-OOM 4 GB VRAM Orchestration: Sequential Stage 1 $\rightarrow$ explicit CUDA barrier flush $\rightarrow$ Stage 2 (`keep_alive=0`) execution holds peak PyTorch allocation to $369.6\text{--}371.1\text{ MB}$ and releases GPU memory to $789.9 / 4095.9\text{ MB}$ before and after local VLM inference.
Sub-45s End-to-End Local Turnaround: Complete perception, temporal scoring, kinematic rep counting, single-pass H.264 dual-YOLO overlay export, and grounded coaching critique finish in $30.65\text{--}45.40\text{ s}$ per clip on a GTX 1050 Ti.
Camera-Invariant Kinematics & Adaptive Rep Counting: Combines bilateral `$p_2/p_{98}$` joint ROM tracking, front-view vertical depth fallback (`wrist_y / ankle_y`) with 2D foreshortening alerts, low-confidence max-ROM joint routing, adaptive peak prominence, and a single-rep boundary safeguard.
Neuro-Symbolic Confidence-Gated VLM Routing: Moves classification trust decisions out of brittle prompt prose and into deterministic Python logic—separating Locked High-Confidence refinement ($p_1 \ge 0.65$, margin $\ge 0.25$) from Open Visual Verification ($p_1 < 0.65$ or narrow margin) with noise runner-up pruning and blind filename scrubbing.
Single-Pass Interactive Streamlit Dashboard (`app.py`): Streams live `YOLO11s_v3.1` equipment boxes + `YOLO11n-pose` skeletons frame-by-frame, exports browser-playable `libx264` MP4 overlays, and renders interactive Top-2 ensemble charts, ROM tables, grounding keyframes, and live GPU telemetry.
---
System Architecture
Co-loading dual object/pose detectors, temporal classifiers, and a 3B–4B multimodal VLM simultaneously causes immediate out-of-memory errors on a 4 GB GPU. `WorkoutAnalyzerAppV2` partitions inference into two strictly non-overlapping stages separated by an explicit CUDA memory flush:
```text
Input Workout Clip (.mp4 / .mov / .avi)
  |
  v
[Stage 1A: Dual-YOLO Perception]  (10 FPS, imgsz=640, Peak ~371 MB PyTorch VRAM)
  |-- YOLO11s_v3.1 (19.2 MB)  --> Custom gym equipment boxes + wrist proximity
  |-- YOLO11n-pose            --> Primary-person COCO-17 skeleton
  |-- Single-pass compositor  --> Live UI stream + disk-streamed H.264 MP4 overlay
  '-- Output: [T, 54] normalized frame-feature sequence
  |
  v  [CUDA Flush Barrier: del models + gc.collect() + synchronize() + empty_cache() + ipc_collect()]
  |
[Stage 1B: Phase 2 Soft-Voting Ensemble]
  |-- TCN_96_c64 (0.34 MB, GPU)          --> 96-dim pose + velocity windows (win=32, stride=8)
  '-- RandomForest_ClipStats (3.88 MB)   --> Clip-level posture, equipment & motion energy stats
      Blend presets: 0.3 TCN / 0.7 RF (82.5% OOF Top-2 Prior) | 0.7 TCN / 0.3 RF (68.3% OOF Top-1)
  |
  v
[Stage 1C/1D: Deterministic Kinematics, Keyframes & Pre-VLM Flush]
  |-- Bilateral ROM (p2/p98), front-view depth fallback, adaptive prominence rep counting
  |-- 3 spaced grounding keyframes (setup, mid-clip peak inflection, finish @ 512px JPEG)
  '-- Pre-VLM CUDA flush verification (~3.3 GB VRAM free)
  |
  v
[Stage 2: Confidence-Gated Grounded Local VLM]  (Ollama: bonsai-2:4b, temp=0.0, keep_alive=0)
  |-- Locked Route (p1 >= 0.65 & margin >= 0.25): Runner-up stripped; visual variant + form critique
  '-- Open Route   (uncertain / hybrid / OOD):    Pruned priors (p >= 0.15); visual override enabled
  |
  v
Structured 4-Section Grounded Coaching Report + Full JSON Telemetry + Annotated MP4
```
---
Edge Hardware Telemetry (GTX 1050 Ti)
Measured across unseen 9.2–20.5 s gym clips on NVIDIA GeForce GTX 1050 Ti (4095.9 MB VRAM) and 8 GB host RAM:
Pipeline Stage	Artifacts / Configuration	Steady-State Latency	Peak PyTorch VRAM	Post-Stage Device VRAM
Stage 1A: Perception	`YOLO11s_v3.1` (19.2 MB) + `YOLO11n-pose`, 10 FPS, `imgsz=640`	$10.35\text{--}18.19\text{ s}$*	$369.6\text{--}371.1\text{ MB}$	$789.9\text{--}793.9 / 4095.9\text{ MB}$
Stage 1B: Ensemble	`TCN_96_c64` (0.34 MB) + `RF_ClipStats` (3.88 MB)	$0.40\text{--}2.86\text{ s}$	$32.0\text{ MB}$ alloc.	$789.9 / 4095.9\text{ MB}$
Stage 1C/1D: Kinematics	ROM, rep counter, 3 JPEGs, final CUDA barrier	$< 0.60\text{ s}$	$32.0\text{ MB}$ alloc.	$789.9 / 4095.9\text{ MB}$ ($\sim 3.3\text{ GB}$ free)
Stage 2: Primary VLM	`bonsai-2:4b` (`reasoning=False`, `temp=0.0`, `keep_alive=0`)	$17.84\text{--}30.16\text{ s}$	Isolated (Ollama)	$789.9 / 4095.9\text{ MB}$ (evicted)
Stage 2: Comparator	`qwen2.5vl:3b` (4-bit GGUF, `keep_alive=0`)	$34.29\text{--}96.49\text{ s}$	Isolated (Ollama)	$789.9 / 4095.9\text{ MB}$ (evicted)
*20.45 s clip (`hacksqaut_unseen.mp4`) ran Stage 1A in $27.78\text{ s}$. Streaming H.264 frames directly to disk via `imageio-ffmpeg` prevents host-RAM pressure on 8 GB systems.
---
Key Engineering & Empirical Findings
Real-world gym testing uncovered several edge-deployment failure modes that directly shaped the final codebase:
Front-View 2D Foreshortening (`kinematics.py`): Straight-on monocular views compress sagittal hip hinges along the camera Z-axis (e.g., $155^\circ\text{--}166^\circ$ apparent hip angle on a full-ROM deadlift). Automatically engaging a normalized wrist-descent fallback (`vertical_depth_fallback`) when 2D ROM $< 20^\circ$ and passing an explicit `camera_projection_note` fixed both rep counting ($0 \rightarrow 2/2$) and VLM depth hallucinations.
Adaptive Prominence & Single-Rep Boundary Safeguard (`kinematics.py`): Scaling `find_peaks` prominence to $\max(\text{base}, 0.30 \times \text{ROM})$ eliminated cropped-pose jitter over-counts on seated curls ($9 \rightarrow 5/5$), while a boundary safeguard ($\text{ROM} \ge 50^\circ$ when `len(peaks) == 0`) preserved single-rep floor-to-floor lifts ($1/1$).
Thinking-Loop Suppression (`vlm_grounding.py`): By default in Ollama, `bonsai-2:4b` entered hidden reasoning loops on structured JSON prompts, timing out with empty strings after $68.9\text{--}72.0\text{ s}$. Prefix-gated `reasoning=False` with `num_ctx=4096` and `num_predict=512` achieved a 100% completion rate and cut Stage 2 latency by $\sim 60%$.
Zero-Noun Prompts Against 4B Attention Anchoring (`vlm_grounding.py`): Placing any concrete exercise or posture noun inside `SYSTEM_PROMPT`—even inside parenthetical examples—hijacked 4B attention (e.g., turning an $88.1%$ locked floor `core` twist into a standing deadlift, or a standing goblet squat into a floor-seated cable row). Purging all example nouns and pruning noise runner-ups ($p_2 < 0.15$) eliminated these hallucinations.
Blind Ablation & Filename Leakage Audit: Passing `video_file` inside the telemetry JSON created an open-book shortcut that let `qwen2.5vl:3b` lexically expand filenames (e.g., `"barbell_landmine_rtwist.mp4"` $\rightarrow$ "Landmine Rotational Twist"). Permanently stripping `video_file` prior to VLM serialization established a true blind evaluation.
---
Controlled VLM Backbone Ablation (`bonsai-2:4b` vs. `qwen2.5vl:3b`)
Evaluated on identical frozen Stage 1 telemetry on GTX 1050 Ti (4 GB VRAM, 8 GB RAM):
Test Clip (Route)	`qwen2.5vl:3b` (Leaked Filename)	`qwen2.5vl:3b` (Blind Control)	`bonsai-2:4b` (Sanitized)	Stage 2 Latency (Bonsai vs. Qwen)
`dumbbell_goblet_squat.mp4` (Open)	Goblet Squat (raw JSON, generic)	—	Goblet Squat w/ Dumbbell (noted belt, $74.3^\circ$ knee bend)	$25.72\text{ s}$ vs. $96.49\text{ s}$ ($3.75\times$ faster)
`dumbbell_Pullover1.mp4` (OOD Open)	Dumbbell Pullover (leaked) — truncated	Bench Press — truncated mid-word	Incline Dumbbell Press (complete 4 sections)	$23.37\text{ s}$ vs. $63.36\text{ s}$ ($2.71\times$ faster)
`barbell_landmine_rtwist.mp4` (Locked `core`)	Landmine Twist (leaked, raw JSON)	Core exercise (misread ROM as static hold)	Core (complete 4 sections, neutral-spine cue)	$17.84\text{ s}$ vs. $34.29\text{ s}$ ($1.92\times$ faster)
`hacksqaut_unseen.mp4` (Open, 9:16)	—	Deadlift — truncated at Section 2	0 truncations across all Bonsai runs	$67.33\text{ s}$ (Qwen)
> **Why `qwen2.5vl:3b` truncated on portrait clips:** `qwen2.5vl` uses dynamic $28 \times 28$ vision patches proportional to image area. Three vertical 9:16 keyframes scaled by `max_width=512` ($\sim 512 \times 910\text{ px}$ each) expanded visual tokens and KV cache beyond the $\sim 3.3\text{ GB}$ free VRAM window, spilling layers onto 8 GB host RAM. `bonsai-2:4b` remained VRAM-resident with **0 truncations** and **$1.92\times\text{--}3.75\times$ lower latency**, making it the locked default.
---
9-Clip Unseen & Out-of-Distribution Validation Matrix
Clip & Stress Condition	Stage 1 Top-1 (Prob)	Route	Reps (Est / GT)	Final VLM Verdict (`bonsai-2:4b` default)	Total Latency
`deadlift_unseen1.mp4` (Front view, Z-foreshortening)	`deadlift` ($86.9%$)	Locked	2 / 2 (`depth`)	Deadlift w/ barbell & green plates	$100.6\text{ s}$*
`deadlift_unseen4.mp4` (Single-rep floor boundary)	`deadlift` ($66.6%$)	Locked	1 / 1 (`hip`)	Deadlift w/ barbell & weight plates	$39.3\text{ s}$
`dumbbell_goblet_squat.mp4` (Occluded DB, bg lifter, cable rack)	`lunge` ($54.2%$)	Open	5 / 5 (`knee`)	Goblet Squat w/ Dumbbell (visual override + belt)	$43.6\text{ s}$
`dumbbell_curl_unseen1.mp4` (Severe close-up crop)	`core` ($21.6%$) / `curl` #2	Open	5 / 5 (`elbow`)	Seated Dumbbell Curl (recovered from Top-2)	$43.3\text{ s}$
`barbell_waliking_lunges.mp4` (Rack bg, flip risk)	`lunge` ($68.7%$)	Locked	2 (`knee`)	Barbell Walking Lunge (exact sub-variant)	$44.5\text{ s}$
`step.mp4` (Bodyweight, unmodeled plyo box)	`lunge` ($79.7%$)	Locked	7 (`knee`)	Step-Up (Lunge variant) w/ black metal box	$39.0\text{ s}$
`barbell_landmine_rtwist.mp4` (Floor V-sit rotation)	`core` ($88.1%$)	Locked	4 (`hip`)	Core (locked; neutral-spine cue)	$30.7\text{ s}$
`dumbbell_Pullover1.mp4` (OOD `zeroshot_holdout`)	`core` ($56.6%$) / `press` #2	Open	4 (`hip`)	Incline Dumbbell Press (documented 3-frame OOD limit)	$42.7\text{ s}$
`hacksqaut_unseen.mp4` (Behind-the-back hack squat, 20.5 s)	`deadlift` ($48.2%$) / `squat` ($40.0%$)	Open	3 (`hip`)	Qwen blind stress test: Stage 1 hybrid split validated	$96.3\text{ s}$†
*Pre-`reasoning=False` baseline run. †Evaluated blind with `qwen2.5vl:3b` (Stage 1A = $27.78\text{ s}$, Stage 2 = $67.33\text{ s}$).
---
Repository Structure
```text
WorkoutAnalyzerAppV2/
|-- app.py                                   # Streamlit edge UI (side-by-side live dual-YOLO + telemetry)
|-- models/
|   |-- yolo11s_v3_1_640_best.pt             # Phase 1 custom gym equipment detector (19.2 MB)
|   |-- phase2_tcn96_c64_best.pt             # Phase 2 temporal convolutional network (0.34 MB)
|   '-- phase2_rf_clipstats.joblib           # Phase 2 clip-level Random Forest bundle (3.88 MB)
|-- src/
|   |-- phase2/
|   |   |-- phase2_features.py               # Frame-level [54]-dim pose + equipment feature extraction
|   |   '-- phase2_models.py                 # TemporalConvNet architecture definition
|   '-- phase3/
|       |-- kinematics.py                    # Velocity math, ROM, adaptive rep counting & 3 keyframes
|       |-- vlm_grounding.py                 # Filename scrubbing, Locked/Open router & Ollama client
|       '-- phase3_pipeline.py               # Sequential 4 GB VRAM orchestrator, H.264 writer & CLI
'-- README.md
```
(Ultralytics automatically downloads `yolo11n-pose.pt` on first run if not already cached locally.)
---
Quickstart (Local Edge Setup)
1. Prerequisites
Python 3.10+ with CUDA-enabled PyTorch (tested on GTX 1050 Ti, 4 GB VRAM).
Ollama installed and running locally.
2. Pull the Local VLM
```bash
ollama pull bonsai-2:4b
# Optional ablation comparator:
# ollama pull qwen2.5vl:3b
```
3. Install Python Dependencies
```bash
pip install torch torchvision ultralytics opencv-python numpy scipy pandas scikit-learn joblib streamlit langchain-core langchain-ollama imageio imageio-ffmpeg
```
4. Launch the Streamlit Edge App
```bash
streamlit run app.py
```
Upload any `.mp4`, `.mov`, or `.avi` clip, choose your Phase 2 ensemble operating point (`0.3 TCN / 0.7 RF` default) and local Ollama tag in the sidebar, and click Analyze Workout Clip to stream live dual-YOLO detections and download both the annotated H.264 video and full JSON report.
5. Run Headless via CLI
```bash
python src/phase3/phase3_pipeline.py \
  --video "path/to/workout_clip.mp4" \
  --model bonsai-2:4b \
  --save-video output_dual_yolo.mp4 \
  --out-json workout_report.json
```
Pass `--skip-vlm` to benchmark Stage 1 perception, ensemble scoring, and kinematics in isolation.
---
Known Limitations & Future Research Directions
Monocular 2D Foreshortening: Straight-on camera angles flatten sagittal flexion; adding a lightweight (< 5 MB) 2D-to-3D temporal pose lifter in Stage 1 would recover view-invariant joint kinematics.
Static 3-Keyframe OOD Boundary: Unseen lifts whose geometry differs primarily by continuous arc trajectory (e.g., supine dumbbell pullover vs. bench press) remain ambiguous from three static frames; compact trajectory overlays or motion tokens are a natural next step.
Dynamic-Patch Portrait Scaling: Bounding the longest image edge ($\max(h, w) \le 512$) rather than `max_width` inside `extract_keyframes_b64()` will mitigate visual-token expansion for variable-resolution ViT encoders on 9:16 mobile video.
Conformal Uncertainty & Grammar-Constrained Decoding: Replacing fixed confidence-gate thresholds ($p_1 \ge 0.65$, margin $\ge 0.25$) with calibrated conformal sets and constrained numeric copying can further harden compact edge VLMs against lexical or angle-scale drift.
---
Author
Shivachetan Ulavi  
Built for edge computer vision research, biomechanical analytics, and resource-constrained multimodal AI deployment.
