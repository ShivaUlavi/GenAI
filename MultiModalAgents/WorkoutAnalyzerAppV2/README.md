# 🏋️ WorkoutAnalyzerApp V2 — Grounded Edge Computer Vision & Biomechanical Coaching

A fully local, **4 GB VRAM edge-deployed** workout analysis pipeline running end-to-end on an **NVIDIA GTX 1050 Ti (4 GB VRAM)** and **8 GB system RAM**.

`WorkoutAnalyzerAppV2` combines deterministic dual-YOLO perception, a lightweight temporal + tabular movement ensemble (`TCN-96 + Random Forest`), rule-based 2D biomechanical kinematics, and a **neuro-symbolic confidence-gated local VLM (`bonsai-2:4b` via Ollama)**—delivered through an interactive Streamlit dashboard with live Dual-YOLO overlays and downloadable H.264 annotated video.

---

## Architecture Overview

To avoid out-of-memory (OOM) crashes on a 4 GB consumer GPU, the pipeline enforces **strict sequential execution** with an explicit CUDA memory flush between perception and multimodal reasoning:

```text
Input Workout Clip (.mp4 / .mov / .avi)
  │
  ├──► Stage 1A: Dual-YOLO Perception (~10 FPS, imgsz=640)
  │      ├── YOLO11n-pose (COCO-17 skeleton keypoints)
  │      ├── YOLO11s_v3.1 (custom gym equipment: barbell, dumbbell, weightplate, trap_bar)
  │      └── Single-pass Dual-Overlay H.264 MP4 encoder (imageio-ffmpeg) + live UI stream
  │
  ├──► Stage 1B: Phase 2 Soft-Voting Ensemble (12 Movement Families)
  │      ├── TCN-96 (0.34 MB PyTorch checkpoint, 96-dim pose+velocity windows)
  │      └── RandomForest_ClipStats (3.88 MB joblib, clip-level kinematic statistics)
  │      └── Default blend: 0.30 TCN + 0.70 RF (82.5% OOF Top-2 VLM prior)
  │
  ├──► Stage 1C: Deterministic Biomechanics & Keyframe Extraction (kinematics.py)
  │      ├── Bilateral joint ROM (p2/p98), adaptive peak prominence & single-rep safeguard
  │      ├── Front-view foreshortening fallback (vertical wrist/ankle depth)
  │      ├── Low-confidence max-ROM joint gate (top1_prob < 0.45)
  │      └── 3 well-separated keyframes (setup, mid-clip peak inflection, finish)
  │
  ├──► Stage 1D: Hard VRAM Flush
  │      └── del models → gc.collect() → torch.cuda.synchronize() / empty_cache() / ipc_collect()
  │          (Peak PyTorch VRAM: ~370.7 MB → Post-flush allocated: 32.0 MB | Device: ~790 / 4096 MB)
  │
  └──► Stage 2: Neuro-Symbolic Grounded Local VLM (vlm_grounding.py via Ollama)
         ├── Confidence-Gated Routing:
         │     • LOCKED (p1 >= 0.65 & margin >= 0.25): strips runner-up distractor, refines variant/form
         │     • OPEN   (p1 <  0.65 or margin <  0.25): prunes noise runner-ups (< 0.15), allows visual override
         ├── Blind & Sanitized Grounding: strips `video_file` (no filename leakage), zero prompt examples
         └── Local Inference: bonsai-2:4b (reasoning=False, temperature=0.0, num_ctx=4096, keep_alive=0)
```

---

## Edge Hardware Telemetry (GTX 1050 Ti — 4 GB VRAM / 8 GB RAM)

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Peak PyTorch VRAM (Stage 1)** | **369.6 – 371.1 MB** | Dual-YOLO + TCN-96 sequential peak |
| **Allocated VRAM Post-Flush** | **32.0 MB** | Verified before Ollama loads Stage 2 VLM |
| **Total Device VRAM Post-Flush** | **789.9 – 793.9 / 4095.9 MB** | Leaves ~3.3 GB free VRAM window for local VLM |
| **Stage 1A Perception Latency** | **10.4 – 17.4 s** | 9–15 s clips (includes Dual-YOLO + H.264 overlay) |
| **Stage 1B Ensemble Latency** | **0.4 – 2.6 s** | TCN-96 sliding windows + Random Forest inference |
| **Stage 2 VLM Latency (`bonsai-2:4b`)** | **17.8 – 30.2 s** | With `reasoning=False`, `temperature=0.0`, `keep_alive=0` |
| **End-to-End Runtime** | **30.7 – 44.5 s** | Typical 10–15 s unseen gym clip |

---

## Key Engineering & Research Findings

Iterative stress testing on unseen self-recorded gym videos uncovered several failure modes common to compact edge VLMs and monocular 2D kinematics, all resolved systematically:

1. **Disabling Hidden Reasoning Loops (`bonsai-2:4b`):** Without `reasoning=False`, `bonsai-2:4b` exhausted its output budget on hidden thinking tokens (~69–72 s) and returned empty responses (`""`). Disabling thinking plus setting `num_ctx=4096` and `num_predict=512` restored 100% completion and cut Stage 2 latency by **~2.5×–3.5×**.
2. **Confidence-Gated Neuro-Symbolic Routing:** Replacing soft natural-language prompt thresholds with deterministic Python routing (`Locked` at `p1 >= 0.65` and `margin >= 0.25` vs. `Open`) and greedy decoding (`temperature=0.0`) eliminated stochastic label flips (e.g., `barbell_waliking_lunges.mp4` flipping from Lunge to Front Squat between runs).
3. **In-Context Example Anchoring in 4B Models:** Concrete exercise or posture examples inside the system prompt—even inside negative rules or `(e.g., ...)` lists—acted as attention magnets for the 4B VLM, hijacking locked clips into unrelated exercises. Removing all concrete nouns from prompts and pruning noise runner-ups (`p2 < 0.15`) resolved these hallucinations.
4. **Prompt-Level Filename Leakage Discovery:** During VLM backbone ablations, passing `video_file` inside the JSON payload allowed `qwen2.5vl:3b` to read exercise labels directly from filenames (e.g., `"dumbbell_Pullover1.mp4"` and `"barbell_landmine_rtwist.mp4"`). Permanently stripping `video_file` before VLM invocation enforced a true blind evaluation.
5. **Monocular Kinematics Hardening (`kinematics.py`):**
   - **Front-View Foreshortening:** Added `p2/p98` percentile ROM, vertical depth fallback (`wrist_y / ankle_y`), and `camera_projection_note` to handle sagittal compression on straight-on deadlifts.
   - **Adaptive Prominence & Boundary Safeguard:** Scaled peak prominence (`max(15°, 0.30 * ROM)`) to prevent cropped-pose jitter over-counting (`9 → 5` reps on concentration curls) paired with a `ROM >= 50°` single-rep safeguard (`0 → 1` rep on 1RM deadlifts).
   - **Spaced Keyframes:** Mid-clip peak selection and minimum frame-gap enforcement prevented bunched keyframes from collapsing temporal context.

---

## Blind VLM Backbone Ablation (`bonsai-2:4b` vs. `qwen2.5vl:3b`)

Controlled head-to-head comparison on the exact same frozen Stage 1 telemetry on a GTX 1050 Ti (4 GB VRAM / 8 GB RAM):

| Clip | `qwen2.5vl:3b` (Leaked Filename) | `qwen2.5vl:3b` (Blind) | `bonsai-2:4b` (Sanitized) | Stage 2 Latency (`Bonsai` vs. `Qwen`) |
| :--- | :--- | :--- | :--- | :--- |
| **`dumbbell_goblet_squat.mp4`** *(Open)* | Goblet Squat (raw JSON, generic prose) | — | **Goblet Squat w/ Dumbbell** (noted belt, knee angle `74.3°`, upright torso) | **25.72 s** vs. 96.49 s (**3.75× faster**) |
| **`dumbbell_Pullover1.mp4`** *(OOD, Open)* | Dumbbell Pullover *(leaked)* — **truncated** | Bench Press — **truncated** at `"- **Re"` | Incline Dumbbell Press (complete 4 sections, clean joint mapping) | **23.37 s** vs. 63.36 s (**2.71× faster**) |
| **`barbell_landmine_rtwist.mp4`** *(Locked `core`)* | Landmine Rotational Twist *(leaked, raw JSON)* | Core exercise (confused `72.0°` ROM for static hold) | **Core** (complete 4 sections, neutral-spine cue) | **17.84 s** vs. 34.29 s (**1.92× faster**) |
| **`hacksqaut_unseen.mp4`** *(Hybrid, Open)* | — | Deadlift — **truncated** at `"- **Reps:"` | *(Bonsai locked as production default)* | Qwen: **67.33 s** (96.29 s total, mid-sentence cutoff) |

**Ablation Takeaway:** `qwen2.5vl:3b` uses dynamic `28×28` patch tokenization, causing tall 9:16 portrait keyframes to generate high visual token counts that spill KV cache / layers beyond the ~3.3 GB post-flush VRAM window into system RAM—leading to **1.9×–3.8× higher latency** and frequent premature truncation. **`bonsai-2:4b`** stayed VRAM-resident with **0% truncation** and richer biomechanical coaching prose.

---

## 9-Clip Unseen Stress-Test Validation Matrix

| Clip (Ground Truth) | Stage 1 Top-1 / Top-2 | Routing Mode | Reps (Est / GT) | Final Grounded Output (`bonsai-2:4b`) | Runtime |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **`deadlift_unseen1.mp4`** *(Front-view deadlift)* | `deadlift` (87%) | Locked | **2 / 2** | Deadlift w/ Barbell *(vertical depth fallback)* | ~38 s |
| **`deadlift_unseen4.mp4`** *(Front-view 1RM deadlift)* | `deadlift` (66.6%) / `squat` (17.4%) | Locked | **1 / 1** | **Deadlift w/ Barbell & Plates** | 39.3 s |
| **`dumbbell_goblet_squat.mp4`** *(3/4 view, moving bg lifter)* | `lunge` (54.2%) / `row` (10.0% pruned) | Open | **5 / 5** | **Goblet Squat w/ Dumbbell** *(recovered occluded DB & belt)* | 43.6 s |
| **`dumbbell_curl_unseen1.mp4`** *(Tight crop, seated conc. curl)* | `core` (21.6%) / `curl` (16.4%) | Open | **5 / 5** | **Dumbbell Curl** *(elbow gate + adaptive prominence)* | 43.3 s |
| **`barbell_waliking_lunges.mp4`** *(Walking lunges)* | `lunge` (68.7%) / `squat` (15.7%) | Locked | **2** | **Barbell Walking Lunge** | 44.5 s |
| **`step.mp4`** *(Box step-ups)* | `lunge` (79.7%) / `core` (3.7%) | Locked | **7** | **Step-Up (Lunge variant) w/ Box** *(visually detected box)* | 39.0 s |
| **`barbell_landmine_rtwist.mp4`** *(Floor landmine twist)* | `core` (88.1%) / `dip` (3.2%) | Locked | **4** | **Core** *(zero standing-lift hallucination)* | 30.7 s |
| **`dumbbell_Pullover1.mp4`** *(OOD `zeroshot_holdout`)* | `core` (56.6%) / `press` (21.3%) | Open | **4** | *Incline Dumbbell Press* *(documented 3-frame OOD boundary)* | 42.7 s |
| **`hacksqaut_unseen.mp4`** *(Behind-the-back barbell hack squat)* | `deadlift` (48.2%) / `squat` (40.0%) | Open | **3** | *Hybrid deadlift/squat ensemble disagreement + Qwen truncation baseline* | 96.3 s *(Qwen)* |

---

## Repository Structure

```text
WorkoutAnalyzerAppV2/
├── app.py                              # Interactive Streamlit UI (side-by-side live Dual-YOLO + dashboard)
├── models/
│   ├── yolo11s_v3_1_640_best.pt        # Custom gym equipment detector (~19.2 MB)
│   ├── phase2_tcn96_c64_best.pt        # Temporal ConvNet classifier (0.34 MB)
│   └── phase2_rf_clipstats.joblib      # Clip-level Random Forest bundle (3.88 MB)
└── src/
    ├── phase2/
    │   ├── phase2_features.py          # Frame-level 54-dim pose + equipment feature extractor
    │   └── phase2_models.py            # TemporalConvNet architecture
    └── phase3/
        ├── kinematics.py               # Joint ROM, adaptive rep counting, depth fallback & keyframe extraction
        ├── vlm_grounding.py            # Confidence-gated router, filename blinding & Ollama VLM interface
        └── phase3_pipeline.py          # Sequential GTX 1050 Ti orchestrator, VRAM flushing & H.264 writer
```

---

## Quick Start (Local Setup)

### 1. Prerequisites
- Python 3.10+ with CUDA-enabled PyTorch (tested on GTX 1050 Ti / 4 GB VRAM)
- [Ollama](https://ollama.com/) installed and running locally

Pull the default local VLM (and optional ablation baseline):
```bash
ollama pull bonsai-2:4b
# Optional ablation model:
ollama pull qwen2.5vl:3b
```

### 2. Install Python Dependencies
```bash
pip install torch torchvision ultralytics opencv-python scipy scikit-learn joblib pandas streamlit langchain-core langchain-ollama imageio imageio-ffmpeg
```
*(Ultralytics automatically fetches `yolo11n-pose.pt` on first run if not already cached locally.)*

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Upload any gym workout video to watch the **live Dual-YOLO equipment + skeleton stream** on the right panel, inspect the 3 grounding keyframes, review ensemble probabilities and joint ROM telemetry, and download both the **H.264 annotated MP4** and **full JSON report**.

### 4. Run via CLI (Batch / Headless Mode)
```bash
python src/phase3/phase3_pipeline.py \
  --video "path/to/workout_clip.mp4" \
  --model bonsai-2:4b \
  --w-tcn 0.3 \
  --w-rf 0.7 \
  --save-video annotated_output.mp4 \
  --out-json workout_report.json
```
Add `--skip-vlm` to benchmark Stage 1 deterministic perception, ensemble scoring, and kinematics independently.

---

## Limitations & Future Work

- **Static-Frame OOD Boundary:** Three keyframes can conflate supine lifts with similar start/end poses (e.g., dumbbell pullover vs. dumbbell press). Feeding compact kinematic trajectory summaries or higher-density token-merged frames is a natural next step.
- **Monocular 2D Occlusion:** Tightly cupped equipment (e.g., goblet dumbbells) and severe frame cropping reduce YOLO detection and pose confidence; multi-view or lightweight 3D pose lifting (e.g., VideoPose3D / MotionBERT) would remove sagittal projection ambiguity.
- **Closed-Set 12-Class Taxonomy:** Expanding Phase 2 training coverage across cross-equipment variants (e.g., goblet/hack squats alongside barbell back squats) will further strengthen Stage 1 calibration.
