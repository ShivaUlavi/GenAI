# src/phase3/vlm_grounding.py
import copy
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

COMMON_RULES = """- Describe only what is directly visible on the athlete in the keyframes and measured in the JSON. Ignore background gym machines not in contact with the athlete.
- Angle convention: 180° = straight joint; smaller angle = deeper bend. Keep each joint's numbers under its own name.
- Respect `camera_projection_note` when judging depth.
- Output ONLY these 4 short sections in plain Markdown prose (do NOT output JSON or code blocks):
  1. Confirmed Exercise & Equipment
  2. Set Summary (Reps & Range of Motion)
  3. Form & Biomechanical Observations
  4. One Actionable Coaching Cue
"""

SYSTEM_PROMPT_LOCKED = (
    "You are a strength coach. `confirmed_movement_family` is fixed ground truth—stay strictly within that family and use the keyframes only for the specific variation, equipment, and form cues.\n"
    + COMMON_RULES
)

SYSTEM_PROMPT_OPEN = (
    "You are a strength coach. Ensemble confidence is low—verify the candidate family against the athlete's posture and held object in the keyframes, and override it if the frames clearly show a different exercise.\n"
    + COMMON_RULES
)

THINKING_MODEL_PREFIXES = ("bonsai", "qwen3", "deepseek-r1", "gpt-oss")


def _prepare_vlm_payload(
    prior_payload: dict,
    lock_thresh: float = 0.65,
    margin_thresh: float = 0.25,
    min_runnerup_prob: float = 0.15,
):
    """Deterministically route between locked refinement and open visual override."""
    vlm_payload = copy.deepcopy(prior_payload)
    # Prevent filename label leakage (e.g., 'dumbbell_Pullover1.mp4') into the VLM
    vlm_payload.pop("video_file", None)
    priors = vlm_payload.get("activity_priors", {})
    top2 = priors.get("top2_candidates", [])

    p1 = float(top2[0].get("blend_prob", 0.0)) if len(top2) >= 1 else 0.0
    p2 = float(top2[1].get("blend_prob", 0.0)) if len(top2) >= 2 else 0.0
    margin = p1 - p2

    if p1 >= lock_thresh and margin >= margin_thresh:
        vlm_payload["activity_priors"] = {
            "routing_mode": "locked_high_confidence",
            "confirmed_movement_family": priors.get("top1_label"),
            "confidence": round(p1, 4),
        }
        return vlm_payload, SYSTEM_PROMPT_LOCKED

    # Prune noise runner-ups (e.g., 9.9% 'row') so they can't trigger background hallucinations
    filtered = [c for c in top2 if float(c.get("blend_prob", 0.0)) >= min_runnerup_prob]
    vlm_payload["activity_priors"] = {
        "routing_mode": "open_visual_verification",
        "candidates": filtered or top2[:1],
    }
    return vlm_payload, SYSTEM_PROMPT_OPEN


def _build_llm(model_name: str, temperature: float, disable_reasoning: bool) -> ChatOllama:
    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "num_ctx": 4096,
        "num_predict": 512,
        "keep_alive": 0,  # Flush VRAM immediately after generation
    }
    if disable_reasoning:
        kwargs["reasoning"] = False
    return ChatOllama(**kwargs)


def run_grounded_vlm_critique(
    prior_payload: dict,
    keyframes_b64: dict[str, str],
    model_name: str = "bonsai-2:4b",
    temperature: float = 0.0,
    use_images: bool = True,
) -> str:
    """Invoke local Ollama VLM with greedy decoding, confidence-gated routing, and keep_alive=0."""
    vlm_payload, system_prompt = _prepare_vlm_payload(prior_payload)

    user_text = (
        "Analyze this workout clip using the telemetry JSON and attached keyframes.\n\n"
        f"TELEMETRY_JSON:\n```json\n{json.dumps(vlm_payload, indent=2)}\n```"
    )

    content = [{"type": "text", "text": user_text}]
    if use_images:
        for _, b64_img in keyframes_b64.items():
            content.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{b64_img}",
            })

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content),
    ]

    wants_nothink = model_name.lower().startswith(THINKING_MODEL_PREFIXES)
    llm = _build_llm(model_name, temperature, disable_reasoning=wants_nothink)

    try:
        response = llm.invoke(messages)
    except Exception as exc:
        if wants_nothink and "does not support thinking" in str(exc).lower():
            response = _build_llm(model_name, temperature, disable_reasoning=False).invoke(messages)
        else:
            raise

    text = (response.content or "").strip()
    if not text:
        fallback = response.additional_kwargs.get("reasoning_content", "")
        text = fallback.strip() if fallback else "[VLM returned empty output — check `ollama --version` / model tag]"
    return text