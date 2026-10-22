# Project Findings

## Background

This project started with a practical goal: to test whether a local multimodal LLM could analyze real gym footage and provide useful form feedback without relying on cloud inference. Instead of using toy benchmark datasets, the app used short self-recorded workout clips to evaluate how a vision-language model behaves in realistic conditions.

## Architecture learning

### Prototype: `moondream-v2`

The first version used a lightweight multimodal model with a smaller context window. This made the app fast and resource efficient, but it imposed hard limits on what the model could reason about:

- lower image resolution
- fewer frames per clip
- less temporal context across the sequence

This configuration allowed quick experiments, but the output was often shallow and less biomechanically grounded.

### Production: `bonsai-2:4b`

The later version used a stronger vision-language model with a much larger context window. This allowed the app to process higher-quality frames and more sampled frames from each clip. The tradeoff was slower inference and heavier compute usage.

However, the model produced much richer results. It was noticeably better at:

- identifying the exercise being performed
- assessing posture and alignment
- judging stability and control
- describing range of motion and movement quality

## Performance patterns

Across the practical test set, the model showed a clear distinction between structured lifts and dynamic movements.

### Strong cases

The model performed best on exercises with a stable, repetitive pattern and clear movement mechanics, including:

- deadlifts
- squats
- weighted pull-ups

These exercises gave the model cleaner signals around:

- trunk alignment
- hip hinge quality
- depth and control
- load transfer across the body

### Weak cases

The model struggled more on dynamic or equipment-heavy movements, including:

- cable rows
- triceps pushdowns
- farmers walks

These errors were caused by a recurring issue: the model sometimes latched onto a misleading pose or object in the scene and built a false narrative around it instead of tracking the actual movement pattern.

## Prompt engineering fix

A major improvement came from making the prompt more structured and more explicit. The model was instructed to:

1. identify the equipment in the frame
2. classify the movement as stationary or traveling
3. distinguish isolated movement from dynamic movement
4. avoid assuming exercise type from a partial visual cue alone

This reduced false positives significantly and made the model more cautious about over-committing to a wrong exercise interpretation.

## Why the hallucinations happened

The underlying issue was not just model size. It was insufficient temporal grounding.

When a clip is reduced to a few sampled frames, the model can:

- lock onto one static pose
- miss motion direction
- confuse walking with a squat pattern
- misread equipment context and infer the wrong exercise

This matters a lot in fitness analysis because many exercise types only become distinguishable when motion and context are interpreted together.

## Future direction

The next step is not to depend on larger cloud models. Instead, the strongest local approach is to add more structure around the visual model.

### Recommended next steps

1. Hybrid telemetry
   - add joint-angle or pose-based signals into the prompt
2. Adaptive keyframe selection
   - sample around acceleration changes and direction shifts instead of fixed time intervals
3. Temporal buffering
   - add frame-difference vectors or velocity cues so motion direction is explicit

## Final takeaway

This project shows that edge AI is not only about using a smaller model. It is also about reducing context gaps through structured prompt logic, motion-aware sampling, and hybrid telemetry. That is what makes a local fitness analyzer more dependable in real-world use.
