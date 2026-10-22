# Evaluation Notes

## Test setup

The app was evaluated on a small set of self-recorded workout clips, including:

- deadlifts
- squats
- rows
- farmers walks
- pushdown-related movement

This was a practical evaluation rather than a benchmark-style dataset run. The goal was to determine how a local multimodal model behaves when analyzing real gym footage under imperfect conditions.

## Evaluation summary

Across the test set, the model performed strongly on structured, fixed-axis lifts and less reliably on dynamic or ambiguous movements.

### Strong cases

The best outcomes were observed on exercises with clear mechanical structure and stable posture patterns, including:

- deadlifts
- squats
- weighted pull-ups

These tasks often produced coherent analysis around:

- range of motion
- alignment
- stability
- control under load

### Weak cases

The weakest outcomes appeared in dynamic or spatially complex movements, including:

- cable rows
- triceps pushdowns
- farmers walks

These are the situations where a model can drift toward a plausible but incorrect explanation if it only sees a sparse subset of frames.

## Major failure patterns

### 1. Cable row misread as a lunge-like stance

A split-kneeling or asymmetrical stance caused the model to infer a lunge or stretch pattern and ignore the most important equipment cue: the cable stack. This shows how easily the model can anchor to leg posture and miss the main motion context.

### 2. Triceps pushdown mistaken for a lat pulldown

The model generated a pulldown critique with comments about swinging or kipping even when the movement was a pushdown pattern with elbows pinned. This is a sign of overgeneralization: the model inferred a narrative from incomplete motion cues instead of the actual action.

### 3. Farmers walk confused with squat

This was one of the clearest examples of spatial-temporal confusion. The model treated a traveling movement as a vertical squat because it focused on one static posture rather than the full body transition through space.

## Image-based interpretation

### Image 3: Correct classification example

This image shows a successful exercise identification case. The model correctly recognized the movement and described the progression, posture, and stability in a way that aligned with the visible mechanics.

This is an example of the app working well when the motion is structured, stable, and visually legible from the sampled frames.

![Image 3 - Correct classification example](./images/workout-correct-classification.png)

### Image 4: Misclassification example

This image shows a farmers walk clip that was incorrectly interpreted as a squat-focused movement. The model latched onto a static posture and inferred a different exercise pattern instead of preserving the fact that the athlete was walking through space.

This example is useful because it demonstrates the main limitation of sparse sampling: the model can lose the movement context that distinguishes one exercise from another.

![Image 4 - Misclassification example](./images/workout-misclassification.png)

## Key evaluation takeaway

The project shows that local multimodal analysis works best when the movement is visually clear and the motion structure is easy to infer. It becomes unreliable when the exercise is dynamic, ambiguous, or partially occluded, especially if the model is forced to reason from a small set of static frames.

## Recommended improvements

1. Use adaptive frame selection instead of fixed-interval sampling.
2. Add pose tracking or joint estimation to the analysis pipeline.
3. Strengthen the prompt with explicit movement classification and negative constraints.
4. Include frame-to-frame motion or velocity cues.
5. Give the model equipment-aware prompts so it does not hallucinate movement type from posture alone.

## Summary

This evaluation was valuable because it identified not only strengths but the exact failure modes of a local multimodal analyzer. The model can provide useful form feedback for many structured lifts, but dynamic and travel-based movement remains a key challenge when frame sampling is too sparse and motion context is weak.
