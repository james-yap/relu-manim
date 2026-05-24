# AGENTS.md

Guidance for future coding-agent sessions in this repository.

## Project overview

This repo is a Manim / manim-slides project for visualizing ReLU neural-network behavior, especially how hidden-unit ReLU planes from a small PyTorch classifier combine into a decision surface for `sklearn.datasets.make_circles`.


## Persistent decision log

- Always update this `AGENTS.md` whenever a project-level decision, convention, architecture choice, workflow rule, or user preference is made.
- Always update /Users/james/Documents/Logseq/pages/relu-manim-code.md for math/conceptual preferences and notes.
- Record durable guidance here so future sessions inherit the decision without needing conversation history.
- Keep entries concise and actionable; prefer updating the relevant section over adding scattered notes.

## Important files

- `scenes/relu_planes_decision_surface.py`
  - Main ReLU-plane / decision-surface implementation.
  - Refactored into smaller atomic scene classes so individual pieces can be rendered during iterative development.

- `src/circles_transformation.py`
  - Original notebook-style PyTorch experiment and source reference for `CirclesModel` and hidden-unit plane decomposition.

## Scene classes

`scenes/relu_planes_decision_surface.py` contains these renderable scene classes:

- `CirclesDataPlaneScene`
  - Shows the circles dataset on the `z=0` decision plane.

- `IndividualReLUPlaneScene`
  - Shows individual hidden ReLU-unit planes one at a time.

- `ReLUPlaneAssemblyScene`
  - Shows output-weighted ReLU planes being added into a partial surface.

- `FinalDecisionSurfaceScene`
  - Shows the final all-unit decision surface and boundary curve.

- `ReLUPlanesDecisionSurface`
  - Backward-compatible alias for `FinalDecisionSurfaceScene`.


## Design constraints and conventions

- Keep large animations broken into small logical scene classes.
- Use `manim-slides` only where slide checkpoints are helpful inside an atomic scene.
- Keep explanatory text fixed to the 2D viewport in 3D scenes:
  - Use `add_fixed_in_frame_mobjects` via the scene helper `fixed_overlay(...)`.
  - Do not position labels/explanatory text as regular 3D mobjects unless intentionally labeling geometry.
- Use camera movement to create the 3D illusion:
  - `set_camera_orientation(...)`
  - `move_camera(...)`
  - `begin_ambient_camera_rotation(...)` / `stop_ambient_camera_rotation()`
- Preserve the model architecture from `src/circles_transformation.py` unless explicitly asked to change it:
  - 2D input
  - `hidden_units = 24`
  - one hidden `nn.Linear(2, hidden_units)`
  - ReLU
  - one output `nn.Linear(hidden_units, 1)`

## Implementation notes

- Key dependencies are managed by `uv` via `pyproject.toml`
- `train_circles_model()` is cached with `functools.cache` so multiple scene classes in the same process reuse trained parameters.
- When exporting PyTorch model weights to NumPy for scene math, prefer explicit `model.state_dict()` keys plus `tensor.numpy(force=True)` over relying on `model.parameters()` ordering.
- Hidden-unit contribution is computed as:

```python
a_i * np.maximum(0, w_i[0] * x1 + w_i[1] * x2 + b_i)
```

- The final logit surface is:

```python
b_o + sum_i a_i * relu(w_i dot x + b_i)
```

- The decision boundary is drawn as a marching-squares approximation of `logit(x1, x2) = 0` on the `z=0` plane.
