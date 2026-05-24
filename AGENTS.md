# AGENTS.md

Guidance for future coding-agent sessions in this repository.

## Project overview

This repo is a Manim / manim-slides project for visualizing ReLU neural-network behavior, especially how hidden-unit ReLU planes from a small PyTorch classifier combine into a decision surface for `sklearn.datasets.make_circles`.

Key dependencies are managed by `uv` via `pyproject.toml`:

- `manim`
- `manim-slides`
- `torch`
- `numpy`
- `scikit-learn`
- `matplotlib`

## Persistent decision log

- Always update this `AGENTS.md` whenever a project-level decision, convention, architecture choice, workflow rule, or user preference is made.
- Record durable guidance here so future sessions inherit the decision without needing conversation history.
- Keep entries concise and actionable; prefer updating the relevant section over adding scattered notes.

## Important files

- `scenes/slides_example.py`
  - Keep this as a small/simple ReLU slide example.
  - Do not re-add the large decision-surface animation here.

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

## Render commands

Use `uv run` for commands.

Render one atomic scene:

```bash
uv run manim-slides render scenes/relu_planes_decision_surface.py IndividualReLUPlaneScene -ql
```

Other useful render targets:

```bash
uv run manim-slides render scenes/relu_planes_decision_surface.py CirclesDataPlaneScene -ql
uv run manim-slides render scenes/relu_planes_decision_surface.py ReLUPlaneAssemblyScene -ql
uv run manim-slides render scenes/relu_planes_decision_surface.py FinalDecisionSurfaceScene -ql
```

Present after rendering:

```bash
uv run manim-slides present IndividualReLUPlaneScene
```

Quick syntax check:

```bash
uv run python -m py_compile scenes/relu_planes_decision_surface.py scenes/slides_example.py
```

For quick frame-only smoke tests, prefer Manim directly rather than `manim-slides`:

```bash
uv run manim -ql -s --disable_caching scenes/relu_planes_decision_surface.py FinalDecisionSurfaceScene
```

Note: `manim -s` can produce slide-export errors for `ThreeDSlide` classes even after rendering frames. Use `manim-slides render` for full slide validation.

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
- Avoid putting the large ReLU-plane decision-surface scene back into `slides_example.py`.
- Preserve the model architecture from `src/circles_transformation.py` unless explicitly asked to change it:
  - 2D input
  - `hidden_units = 24`
  - one hidden `nn.Linear(2, hidden_units)`
  - ReLU
  - one output `nn.Linear(hidden_units, 1)`

## Implementation notes

- `train_circles_model()` is cached with `functools.cache` so multiple scene classes in the same process reuse trained parameters.
- Hidden-unit contribution is computed as:

```python
a_i * np.maximum(0, w_i[0] * x1 + w_i[1] * x2 + b_i)
```

- The final logit surface is:

```python
b_o + sum_i a_i * relu(w_i dot x + b_i)
```

- The decision boundary is drawn as a marching-squares approximation of `logit(x1, x2) = 0` on the `z=0` plane.

## Validation checklist

Before finishing a scene-related change:

1. Run `py_compile` on edited Python files.
2. Render the specific atomic scene you changed with `manim-slides render ... -ql`.
3. If touching shared helpers, render at least:
   - `IndividualReLUPlaneScene`
   - `FinalDecisionSurfaceScene`
4. Confirm explanatory text remains fixed to the 2D viewport while the camera moves.
