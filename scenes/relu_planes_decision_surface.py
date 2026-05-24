from functools import cache

import numpy as np
import torch
from manim import *
from manim_slides import ThreeDSlide
from sklearn.datasets import make_circles
from torch import nn
import torch.nn.functional as F


HIDDEN_UNITS = 4
GRID_LIMIT = 1.45
VISIBLE_UNIT_COUNT = 4


class CirclesModel(nn.Module):
    """Same two-layer architecture used in src/circles_transformation.py."""

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, HIDDEN_UNITS)
        self.output = nn.Linear(HIDDEN_UNITS, 1)

    def forward(self, x):
        x_hidden = self.hidden(x)
        return self.output(F.relu(x_hidden))


@cache
def train_circles_model(seed=6, epochs=100):
    """Train the circle classifier with PyTorch, then expose NumPy parameters."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_np, y_np = make_circles(noise=0.1, factor=0.3, random_state=0)
    x = torch.as_tensor(x_np, dtype=torch.float32)
    y = torch.as_tensor(y_np, dtype=torch.float32).unsqueeze(1)

    model = CirclesModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)

    model.train()
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    model.eval()
    state = {name: tensor.numpy(force=True) for name, tensor in model.state_dict().items()}
    params = (
        state["hidden.weight"],
        state["hidden.bias"],
        state["output.weight"],
        state["output.bias"],
    )
    return x_np, y_np, params, tuple(losses)


def component_value(params, relu_idx, x1, x2):
    """Output-weighted contribution of one hidden ReLU unit."""
    hidden_weights, hidden_biases, output_weights, _output_bias = params
    w1, w2 = hidden_weights[relu_idx]
    b = hidden_biases[relu_idx]
    output_weight = output_weights[0][relu_idx]
    return output_weight * np.maximum(0, w1 * x1 + w2 * x2 + b)


def logit_value(params, x1, x2, upto=None):
    """Network logit as a sum of output-weighted ReLU planes."""
    hidden_weights, hidden_biases, output_weights, output_bias = params
    unit_indices = range(HIDDEN_UNITS) if upto is None else upto

    z = np.zeros_like(np.asarray(x1, dtype=float)) + float(output_bias[0])
    for relu_idx in unit_indices:
        w1, w2 = hidden_weights[relu_idx]
        b = hidden_biases[relu_idx]
        output_weight = output_weights[0][relu_idx]
        z += output_weight * np.maximum(0, w1 * x1 + w2 * x2 + b)
    return z


def grid_values(params, fn, samples=61):
    axis = np.linspace(-GRID_LIMIT, GRID_LIMIT, samples)
    x1, x2 = np.meshgrid(axis, axis)
    return fn(params, x1, x2)


def choose_visible_units(params, count=VISIBLE_UNIT_COUNT):
    """Pick the hidden units with the largest grid contribution."""
    scores = []
    for relu_idx in range(HIDDEN_UNITS):
        values = grid_values(
            params, lambda p, x1, x2: component_value(p, relu_idx, x1, x2)
        )
        scores.append((float(np.max(np.abs(values))), relu_idx))
    return tuple(relu_idx for _score, relu_idx in sorted(scores, reverse=True)[:count])


def compute_z_scale(params, selected_units):
    final_grid = grid_values(params, lambda p, x1, x2: logit_value(p, x1, x2))
    component_grids = [
        grid_values(params, lambda p, x1, x2, idx=idx: component_value(p, idx, x1, x2))
        for idx in selected_units
    ]
    max_abs = max(
        np.max(np.abs(final_grid)), *(np.max(np.abs(g)) for g in component_grids)
    )
    return 1.85 / max(max_abs, 1e-6)


def zero_contour_segments(params, samples=70):
    """Marching-squares line segments for logit(x, y) = 0."""
    axis = np.linspace(-GRID_LIMIT, GRID_LIMIT, samples)
    x1, x2 = np.meshgrid(axis, axis)
    z = logit_value(params, x1, x2)
    segments = []

    def interp(p_a, p_b, v_a, v_b):
        if abs(v_a - v_b) < 1e-9:
            t = 0.5
        else:
            t = float(v_a / (v_a - v_b))
        t = np.clip(t, 0.0, 1.0)
        return p_a + t * (p_b - p_a)

    for row in range(samples - 1):
        for col in range(samples - 1):
            pts = [
                np.array([axis[col], axis[row]]),
                np.array([axis[col + 1], axis[row]]),
                np.array([axis[col + 1], axis[row + 1]]),
                np.array([axis[col], axis[row + 1]]),
            ]
            vals = [z[row, col], z[row, col + 1], z[row + 1, col + 1], z[row + 1, col]]
            crossings = []
            for edge_idx in range(4):
                v_a = vals[edge_idx]
                v_b = vals[(edge_idx + 1) % 4]
                if ((v_a <= 0 <= v_b) or (v_b <= 0 <= v_a)) and abs(v_a - v_b) > 1e-9:
                    crossings.append(
                        interp(pts[edge_idx], pts[(edge_idx + 1) % 4], v_a, v_b)
                    )
            if len(crossings) == 2:
                segments.append((crossings[0], crossings[1]))
            elif len(crossings) == 4:
                segments.append((crossings[0], crossings[1]))
                segments.append((crossings[2], crossings[3]))
    return segments


class ReLUPlanesBaseScene(ThreeDSlide):
    """Shared setup for the small ReLU-plane scenes.

    Manim Community docs: ``add_fixed_in_frame_mobjects`` overlays mobjects so
    camera rotation/movement does not affect them. Text/UI helpers here use that
    API so annotations stay in the 2D viewport rather than the 3D scene.
    """

    palette = [BLUE_C, TEAL_C, GREEN_C, GOLD_C, ORANGE, PURPLE_C]

    def setup_model_data(self):
        self.x_np, self.y_np, self.params, self.losses = train_circles_model()
        self.selected_units = choose_visible_units(self.params)
        self.z_scale = compute_z_scale(self.params, self.selected_units)

    def make_axes(self):
        return ThreeDAxes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            z_range=[-2.0, 2.0, 1.0],
            x_length=6.0,
            y_length=6.0,
            z_length=3.4,
        ).shift(DOWN * 0.25)

    def set_default_camera(self):
        self.set_camera_orientation(phi=62 * DEGREES, theta=-42 * DEGREES, zoom=0.72)

    def fixed_overlay(self, *mobjects, visible=False):
        """Place text/UI mobjects in the 2D viewport, not the 3D camera frame."""
        for mobject in mobjects:
            mobject.set_opacity(1 if visible else 0)
        self.add_fixed_in_frame_mobjects(*mobjects)
        return mobjects[0] if len(mobjects) == 1 else VGroup(*mobjects)

    def title_overlay(self, title_text, subtitle_tex=None):
        title = Text(title_text, font_size=38).to_edge(UP)
        if subtitle_tex is None:
            return self.fixed_overlay(title)
        subtitle = MathTex(subtitle_tex, font_size=36).next_to(title, DOWN, buff=0.25)
        return self.fixed_overlay(title, subtitle)

    def make_surface(self, axes, fn, color=BLUE_C, opacity=0.55, resolution=18):
        surface = Surface(
            lambda u, v: axes.c2p(u, v, fn(u, v)),
            u_range=[-GRID_LIMIT, GRID_LIMIT],
            v_range=[-GRID_LIMIT, GRID_LIMIT],
            resolution=(resolution, resolution),
        )
        surface.set_style(fill_opacity=opacity, stroke_width=0.35, stroke_color=color)
        surface.set_fill(color, opacity=opacity)
        return surface

    def make_zero_plane(self, axes):
        plane = self.make_surface(
            axes,
            lambda _u, _v: 0,
            color=GREY_B,
            opacity=0.18,
            resolution=8,
        )
        plane.set_style(stroke_width=0.15, stroke_color=GREY_C)
        return plane

    def make_points(self, axes):
        points = VGroup()
        for point, label in zip(self.x_np, self.y_np):
            color = YELLOW if label == 1 else RED_C
            points.add(
                Dot3D(axes.c2p(point[0], point[1], 0.035), radius=0.035, color=color)
            )
        return points

    def make_boundary(self, axes):
        boundary = VGroup()
        for start, end in zero_contour_segments(self.params):
            boundary.add(
                Line3D(
                    axes.c2p(start[0], start[1], 0.04),
                    axes.c2p(end[0], end[1], 0.04),
                    color=WHITE,
                    thickness=0.018,
                )
            )
        return boundary

    def make_component_surface(self, axes, relu_idx, color=BLUE_C, opacity=0.58):
        return self.make_surface(
            axes,
            lambda u, v, idx=relu_idx: (
                self.z_scale * component_value(self.params, idx, u, v)
            ),
            color=color,
            opacity=opacity,
        )

    def make_partial_surface(self, axes, unit_indices, color=GREEN_C, opacity=0.52):
        return self.make_surface(
            axes,
            lambda u, v, units=tuple(unit_indices): (
                self.z_scale * logit_value(self.params, u, v, upto=units)
            ),
            color=color,
            opacity=opacity,
        )

    def make_final_surface(self, axes):
        return self.make_surface(
            axes,
            lambda u, v: self.z_scale * logit_value(self.params, u, v),
            color=GREEN_C,
            opacity=0.66,
            resolution=24,
        )


class CirclesDataPlaneScene(ReLUPlanesBaseScene):
    """Atomic scene: the circles dataset on the z=0 decision plane.

    Render:
        uv run manim-slides render scenes/relu_planes_decision_surface.py CirclesDataPlaneScene -ql
    """

    def construct(self):
        self.setup_model_data()
        axes = self.make_axes()
        zero_plane = self.make_zero_plane(axes)
        points = self.make_points(axes)

        title = self.title_overlay(
            "Circles data lives on the decision plane",
            r"z=0 \text{ is the classification threshold}",
        )
        caption = Text(
            "yellow = inner class, red = outer class", font_size=24, color=GREY_A
        ).to_corner(UL)
        self.fixed_overlay(caption)

        self.set_default_camera()
        self.play(title.animate.set_opacity(1))
        self.next_slide()

        self.play(
            Create(axes),
            FadeIn(zero_plane),
            FadeIn(points),
            caption.animate.set_opacity(1),
        )
        self.move_camera(phi=70 * DEGREES, theta=25 * DEGREES, zoom=0.72, run_time=2)
        self.move_camera(phi=58 * DEGREES, theta=-64 * DEGREES, zoom=0.72, run_time=2)
        self.next_slide()


class IndividualReLUPlaneScene(ReLUPlanesBaseScene):
    """Atomic scene: inspect individual hidden ReLU planes one at a time.

    Render:
        uv run manim-slides render scenes/relu_planes_decision_surface.py IndividualReLUPlaneScene -ql
    """

    def construct(self):
        self.setup_model_data()
        axes = self.make_axes()
        zero_plane = self.make_zero_plane(axes)
        points = self.make_points(axes)

        title = self.title_overlay(
            "A hidden unit is one clipped plane",
            r"z_i(x)=a_i\max(0,w_i\cdot x+b_i)",
        )
        unit_label = Text("", font_size=24).to_corner(DR)
        self.fixed_overlay(unit_label)

        self.set_default_camera()
        self.play(
            title.animate.set_opacity(1),
            Create(axes),
            FadeIn(zero_plane),
            FadeIn(points),
        )
        self.next_slide()

        first_idx = self.selected_units[0]
        current_surface = self.make_component_surface(
            axes, first_idx, color=self.palette[0]
        )
        next_label = Text(
            f"unit {first_idx}", font_size=24, color=self.palette[0]
        ).to_corner(DR)
        self.play(
            FadeIn(current_surface),
            Transform(unit_label, next_label),
            unit_label.animate.set_opacity(1),
        )
        self.move_camera(phi=72 * DEGREES, theta=-15 * DEGREES, zoom=0.72, run_time=1.8)
        self.next_slide()

        for color, relu_idx in zip(self.palette[1:], self.selected_units[1:]):
            next_surface = self.make_component_surface(axes, relu_idx, color=color)
            next_label = Text(f"unit {relu_idx}", font_size=24, color=color).to_corner(
                DR
            )
            self.play(
                ReplacementTransform(current_surface, next_surface),
                Transform(unit_label, next_label),
            )
            current_surface = next_surface
            self.next_slide()


class ReLUPlaneAssemblyScene(ReLUPlanesBaseScene):
    """Atomic scene: add the output-weighted ReLU planes into a partial sum.

    Render:
        uv run manim-slides render scenes/relu_planes_decision_surface.py ReLUPlaneAssemblyScene -ql
    """

    def construct(self):
        self.setup_model_data()
        axes = self.make_axes()
        zero_plane = self.make_zero_plane(axes)
        points = self.make_points(axes)

        title = self.title_overlay("Add output-weighted ReLU planes")
        running_label = MathTex("b_o", font_size=34, color=GREY_A).to_corner(UR)
        self.fixed_overlay(running_label)

        bias_surface = self.make_surface(
            axes,
            lambda _u, _v: self.z_scale * float(self.params[3][0]),
            color=GREY_A,
            opacity=0.35,
            resolution=8,
        )

        self.set_default_camera()
        self.play(
            title.animate.set_opacity(1),
            Create(axes),
            FadeIn(zero_plane),
            FadeIn(points),
        )
        self.play(FadeIn(bias_surface), running_label.animate.set_opacity(1))
        self.next_slide()

        running_units = []
        running_surface = bias_surface
        for step, (color, relu_idx) in enumerate(
            zip(self.palette, self.selected_units), start=1
        ):
            running_units.append(relu_idx)
            partial_surface = self.make_partial_surface(
                axes, running_units, color=color
            )
            partial_label = MathTex(
                rf"b_o+\sum_{{k=1}}^{{{step}}} z_k(x)",
                font_size=32,
                color=color,
            ).to_corner(UR)
            self.play(
                ReplacementTransform(running_surface, partial_surface),
                Transform(running_label, partial_label),
            )
            running_surface = partial_surface
            self.next_slide()


class FinalDecisionSurfaceScene(ReLUPlanesBaseScene):
    """Atomic scene: show the final all-unit decision surface and boundary.

    Render:
        uv run manim-slides render scenes/relu_planes_decision_surface.py FinalDecisionSurfaceScene -ql
    """

    def construct(self):
        self.setup_model_data()
        axes = self.make_axes()
        zero_plane = self.make_zero_plane(axes)
        points = self.make_points(axes)
        final_surface = self.make_final_surface(axes)
        boundary = self.make_boundary(axes)

        title = self.title_overlay(
            "All hidden planes form the decision surface",
            rf"\text{{all }}{HIDDEN_UNITS}\text{{ ReLU planes summed together}}",
        )
        caption = Text(
            "The white curve is where the summed surface crosses z=0",
            font_size=24,
            color=WHITE,
        ).to_corner(UL)
        loss_text = Text(
            f"PyTorch trained CirclesModel: final loss {self.losses[-1]:.4f}",
            font_size=24,
            color=GREY_A,
        ).to_edge(DOWN)
        self.fixed_overlay(caption, loss_text)

        self.set_default_camera()
        self.play(
            title.animate.set_opacity(1),
            Create(axes),
            FadeIn(zero_plane),
            FadeIn(points),
        )
        self.play(
            FadeIn(final_surface), FadeIn(boundary), caption.animate.set_opacity(1)
        )
        self.next_slide()

        self.move_camera(phi=50 * DEGREES, theta=-110 * DEGREES, zoom=0.72, run_time=2)
        self.move_camera(phi=66 * DEGREES, theta=35 * DEGREES, zoom=0.72, run_time=2)
        self.play(loss_text.animate.set_opacity(1))
        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(3)
        self.stop_ambient_camera_rotation()
        self.next_slide()


class ReLUPlanesDecisionSurface(FinalDecisionSurfaceScene):
    """Backward-compatible alias for the final-surface atomic scene."""


if __name__ == "__main__":
    train_circles_model()
