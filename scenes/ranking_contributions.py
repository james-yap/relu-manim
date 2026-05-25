"""Manim-slides scenes illustrating ReLU contribution ranking.

Render slide videos, for example:

    uv run manim-slides render scenes/ranking_contributions.py ReLUBentPlaneImage -ql
    uv run manim-slides render scenes/ranking_contributions.py ContributionVsGradientImage -ql
    uv run manim-slides render scenes/ranking_contributions.py RankingMetricSummaryImage -ql

The 3D plot scenes include a slow looped ambient camera rotation so the clipped
ReLU surfaces are easier to read spatially. The formula summary remains static.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import *
from manim_slides import Slide, ThreeDSlide


GRID_LIMIT = 1.45
SURFACE_RESOLUTION = 24
AMBIENT_LOOP_RUN_TIME = 12
AMBIENT_ROTATION_RATE = 0.045


@dataclass(frozen=True)
class ReLUComponent:
    """One output-weighted hidden ReLU component.

    c(x, y) = a * max(0, w1*x + w2*y + b)
    """

    label: str
    a: float
    w1: float
    w2: float
    b: float
    color: ManimColor

    def value(self, x: float | np.ndarray, y: float | np.ndarray):
        return self.a * np.maximum(0, self.w1 * x + self.w2 * y + self.b)

    @property
    def gradient_magnitude(self) -> float:
        return abs(self.a) * float(np.hypot(self.w1, self.w2))

    def max_abs_on_grid(self, samples: int = 201) -> float:
        axis = np.linspace(-GRID_LIMIT, GRID_LIMIT, samples)
        x, y = np.meshgrid(axis, axis)
        return float(np.max(np.abs(self.value(x, y))))

    def hinge_points(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return two points on w1*x + w2*y + b = 0 inside the square grid."""
        points: list[np.ndarray] = []
        lo, hi = -GRID_LIMIT, GRID_LIMIT

        if abs(self.w2) > 1e-9:
            for x in (lo, hi):
                y = -(self.w1 * x + self.b) / self.w2
                if lo <= y <= hi:
                    points.append(np.array([x, y]))

        if abs(self.w1) > 1e-9:
            for y in (lo, hi):
                x = -(self.w2 * y + self.b) / self.w1
                if lo <= x <= hi:
                    points.append(np.array([x, y]))

        unique: list[np.ndarray] = []
        for point in points:
            if not any(np.linalg.norm(point - other) < 1e-6 for other in unique):
                unique.append(point)

        if len(unique) < 2:
            return None
        return unique[0], unique[1]


def make_axes() -> ThreeDAxes:
    return ThreeDAxes(
        x_range=[-1.5, 1.5, 0.5],
        y_range=[-1.5, 1.5, 0.5],
        z_range=[-0.2, 1.9, 0.5],
        x_length=4.7,
        y_length=4.7,
        z_length=2.7,
    )


def make_surface(
    axes: ThreeDAxes,
    component: ReLUComponent,
    z_scale: float = 1.0,
    opacity: float = 0.72,
) -> Surface:
    surface = Surface(
        lambda u, v: axes.c2p(u, v, z_scale * component.value(u, v)),
        u_range=[-GRID_LIMIT, GRID_LIMIT],
        v_range=[-GRID_LIMIT, GRID_LIMIT],
        resolution=(SURFACE_RESOLUTION, SURFACE_RESOLUTION),
    )
    surface.set_style(
        fill_opacity=opacity,
        stroke_width=0.45,
        stroke_color=component.color,
    )
    surface.set_fill(component.color, opacity=opacity)
    return surface


def make_zero_plane(axes: ThreeDAxes) -> Surface:
    plane = Surface(
        lambda u, v: axes.c2p(u, v, 0),
        u_range=[-GRID_LIMIT, GRID_LIMIT],
        v_range=[-GRID_LIMIT, GRID_LIMIT],
        resolution=(8, 8),
    )
    plane.set_style(fill_opacity=0.14, stroke_width=0.2, stroke_color=GREY_B)
    plane.set_fill(GREY_B, opacity=0.14)
    return plane


def make_hinge_line(axes: ThreeDAxes, component: ReLUComponent) -> Mobject:
    endpoints = component.hinge_points()
    if endpoints is None:
        return VGroup()
    start, end = endpoints
    return Line(
        axes.c2p(start[0], start[1], 0.035),
        axes.c2p(end[0], end[1], 0.035),
        color=WHITE,
        stroke_width=5,
    )


def metric_card(component: ReLUComponent, title: str, rank_text: str) -> VGroup:
    max_abs = component.max_abs_on_grid()
    grad = component.gradient_magnitude
    card = VGroup(
        Text(title, font_size=25, color=component.color),
        MathTex(r"\max_{grid}|c_i|", "=", f"{max_abs:.2f}", font_size=30),
        MathTex(r"|a_i|\lVert w_i\rVert", "=", f"{grad:.2f}", font_size=30),
        Text(rank_text, font_size=20, color=GREY_A),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
    return card


class StaticOverlayMixin:
    """Helpers for 3D slide scenes with text locked to the image plane."""

    def fixed(self, *mobjects: Mobject) -> VGroup:
        self.add_fixed_in_frame_mobjects(*mobjects)
        return VGroup(*mobjects)

    def ambient_rotation_loop(
        self,
        run_time: float = AMBIENT_LOOP_RUN_TIME,
        rate: float = AMBIENT_ROTATION_RATE,
    ) -> None:
        """Create one manim-slides loop containing a slow ambient rotation."""
        self.next_slide(loop=True)
        self.begin_ambient_camera_rotation(rate=rate)
        # Animate an invisible tracker so manim-slides records a real loop segment
        # while Manim's ambient camera updater advances the 3D camera.
        ticker = ValueTracker(0)
        self.play(ticker.animate.set_value(1), run_time=run_time, rate_func=linear)
        self.stop_ambient_camera_rotation()
        self.next_slide()

    def rotating_mobjects_loop(
        self,
        *rotations: tuple[Mobject, np.ndarray, float],
        run_time: float = AMBIENT_LOOP_RUN_TIME,
    ) -> None:
        """Create one loop by rotating mobjects around independent centers.

        Each rotation is ``(mobject, about_point, angle)``. Use angles that are
        whole multiples of TAU so the repeated manim-slides loop is seamless.
        """
        self.next_slide(loop=True)
        self.play(
            *(
                Rotate(mobject, angle=angle, axis=OUT, about_point=about_point)
                for mobject, about_point, angle in rotations
            ),
            run_time=run_time,
            rate_func=linear,
        )
        self.next_slide()


class ReLUBentPlaneImage(StaticOverlayMixin, ThreeDSlide):
    """Slide loop: a ReLU component is a clipped/hinged plane."""

    def construct(self):
        component = ReLUComponent(
            label="one hidden unit",
            a=1.0,
            w1=1.0,
            w2=-0.65,
            b=0.0,
            color=BLUE_C,
        )

        axes = make_axes().shift(DOWN * 0.25)
        zero_plane = make_zero_plane(axes)
        surface = make_surface(axes, component, z_scale=0.8)
        hinge = make_hinge_line(axes, component)

        title = Text("One ReLU unit is a bent plane", font_size=38).to_edge(UP)
        formula = MathTex(
            r"c_i(x,y)=a_i\max(0,w_{i1}x+w_{i2}y+b_i)",
            font_size=34,
        ).next_to(title, DOWN, buff=0.22)
        notes = VGroup(
            Text("inactive side: flat at z = 0", font_size=23, color=GREY_A),
            Text("active side: follows the sloped plane", font_size=23, color=BLUE_B),
            Text("white line: hinge where w·x + b = 0", font_size=23, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_corner(DL)

        self.set_camera_orientation(phi=63 * DEGREES, theta=-42 * DEGREES, zoom=0.82)
        self.fixed(title, formula, notes)
        self.play(
            FadeIn(axes),
            FadeIn(zero_plane),
            FadeIn(surface),
            FadeIn(hinge),
            run_time=0.8,
        )
        self.ambient_rotation_loop()


class ContributionVsGradientImage(StaticOverlayMixin, ThreeDSlide):
    """Slide loop: contribution and gradient magnitude can rank units differently."""

    def construct(self):
        high_contribution = ReLUComponent(
            label="A",
            a=1.0,
            w1=0.35,
            w2=0.10,
            b=1.0,
            color=GREEN_C,
        )
        high_gradient = ReLUComponent(
            label="B",
            a=1.2,
            w1=1.2,
            w2=-0.8,
            b=-2.0,
            color=ORANGE,
        )

        left_axes = make_axes().scale(0.86).shift(LEFT * 3.2 + DOWN * 0.45)
        right_axes = make_axes().scale(0.86).shift(RIGHT * 3.2 + DOWN * 0.45)

        left_surface = make_surface(left_axes, high_contribution, opacity=0.78)
        right_surface = make_surface(right_axes, high_gradient, opacity=0.78)

        title = Text("Contribution ranking is not the same as steepness ranking", font_size=34)
        title.to_edge(UP)
        subtitle = MathTex(
            r"\text{current code ranks by }\max_{grid}|c_i(x,y)|",
            font_size=31,
        ).next_to(title, DOWN, buff=0.20)

        left_card = metric_card(
            high_contribution,
            "Unit A: bigger visible height",
            "rank #1 by max |contribution|",
        ).to_corner(UL).shift(DOWN * 1.15)
        right_card = metric_card(
            high_gradient,
            "Unit B: steeper active plane",
            "rank #1 by gradient magnitude",
        ).to_corner(UR).shift(DOWN * 1.15)

        conclusion = Text(
            "max |cᵢ| measures visible logit effect; |aᵢ|‖wᵢ‖ measures active-side slope",
            font_size=23,
            color=GREY_A,
        ).to_edge(DOWN)

        left_zero_plane = make_zero_plane(left_axes)
        left_hinge = make_hinge_line(left_axes, high_contribution)
        right_zero_plane = make_zero_plane(right_axes)
        right_hinge = make_hinge_line(right_axes, high_gradient)

        left_plot = VGroup(left_axes, left_zero_plane, left_surface, left_hinge)
        right_plot = VGroup(right_axes, right_zero_plane, right_surface, right_hinge)

        self.set_camera_orientation(phi=62 * DEGREES, theta=-48 * DEGREES, zoom=0.75)
        self.fixed(title, subtitle, left_card, right_card, conclusion)
        self.play(
            FadeIn(left_plot),
            FadeIn(right_plot),
            run_time=0.8,
        )
        self.rotating_mobjects_loop(
            (left_plot, left_axes.c2p(0, 0, 0), TAU),
            (right_plot, right_axes.c2p(0, 0, 0), -TAU),
        )


class RankingMetricSummaryImage(Slide):
    """Static slide: summary of the two different ranking questions."""

    def construct(self):
        title = Text("Two different questions give two different rankings", font_size=38)
        title.to_edge(UP)

        left_box = RoundedRectangle(
            width=6.1,
            height=4.1,
            corner_radius=0.18,
            color=GREEN_C,
            fill_color=GREEN_E,
            fill_opacity=0.16,
        )
        right_box = RoundedRectangle(
            width=6.1,
            height=4.1,
            corner_radius=0.18,
            color=ORANGE,
            fill_color=ORANGE,
            fill_opacity=0.11,
        )
        boxes = VGroup(left_box, right_box).arrange(RIGHT, buff=0.45).shift(DOWN * 0.1)

        left_header = Text("Visible contribution", font_size=30, color=GREEN_B)
        left_formula = MathTex(r"\max_{grid}|a_i\max(0,w_i\cdot x+b_i)|", font_size=32)
        left_question = Text("Which unit moves the logit surface\nmost up/down in this viewport?", font_size=23)
        left_factors = VGroup(
            Text("depends on:", font_size=22, color=GREY_A),
            Text("output weight aᵢ", font_size=21),
            Text("slope wᵢ", font_size=21),
            Text("hinge  offset bᵢ", font_size=21),
            Text("active  area", font_size=21),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        left_content = VGroup(left_header, left_formula, left_question, left_factors)
        left_content.arrange(DOWN, buff=0.22).move_to(left_box)

        right_header = Text("Active-side steepness", font_size=30, color=ORANGE)
        right_formula = MathTex(r"\lVert\nabla c_i\rVert=|a_i|\lVert w_i\rVert", font_size=32)
        right_question = Text("Which unit has the largest local\ngradient when it is active?", font_size=23)
        right_factors = VGroup(
            Text("ignores:", font_size=22, color=GREY_A),
            Text("vertical  offset", font_size=21),
            Text("hinge  location", font_size=21),
            Text("active  area", font_size=21),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        right_content = VGroup(right_header, right_formula, right_question, right_factors)
        right_content.arrange(DOWN, buff=0.22).move_to(right_box)

        footer = Text(
            "So choose_visible_units(...) ranks by visual/logit impact, not by pure gradient.",
            font_size=25,
            color=GREY_A,
        ).to_edge(DOWN)

        self.play(FadeIn(title, boxes, left_content, right_content, footer), run_time=0.8)
        self.next_slide()
