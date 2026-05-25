"""Problem-statement slides for the linear Study Time vs. GPA setup.

Render with:

    uv run manim-slides render scenes/presentation.py StudyTimeGPAProblem -ql
    uv run manim-slides present StudyTimeGPAProblem

The slide sequence follows the Logseq notes under
``#relu-manim-code/1. Problem Statement``:

1. Show mock linear data on a centered Cartesian plane.
2. Fit a least-squares regression line and draw residual squares.
3. Remove the squares, shift the plot left, and introduce ``Wx + b``.
4. Expand the scalar equation into a matrix multiplication view for all points.
"""

from __future__ import annotations

import numpy as np
from manim import *
from manim_slides import Slide


# Deterministic mock data: study time in hours and GPA on a 4-point scale.
POINT_COUNT = 13
NOISE_RNG = np.random.default_rng(18)
STUDY_HOURS = np.linspace(0.7, 8.6, POINT_COUNT)
GPA_TREND = 1.35 + 0.28 * STUDY_HOURS
GPA_NOISE = NOISE_RNG.normal(loc=0.0, scale=0.28, size=POINT_COUNT)
GPA = GPA_TREND + GPA_NOISE
FIT_SLOPE, FIT_INTERCEPT = np.polyfit(STUDY_HOURS, GPA, deg=1)
GPA_RESIDUALS = GPA - (FIT_SLOPE * STUDY_HOURS + FIT_INTERCEPT)
MAX_ABS_RESIDUAL = np.max(np.abs(GPA_RESIDUALS))
if MAX_ABS_RESIDUAL > 0:
    # Make the least-squares squares legible while preserving the trend/noise shape.
    GPA += (0.5 / MAX_ABS_RESIDUAL - 1.0) * GPA_RESIDUALS
    FIT_SLOPE, FIT_INTERCEPT = np.polyfit(STUDY_HOURS, GPA, deg=1)
    GPA_RESIDUALS = GPA - (FIT_SLOPE * STUDY_HOURS + FIT_INTERCEPT)
    MAX_ABS_RESIDUAL = np.max(np.abs(GPA_RESIDUALS))

PLOT_SHIFT = DOWN * 0.35
COMPACT_PLOT_SHIFT = LEFT * 4.25 + DOWN * 0.55
PLOT_SCALE = 0.55


class StudyTimeGPAProblem(Slide):
    """Four-slide setup for introducing a linear model before ReLU scenes."""

    def make_axes(self) -> Axes:
        axes = Axes(
            x_range=[0, 9, 1],
            y_range=[1.2, 4.1, 0.4],
            x_length=8.7,
            y_length=4.9,
            tips=False,
            axis_config={"include_numbers": True, "font_size": 22},
        )
        axes.shift(PLOT_SHIFT)
        return axes

    def make_axis_labels(self, axes: Axes) -> VGroup:
        x_label = Text("Study time (hours)", font_size=24).next_to(axes.x_axis, DOWN, buff=0.35)
        y_label = Text("GPA", font_size=24).rotate(PI / 2).next_to(axes.y_axis, LEFT, buff=0.35)
        return VGroup(x_label, y_label)

    def make_scatter(self, axes: Axes) -> VGroup:
        dots = VGroup(
            *(
                Dot(axes.c2p(x, y), radius=0.065, color=BLUE_C)
                for x, y in zip(STUDY_HOURS, GPA, strict=True)
            )
        )
        return dots

    def make_regression_line(self, axes: Axes) -> Mobject:
        return axes.plot(
            lambda x: FIT_SLOPE * x + FIT_INTERCEPT,
            x_range=[0.35, 8.85],
            color=YELLOW_C,
            stroke_width=5,
        )

    def make_residual_squares(self, axes: Axes) -> tuple[VGroup, VGroup]:
        residual_segments = VGroup()
        residual_squares = VGroup()

        for x, y in zip(STUDY_HOURS, GPA, strict=True):
            prediction = FIT_SLOPE * x + FIT_INTERCEPT
            observed_point = axes.c2p(x, y)
            predicted_point = axes.c2p(x, prediction)
            residual_segment = Line(
                predicted_point,
                observed_point,
                color=RED_C,
                stroke_width=3,
            )
            side_length = residual_segment.get_length()
            residual_square = Square(side_length=side_length, color=RED_C, stroke_width=3)
            residual_square.set_fill(RED_C, opacity=0.22)
            residual_square.move_to(residual_segment.get_center() + RIGHT * side_length / 2)

            residual_segments.add(residual_segment)
            residual_squares.add(residual_square)

        return residual_segments, residual_squares

    def make_formula_panel(self) -> VGroup:
        panel = RoundedRectangle(
            corner_radius=0.2,
            width=7.35,
            height=5.15,
            color=GREY_B,
            fill_color=BLACK,
            fill_opacity=0.45,
            stroke_width=2,
        )
        panel.to_edge(RIGHT, buff=0.25).shift(DOWN * 0.2)

        header = Text("A first model", font_size=32, color=BLUE_C)
        equation = MathTex(r"\widehat{\mathrm{GPA}} = W x + b", font_size=46)
        definitions = VGroup(
            MathTex(r"x = \mathrm{study\ time}", font_size=30),
            MathTex(r"W = \mathrm{slope}", font_size=30),
            MathTex(r"b = \mathrm{intercept}", font_size=30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        fitted_values = MathTex(
            rf"W \approx {FIT_SLOPE:.2f},\quad b \approx {FIT_INTERCEPT:.2f}",
            font_size=30,
            color=YELLOW_C,
        )
        question = Text("Can this idea handle curved boundaries?", font_size=23, color=GREY_A)

        content = VGroup(header, equation, definitions, fitted_values, question).arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=0.28,
        )
        content.move_to(panel.get_center())
        return VGroup(panel, content)

    def make_matrix_panel_content(self, panel: Mobject) -> VGroup:
        header = Text("All points at once", font_size=22, color=BLUE_C)
        scalar_equation = MathTex(r"y = Wx + b", font_size=30, color=YELLOW_C)
        matrix_hint = Text("Stack examples into one matrix multiply", font_size=16, color=GREY_A)
        matrix_equation = MathTex(
            r"\begin{bmatrix}y_1\\y_2\\\vdots\end{bmatrix}",
            r"=",
            r"\begin{bmatrix}x_{1,1}&x_{1,2}&1\\x_{2,1}&x_{2,2}&1\\\vdots&\vdots&\vdots\end{bmatrix}",
            r"\begin{bmatrix}W_1\\W_2\\b\end{bmatrix}",
            font_size=40,
        )
        matrix_equation[0].set_color(BLUE_C)
        matrix_equation[2].set_color(GREY_A)
        matrix_equation[3].set_color(YELLOW_C)
        matrix_equation.scale_to_fit_width(panel.width - 0.35)

        labels = VGroup(
            VGroup(
                MathTex(r"\mathbf{y}", font_size=21, color=BLUE_C),
                Text("ground truth GPA values", font_size=16),
            ).arrange(RIGHT, buff=0.12),
            VGroup(
                MathTex(r"\mathbf{X}", font_size=21, color=GREY_A),
                Text("fixed inputs", font_size=16),
            ).arrange(RIGHT, buff=0.12),
            VGroup(
                MathTex(r"\theta=[W,b]", font_size=21, color=YELLOW_C),
                Text("optimized parameters", font_size=16),
            ).arrange(RIGHT, buff=0.12),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        footnote = Text(
            "Also see: convex optimization · gradient descent · loss function",
            font_size=12,
            color=GREY_B,
        )

        content = VGroup(
            header,
            scalar_equation,
            matrix_hint,
            matrix_equation,
            labels,
            footnote,
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        content.scale_to_fit_height(panel.height - 0.2)
        content.move_to(panel.get_center())
        return content

    def construct(self):
        title = Text("Problem: Study Time vs. GPA", font_size=22).to_edge(UP, buff=0.12)

        axes = self.make_axes()
        axis_labels = self.make_axis_labels(axes)
        scatter = self.make_scatter(axes)
        plot_group = VGroup(axes, axis_labels, scatter)

        # Slide 1: centered scatter plot.
        self.play(FadeIn(title))
        self.play(FadeIn(axes), FadeIn(axis_labels))
        self.play(LaggedStart(*(FadeIn(dot, scale=0.7) for dot in scatter), lag_ratio=0.07))
        self.next_slide()

        # Slide 2: least-squares line and residual squares.
        regression_line = self.make_regression_line(axes)
        line_label = MathTex(r"\hat y = Wx+b", font_size=32, color=YELLOW_C)
        line_label.next_to(axes.c2p(6.2, FIT_SLOPE * 6.2 + FIT_INTERCEPT), UP, buff=0.18)
        residual_segments, residual_squares = self.make_residual_squares(axes)
        least_squares_note = Text(
            "Least squares chooses the line that makes these squares small",
            font_size=24,
            color=RED_B,
        ).to_edge(DOWN)

        self.play(Create(regression_line), FadeIn(line_label))
        self.play(
            LaggedStart(
                *(Create(segment) for segment in residual_segments),
                lag_ratio=0.04,
                run_time=1.2,
            ),
            LaggedStart(
                *(FadeIn(square, scale=0.85) for square in residual_squares),
                lag_ratio=0.04,
                run_time=1.2,
            ),
            FadeIn(least_squares_note, shift=UP * 0.15),
        )
        self.next_slide()

        # Slide 3: remove squares, shift the data left, and state Wx + b.
        plot_group.add(regression_line, line_label)
        formula_panel = self.make_formula_panel()

        self.play(
            FadeOut(residual_squares),
            FadeOut(residual_segments),
            FadeOut(least_squares_note),
            plot_group.animate.scale(PLOT_SCALE).shift(COMPACT_PLOT_SHIFT),
            run_time=1.25,
        )
        self.play(FadeIn(formula_panel))
        self.next_slide()

        # Slide 4: expand y = Wx + b into one matrix multiplication over all points.
        matrix_region = Rectangle(width=8.65, height=5.35)
        matrix_region.to_edge(RIGHT, buff=0.15).shift(DOWN * 0.12)
        matrix_content = self.make_matrix_panel_content(matrix_region)

        self.play(FadeOut(formula_panel), run_time=0.55)
        self.play(FadeIn(matrix_content), run_time=0.9)
        self.next_slide()
