"""Problem-statement slides for the linear Study Time vs. GPA setup.

Render with:

    uv run manim-slides render scenes/presentation.py ReluPresentation -ql
    uv run manim-slides present ReluPresentation

The slide sequence follows the Logseq notes under
``#relu-manim-code/1. Problem Statement``:

0. Open with ``ReLUs, explained quickly`` and the iconic ReLU graph.
1. Show mock linear data on a centered Cartesian plane.
2. Fit a least-squares regression line and draw residual squares.
3. Remove the squares, fade out the plot, and introduce centered ``Xw + b``.
4. Expand the scalar equation into a centered matrix multiplication view for all points.
5. Dim fixed data terms and highlight trainable ``w`` and ``b``.
6. Switch to a U-shaped Temperature vs. Energy Consumption dataset.
7. Show why the same linear ``Xw + b`` model fails on the curve.
8. Fade the graph away and center the linear formulation for the next step.
9. Rewrite that same linear formulation as a one-feature matrix equation.
10. Augment the input matrix with a squared feature column and matching weight.
11. Cross fade back to the temperature graph and show the quadratic fit working.
12. Return to the matrix formulation and highlight the full design matrix.
13. Highlight the engineered ``x^2`` column and matching ``w_2`` weight.
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
# Deterministic U-shaped data: temperature and energy consumption.
TEMPERATURE_COUNT = 15
TEMPERATURE_RNG = np.random.default_rng(22)
TEMPERATURE_C = np.linspace(0, 40, TEMPERATURE_COUNT)
ENERGY_NOISE = TEMPERATURE_RNG.normal(loc=0.0, scale=0.11, size=TEMPERATURE_COUNT)
ENERGY_USE = 2.05 + 0.0065 * (TEMPERATURE_C - 21) ** 2 + ENERGY_NOISE
ENERGY_FIT_SLOPE, ENERGY_FIT_INTERCEPT = np.polyfit(TEMPERATURE_C, ENERGY_USE, deg=1)
ENERGY_QUAD_A, ENERGY_QUAD_B, ENERGY_QUAD_C = np.polyfit(TEMPERATURE_C, ENERGY_USE, deg=2)


def energy_curve(temperature: float | np.ndarray) -> float | np.ndarray:
    """Smooth quadratic curve underlying the temperature-energy mock data."""

    return 2.05 + 0.0065 * (temperature - 21) ** 2


class ReluPresentation(Slide):
    """Fourteen-slide setup for motivating nonlinear features before ReLU scenes."""

    def make_relu_title_graph(self) -> VGroup:
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 3, 1],
            x_length=5.6,
            y_length=3.2,
            tips=True,
            axis_config={"include_numbers": False, "stroke_width": 4},
        )
        axes.move_to(ORIGIN).shift(DOWN * 0.2)

        graph = axes.plot(
            lambda x: max(0, x),
            x_range=[-3, 3],
            color=YELLOW_C,
            stroke_width=8,
        )
        origin_dot = Dot(axes.c2p(0, 0), radius=0.08, color=BLUE_C)
        equation = MathTex(r"\operatorname{ReLU}(x)=\max(0,x)", font_size=38, color=GREY_A)
        equation.next_to(axes, DOWN, buff=0.35)

        return VGroup(axes, graph, origin_dot, equation)

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
            width=8.6,
            height=5.0,
            color=GREY_B,
            fill_color=BLACK,
            fill_opacity=0.45,
            stroke_width=2,
        )
        panel.move_to(ORIGIN).shift(DOWN * 0.15)

        header = Text("A first model", font_size=32, color=BLUE_C)
        equation = MathTex(r"\widehat{\mathrm{GPA}} = Xw + b", font_size=46)
        definitions = VGroup(
            MathTex(r"X = \mathrm{study\ time}", font_size=30),
            MathTex(r"w = \mathrm{slope}", font_size=30),
            MathTex(r"b = \mathrm{intercept}", font_size=30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        fitted_values = MathTex(
            rf"w \approx {FIT_SLOPE:.2f},\quad b \approx {FIT_INTERCEPT:.2f}",
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

    def make_temperature_axes(self) -> Axes:
        axes = Axes(
            x_range=[0, 40, 5],
            y_range=[1.6, 5.25, 0.5],
            x_length=8.8,
            y_length=4.9,
            tips=False,
            axis_config={"include_numbers": True, "font_size": 22},
        )
        axes.shift(PLOT_SHIFT)
        return axes

    def make_temperature_axis_labels(self, axes: Axes) -> VGroup:
        x_label = Text("Temperature (°C)", font_size=24).next_to(axes.x_axis, DOWN, buff=0.35)
        y_label = Text("Energy use", font_size=24).rotate(PI / 2).next_to(axes.y_axis, LEFT, buff=0.35)
        return VGroup(x_label, y_label)

    def make_temperature_scatter(self, axes: Axes) -> VGroup:
        return VGroup(
            *(
                Dot(axes.c2p(x, y), radius=0.06, color=GREEN_C)
                for x, y in zip(TEMPERATURE_C, ENERGY_USE, strict=True)
            )
        )

    def make_temperature_curve(self, axes: Axes) -> Mobject:
        return axes.plot(
            energy_curve,
            x_range=[0, 40],
            color=GREEN_B,
            stroke_width=5,
        )

    def make_temperature_linear_attempt(self, axes: Axes) -> Mobject:
        return axes.plot(
            lambda x: ENERGY_FIT_SLOPE * x + ENERGY_FIT_INTERCEPT,
            x_range=[0, 40],
            color=RED_C,
            stroke_width=5,
        )

    def make_temperature_quadratic_fit(self, axes: Axes) -> Mobject:
        return axes.plot(
            lambda x: ENERGY_QUAD_A * x**2 + ENERGY_QUAD_B * x + ENERGY_QUAD_C,
            x_range=[0, 40],
            color=YELLOW_C,
            stroke_width=5,
        )

    def make_temperature_residuals(self, axes: Axes) -> VGroup:
        residuals = VGroup()
        for x, y in zip(TEMPERATURE_C, ENERGY_USE, strict=True):
            prediction = ENERGY_FIT_SLOPE * x + ENERGY_FIT_INTERCEPT
            residuals.add(
                Line(
                    axes.c2p(x, prediction),
                    axes.c2p(x, y),
                    color=RED_C,
                    stroke_width=2,
                )
            )
        return residuals

    def make_matrix_panel_content(self, panel: Mobject) -> VGroup:
        header = Text("All points at once", font_size=22, color=BLUE_C)
        scalar_equation = MathTex(r"y = Xw + b", font_size=30, color=YELLOW_C)
        matrix_hint = Text("Stack examples into one matrix multiply", font_size=16, color=GREY_A)
        matrix_equation = MathTex(
            r"\begin{bmatrix}y_1\\y_2\\\vdots\end{bmatrix}",
            r"=",
            r"\begin{bmatrix}x_1\\x_2\\\vdots\end{bmatrix}",
            r"w",
            r"+",
            r"\begin{bmatrix}b\\b\\\vdots\end{bmatrix}",
            font_size=46,
        )
        matrix_equation[0].set_color(BLUE_C)
        matrix_equation[2].set_color(GREY_A)
        matrix_equation[3].set_color(YELLOW_C)
        matrix_equation[5].set_color(YELLOW_C)
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
                MathTex(r"\mathbf{w}, b", font_size=21, color=YELLOW_C),
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

    def make_quadratic_matrix_content(self) -> VGroup:
        header = Text("Same model, now stacked", font_size=22, color=BLUE_C)
        scalar_equation = MathTex(r"\hat{\mathbf y} = Xw + b", font_size=36, color=YELLOW_C)
        matrix_hint = Text("One temperature column means one slope", font_size=18, color=GREY_A)
        matrix_equation = MathTex(
            r"\begin{bmatrix}\hat y_1\\\hat y_2\\\vdots\end{bmatrix}",
            r"=",
            r"\begin{bmatrix}x_1\\x_2\\\vdots\end{bmatrix}",
            r"w",
            r"+",
            r"\begin{bmatrix}b\\b\\\vdots\end{bmatrix}",
            font_size=48,
        )
        matrix_equation[0].set_color(BLUE_C)
        matrix_equation[2].set_color(GREY_A)
        matrix_equation[3].set_color(YELLOW_C)
        matrix_equation[5].set_color(YELLOW_C)
        matrix_equation.scale_to_fit_width(9.7)
        limitation = Text("Still no way to turn upward at both ends", font_size=22, color=RED_B)

        content = VGroup(header, scalar_equation, matrix_hint, matrix_equation, limitation).arrange(
            DOWN,
            buff=0.22,
        )
        content.move_to(ORIGIN).shift(DOWN * 0.05)
        return content

    def make_quadratic_augmentation_content(self) -> VGroup:
        header = Text("Augment X with one new feature", font_size=22, color=BLUE_C)
        feature_rule = MathTex(r"x_{i,2}=x_i^2", font_size=32, color=GREEN_B)
        feature_rule_note = Text("squared temperature becomes a second input column", font_size=17, color=GREY_A)
        feature_rule_group = VGroup(feature_rule, feature_rule_note).arrange(DOWN, buff=0.08)

        matrix_equation = MathTex(
            r"\begin{bmatrix}\hat y_1\\\hat y_2\\\vdots\end{bmatrix}",
            r"=",
            r"\begin{bmatrix}x_1&x_1^2\\x_2&x_2^2\\\vdots&\vdots\end{bmatrix}",
            r"\begin{bmatrix}w_1\\w_2\end{bmatrix}",
            r"+",
            r"\begin{bmatrix}b\\b\\\vdots\end{bmatrix}",
            font_size=43,
            substrings_to_isolate=[r"x_1^2", r"x_2^2", r"w_2"],
        )
        matrix_equation[0].set_color(BLUE_C)
        matrix_equation[2].set_color(GREY_A)
        matrix_equation[3].set_color(YELLOW_C)
        matrix_equation[5].set_color(YELLOW_C)
        for tex in (r"x_1^2", r"x_2^2", r"w_2"):
            matrix_equation.set_color_by_tex(tex, GREEN_B)
        matrix_equation.scale_to_fit_width(10.8)

        resolved_equation = MathTex(
            r"\hat y_i = w_1 x_i + w_2 x_i^2 + b",
            font_size=39,
            substrings_to_isolate=[r"w_2 x_i^2"],
        )
        resolved_equation.set_color_by_tex(r"w_2 x_i^2", GREEN_B)
        resolved_equation.to_edge(DOWN, buff=0.38)

        content = VGroup(
            header,
            feature_rule_group,
            matrix_equation,
            resolved_equation,
        ).arrange(DOWN, buff=0.26)
        content.move_to(ORIGIN).shift(DOWN * 0.05)
        return content

    def construct(self):
        title_slide_title = Text("ReLUs, explained quickly", font_size=54, color=BLUE_C).to_edge(
            UP,
            buff=0.5,
        )
        relu_title_graph = self.make_relu_title_graph()

        self.play(FadeIn(title_slide_title), FadeIn(relu_title_graph), run_time=0.9)
        self.next_slide()

        title = Text("Problem: Study Time vs. GPA", font_size=22).to_edge(UP, buff=0.12)

        axes = self.make_axes()
        axis_labels = self.make_axis_labels(axes)
        scatter = self.make_scatter(axes)
        plot_group = VGroup(axes, axis_labels, scatter)

        # Slide 1: centered scatter plot.
        self.play(
            FadeOut(title_slide_title),
            FadeOut(relu_title_graph),
            FadeIn(title),
            FadeIn(axes),
            FadeIn(axis_labels),
            LaggedStart(*(FadeIn(dot, scale=0.7) for dot in scatter), lag_ratio=0.07),
        )
        self.next_slide()

        # Slide 2: least-squares line and residual squares.
        regression_line = self.make_regression_line(axes)
        line_label = MathTex(r"\hat y = Xw+b", font_size=32, color=YELLOW_C)
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

        # Slide 3: remove squares and graph, then center the Xw + b formulation.
        plot_group.add(regression_line, line_label)
        formula_panel = self.make_formula_panel()

        self.play(
            FadeOut(residual_squares),
            FadeOut(residual_segments),
            FadeOut(least_squares_note),
            FadeOut(plot_group),
            run_time=1.25,
        )
        self.play(FadeIn(formula_panel))
        self.next_slide()

        # Slide 4: expand y = Xw + b into one matrix multiplication over all points.
        matrix_region = Rectangle(width=11.4, height=5.55)
        matrix_region.move_to(ORIGIN).shift(DOWN * 0.12)
        matrix_content = self.make_matrix_panel_content(matrix_region)

        self.play(FadeOut(formula_panel), run_time=0.55)
        self.play(FadeIn(matrix_content), run_time=0.9)
        self.next_slide()

        # Slide 5: fixed data terms dim; w and b remain the trainable parameters.
        matrix_equation = matrix_content[3]
        matrix_labels = matrix_content[4]
        fixed_terms = VGroup(matrix_equation[0], matrix_equation[2], matrix_labels[0], matrix_labels[1])
        trainable_terms = VGroup(matrix_equation[3], matrix_equation[5], matrix_labels[2])

        self.play(fixed_terms.animate.set_opacity(0.28), trainable_terms.animate.set_opacity(1.0))
        self.next_slide()

        # Slide 6: switch to a U-shaped Temperature vs. Energy Consumption problem.
        temperature_title = Text("Problem: Temperature vs. Energy Consumption", font_size=22).to_edge(
            UP,
            buff=0.12,
        )
        temperature_axes = self.make_temperature_axes()
        temperature_axis_labels = self.make_temperature_axis_labels(temperature_axes)
        temperature_scatter = self.make_temperature_scatter(temperature_axes)
        temperature_curve = self.make_temperature_curve(temperature_axes)
        temperature_group = VGroup(
            temperature_axes,
            temperature_axis_labels,
            temperature_scatter,
            temperature_curve,
        )
        quadratic_label = MathTex(r"\text{U-shaped relationship}", font_size=32, color=GREEN_B)
        quadratic_label.next_to(temperature_axes.c2p(22, energy_curve(22)), UP, buff=0.3)

        self.play(
            FadeOut(matrix_content),
            FadeOut(title),
            FadeIn(temperature_title),
            run_time=0.9,
        )
        self.play(FadeIn(temperature_axes), FadeIn(temperature_axis_labels))
        self.play(
            LaggedStart(*(FadeIn(dot, scale=0.7) for dot in temperature_scatter), lag_ratio=0.05),
            Create(temperature_curve),
            FadeIn(quadratic_label),
        )
        self.next_slide()

        # Slide 7: the old linear formulation cannot bend to match the quadratic curve.
        linear_attempt = self.make_temperature_linear_attempt(temperature_axes)
        temperature_residuals = self.make_temperature_residuals(temperature_axes)
        attempt_equation = MathTex(r"\hat y = Xw + b", font_size=44, color=YELLOW_C)
        attempt_caption = Text("one slope, no turn", font_size=22, color=RED_B)
        attempt_group = VGroup(attempt_equation, attempt_caption).arrange(DOWN, buff=0.18)
        attempt_group.to_edge(RIGHT, buff=0.55).shift(UP * 1.25)
        failure_note = Text(
            "A straight line misses the low middle and high extremes",
            font_size=24,
            color=RED_B,
        ).to_edge(DOWN)

        self.play(
            Create(linear_attempt),
            FadeIn(attempt_group),
            LaggedStart(*(Create(segment) for segment in temperature_residuals), lag_ratio=0.035),
            FadeIn(failure_note, shift=UP * 0.15)
        )
        self.next_slide()

        # Slide 8: fade the graph away and center the linear model as the object to fix.
        self.play(
            FadeOut(temperature_group),
            FadeOut(quadratic_label),
            FadeOut(linear_attempt),
            FadeOut(temperature_residuals),
            FadeOut(failure_note),
            FadeOut(temperature_title),
            FadeOut(attempt_caption),
            attempt_equation.animate.scale(1.45).move_to(ORIGIN),
            run_time=1.25,
        )
        self.next_slide()

        # Slide 9: rewrite the same one-feature model as a matrix equation.
        quadratic_matrix_content = self.make_quadratic_matrix_content()
        self.play(FadeOut(attempt_equation), FadeIn(quadratic_matrix_content), run_time=0.9)
        self.next_slide()

        # Slide 10: one-step transition from the one-feature matrix to the augmented matrix.
        quadratic_augmentation_content = self.make_quadratic_augmentation_content()
        self.play(
            ReplacementTransform(quadratic_matrix_content, quadratic_augmentation_content),
            run_time=1.0,
        )
        quadratic_augmented_preview = quadratic_augmentation_content
        resolved_equation = quadratic_augmentation_content[3]
        self.next_slide()

        # Slide 11: return to the graph and show the quadratic feature fit working.
        fit_title = Text("Quadratic feature fit", font_size=22).to_edge(UP, buff=0.12)
        fit_axes = self.make_temperature_axes()
        fit_axes.scale(0.86).shift(UP * 0.45)
        fit_axis_labels = self.make_temperature_axis_labels(fit_axes)
        fit_scatter = self.make_temperature_scatter(fit_axes)
        quadratic_fit = self.make_temperature_quadratic_fit(fit_axes)
        fit_note = Text("Now the fitted curve can bend", font_size=24, color=GREEN_B)
        fit_note.to_edge(RIGHT, buff=0.55).shift(UP * 1.2)
        fit_graph_group = VGroup(fit_title, fit_axes, fit_axis_labels, fit_scatter, quadratic_fit, fit_note)

        self.play(
            FadeOut(VGroup(*quadratic_augmented_preview[:-1])),
            FadeIn(fit_title),
            FadeIn(fit_axes),
            FadeIn(fit_axis_labels),
            LaggedStart(*(FadeIn(dot, scale=0.7) for dot in fit_scatter), lag_ratio=0.04),
            run_time=1.0,
        )
        self.play(Create(quadratic_fit), FadeIn(fit_note, shift=LEFT * 0.15), run_time=0.9)
        self.next_slide()

        # Slide 12: return to the formulation and identify the full design matrix.
        design_content = self.make_quadratic_augmentation_content()
        design_matrix_equation = design_content[2]
        design_matrix = design_matrix_equation[2]
        design_matrix_label = Text("Design Matrix", font_size=30, color=BLUE_C)
        design_matrix_label.next_to(design_matrix, DOWN, buff=0.16)
        design_matrix_label.add_background_rectangle(color=BLACK, opacity=0.85, buff=0.1)
        design_dim_terms = VGroup(
            design_content[0],
            design_content[1],
            design_matrix_equation[0],
            design_matrix_equation[1],
            design_matrix_equation[3],
            design_matrix_equation[4],
            design_matrix_equation[5],
            design_content[3],
        )

        self.play(
            FadeOut(fit_graph_group),
            FadeOut(resolved_equation),
            FadeIn(design_content),
            run_time=0.9,
        )
        self.play(
            design_dim_terms.animate.set_opacity(0.22),
            design_matrix.animate.set_opacity(1.0),
            FadeIn(design_matrix_label, shift=UP * 0.12),
            run_time=0.75,
        )
        self.next_slide()

        # Slide 13: isolate the engineered squared column and its matching weight.
        squared_terms = VGroup(
            design_matrix_equation.get_part_by_tex(r"x_1^2"),
            design_matrix_equation.get_part_by_tex(r"x_2^2"),
        )
        w2_terms = VGroup(design_matrix_equation.get_part_by_tex(r"w_2"))
        feature_column_box = SurroundingRectangle(squared_terms, buff=0.16, color=GREEN_B)
        w2_box = SurroundingRectangle(w2_terms, buff=0.14, color=GREEN_B)
        feature_engineering_label = Text("Feature Engineering", font_size=30, color=GREEN_B)
        feature_engineering_label.next_to(VGroup(feature_column_box, w2_box), DOWN, buff=0.35)
        feature_engineering_label.add_background_rectangle(color=BLACK, opacity=0.85, buff=0.1)

        self.play(
            FadeOut(design_matrix_label),
            design_content.animate.set_opacity(0.18),
            run_time=0.65,
        )
        self.play(
            squared_terms.animate.set_opacity(1.0),
            w2_terms.animate.set_opacity(1.0),
            FadeIn(feature_column_box),
            FadeIn(w2_box),
            FadeIn(feature_engineering_label, shift=UP * 0.12),
            run_time=0.75,
        )
        self.next_slide()
