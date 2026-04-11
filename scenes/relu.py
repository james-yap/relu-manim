from manim import *


class ArtificialNeuron(Scene):
    def construct(self):
        # --- Color Palette ---
        input_color = GREEN_C
        bias_color = RED_C
        sum_color = YELLOW_C
        act_color = BLUE_C
        out_color = TEAL_C

        # --- Nodes ---

        # 1. Inputs (Left side)
        x1 = VGroup(
            Circle(color=input_color, radius=0.5, fill_opacity=0.1), MathTex("X_1")
        )
        x2 = VGroup(
            Circle(color=input_color, radius=0.5, fill_opacity=0.1), MathTex("X_2")
        )
        dots = MathTex(r"\vdots").scale(1.5)
        xn = VGroup(
            Circle(color=input_color, radius=0.5, fill_opacity=0.1), MathTex("X_n")
        )

        inputs = VGroup(x1, x2, dots, xn).arrange(DOWN, buff=0.5).shift(LEFT * 5)

        # 2. Summation (Center)
        sum_circle = Circle(color=sum_color, radius=0.8, fill_opacity=0.1)
        sum_symbol = MathTex(r"\sum").scale(1).set_color(sum_color)
        sum_node = VGroup(sum_circle, sum_symbol).shift(LEFT * 1)
        sum_label = Text("Summation", font_size=24).next_to(sum_node, DOWN, buff=0.4)

        # 3. Bias (Top)
        bias_circle = Circle(color=bias_color, radius=0.5, fill_opacity=0.1)
        bias_symbol = MathTex("b").set_color(bias_color)
        bias_node = VGroup(bias_circle, bias_symbol).next_to(sum_node, UP, buff=1.5)
        bias_label = Text("bias", font_size=24).next_to(bias_node, UP, buff=0.2)

        # 4. Activation Function (Right)
        act_circle = Circle(color=act_color, radius=0.8, fill_opacity=0.1)

        # Creating the small ReLU graph inside the activation circle
        relu_lines = VGroup(
            Line(
                act_circle.get_center() + LEFT * 0.3,
                act_circle.get_center(),
                color=RED,
                stroke_width=4,
            ),
            Line(
                act_circle.get_center(),
                act_circle.get_center() + UP * 0.3 + RIGHT * 0.3,
                color=RED,
                stroke_width=4,
            ),
        )
        act_node = VGroup(act_circle, relu_lines).shift(RIGHT * 2.5)
        act_label = Text("Activation\nFunction", font_size=24, line_spacing=1).next_to(
            act_node, DOWN, buff=0.4
        )

        # 5. Output (Far Right)
        out_text = Text("Output", font_size=32).shift(RIGHT * 6)

        # --- Connections & Weights ---

        # Arrows from inputs to sum
        w1_arrow = Arrow(
            x1.get_right(),
            sum_node.get_left(),
            buff=0.2,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.08,
        )
        w1_label = (
            MathTex("w_1")
            .move_to(w1_arrow)
            .add_background_rectangle(color=BLACK, opacity=1, buff=0.1)
        )

        w2_arrow = Arrow(
            x2.get_right(),
            sum_node.get_left(),
            buff=0.2,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.08,
        )
        w2_label = (
            MathTex("w_2")
            .move_to(w2_arrow)
            .add_background_rectangle(color=BLACK, opacity=1, buff=0.1)
        )

        wn_arrow = Arrow(
            xn.get_right(),
            sum_node.get_left(),
            buff=0.2,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.08,
        )
        wn_label = (
            MathTex("w_n")
            .move_to(wn_arrow)
            .add_background_rectangle(color=BLACK, opacity=1, buff=0.1)
        )

        # Arrow from bias to sum
        bias_arrow = Arrow(
            bias_node.get_bottom(), sum_node.get_top(), buff=0.2, stroke_width=2
        )

        # Arrow from sum to activation
        sum_to_act_arrow = Arrow(
            sum_node.get_right(), act_node.get_left(), buff=0.2, stroke_width=3
        )

        # Arrow from activation to output
        act_to_out_arrow = Arrow(
            act_node.get_right(), out_text.get_left(), buff=0.2, stroke_width=3
        )

        # --- Animation Sequence ---
        self.play(FadeIn(inputs), FadeIn(bias_node), FadeIn(bias_label))

        self.play(
            Create(w1_arrow),
            FadeIn(w1_label),
            Create(w2_arrow),
            FadeIn(w2_label),
            Create(wn_arrow),
            FadeIn(wn_label),
            Create(bias_arrow),
        )

        self.play(FadeIn(sum_node), Write(sum_label))
        self.play(Create(sum_to_act_arrow))

        self.play(FadeIn(act_node), Write(act_label))
        self.play(Create(act_to_out_arrow))

        self.play(FadeIn(out_text))

        self.wait(2)
