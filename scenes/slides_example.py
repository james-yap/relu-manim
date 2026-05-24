from manim import *
from manim_slides import Slide


class ReLUSlidesExample(Slide):
    """Small manim-slides presentation example.

    Render the video frames with:
        uv run manim-slides render scenes/slides_example.py ReLUSlidesExample

    Then present with:
        uv run manim-slides present ReLUSlidesExample
    """

    def construct(self):
        title = Text("ReLU in one slide deck", font_size=44)
        subtitle = Text("Built with manim-slides", font_size=28, color=BLUE_C)
        subtitle.next_to(title, DOWN, buff=0.35)

        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.2))
        self.next_slide()

        formula = MathTex(r"f(x)=\max(0,x)", font_size=60)
        note = Text("negative inputs become 0", font_size=28, color=YELLOW_C)
        note.next_to(formula, DOWN, buff=0.5)

        self.play(FadeOut(title), FadeOut(subtitle), Write(formula))
        self.next_slide()
        self.play(FadeIn(note, shift=UP * 0.2))
        self.next_slide()

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1, 3, 1],
            x_length=6,
            y_length=4,
            tips=False,
        ).to_edge(DOWN)
        labels = axes.get_axis_labels(MathTex("x"), MathTex("f(x)"))
        relu_graph = axes.plot(lambda x: max(0, x), x_range=[-3, 3], color=GREEN_C)
        graph_label = MathTex(r"\max(0,x)", color=GREEN_C).next_to(
            axes.c2p(2, 2), UP
        )

        graph_group = VGroup(axes, labels, relu_graph, graph_label)
        self.play(FadeOut(note), formula.animate.to_edge(UP), Create(axes), FadeIn(labels))
        self.play(Create(relu_graph), Write(graph_label))
        self.next_slide()

        highlight = Dot(axes.c2p(-1.5, 0), color=RED_C)
        explanation = Text("For x < 0, ReLU outputs 0", font_size=28, color=RED_C)
        explanation.next_to(formula, DOWN, buff=0.4)

        self.play(FadeIn(highlight), Write(explanation))
        self.play(highlight.animate.move_to(axes.c2p(2, 2)))
        self.play(Transform(explanation, Text("For x > 0, ReLU is linear", font_size=28, color=GREEN_C).move_to(explanation)))
        self.next_slide()

        self.play(FadeOut(VGroup(formula, graph_group, highlight, explanation)))
        self.play(Write(Text("Done!", font_size=56, color=GREEN)))
        self.next_slide()
