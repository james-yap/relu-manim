from manim import *


class TinyNeuralNetwork(Scene):
    def construct(self):
        # 1. Define network architecture: 2 inputs, 3 hidden nodes, 1 output
        layer_sizes = [2, 3, 1]
        layers = VGroup()

        # Create the nodes (circles)
        for i, size in enumerate(layer_sizes):
            layer = VGroup(*[Circle(radius=0.3, color=WHITE) for _ in range(size)])
            layer.arrange(DOWN, buff=0.5)
            layers.add(layer)

        # Arrange the layers horizontally
        layers.arrange(RIGHT, buff=2)

        # Add labels to inputs and output
        input_labels = VGroup(MathTex("x_1"), MathTex("x_2"))
        for i, node in enumerate(layers[0]):
            input_labels[i].move_to(node.get_center())

        output_label = MathTex("\\hat{y}").move_to(layers[2][0].get_center())

        # 2. Create edges (connections)
        edges = VGroup()
        for i in range(len(layers) - 1):
            layer_edges = VGroup()
            for node1 in layers[i]:
                for node2 in layers[i + 1]:
                    # Draw a line from the right side of the previous node to the left side of the next
                    edge = Line(node1.get_right(), node2.get_left(), stroke_opacity=0.3)
                    layer_edges.add(edge)
            edges.add(layer_edges)

        # 3. Animate the creation of the network
        self.play(FadeIn(layers), FadeIn(input_labels), FadeIn(output_label))
        self.play(Create(edges))
        self.wait(1)

        # 4. Animate the forward pass (Prediction)

        # Step A: Data flows from Input to Hidden Layer
        dots_layer1 = VGroup()
        for edge in edges[0]:
            dot = Dot(color=YELLOW).move_to(edge.get_start())
            dots_layer1.add(dot)

        self.play(
            LaggedStart(
                *[MoveAlongPath(dot, edge) for dot, edge in zip(dots_layer1, edges[0])],
                lag_ratio=0.1,
                run_time=1.5,
            )
        )

        # Highlight hidden nodes to show activation
        self.play(
            *[node.animate.set_fill(YELLOW, opacity=0.5) for node in layers[1]],
            run_time=0.5,
        )
        self.play(FadeOut(dots_layer1))

        # Step B: Data flows from Hidden Layer to Output
        dots_layer2 = VGroup()
        for edge in edges[1]:
            dot = Dot(color=ORANGE).move_to(edge.get_start())
            dots_layer2.add(dot)

        self.play(
            LaggedStart(
                *[MoveAlongPath(dot, edge) for dot, edge in zip(dots_layer2, edges[1])],
                lag_ratio=0.1,
                run_time=1.5,
            )
        )

        # Highlight output node to show final prediction
        self.play(layers[2][0].animate.set_fill(ORANGE, opacity=0.8), run_time=0.5)
        self.play(FadeOut(dots_layer2))

        # 5. Final text reveal
        pred_text = Text("Prediction Complete", font_size=36, color=GREEN).next_to(
            layers, UP, buff=1
        )
        self.play(Write(pred_text))
        self.wait(2)
