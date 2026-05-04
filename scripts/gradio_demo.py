from __future__ import annotations

import argparse
import sys
from threading import Lock
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cropvlm import CROP_CLASSES, load_cropvlm
from cropvlm.model import parse_class_names


DEFAULT_CLASSES_TEXT = "\n".join(CROP_CLASSES)


def build_demo(checkpoint: str, device: str | None, prompt_template: str, top_k: int) -> gr.Blocks:
    import gradio as gr

    classifier = load_cropvlm(
        checkpoint=checkpoint,
        class_names=CROP_CLASSES,
        device=device,
        prompt_template=prompt_template,
    )
    classifier_lock = Lock()
    current_classes = tuple(CROP_CLASSES)

    def classify(image: Image.Image, classes_text: str, top_k_value: int):
        if image is None:
            return {}, []
        nonlocal current_classes
        requested_classes = tuple(parse_class_names(classes_text))
        if not requested_classes:
            return {}, []
        with classifier_lock:
            if requested_classes != current_classes:
                classifier.set_classes(requested_classes)
                current_classes = requested_classes
            predictions = classifier.predict_with_scores(image, top_k=int(top_k_value))

        label_scores = {label: probability for label, probability, _ in predictions}
        score_text = "\n".join(
            f"{rank}. {label}: probability={probability:.6f}, cosine={cosine:.6f}"
            for rank, (label, probability, cosine) in enumerate(predictions, start=1)
        )
        return label_scores, score_text

    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    example_paths = [
        str(examples_dir / name)
        for name in ["cacao.png", "olive.png", "cauliflower.png", "sugarcane.png", "sunflower.png"]
        if (examples_dir / name).exists()
    ]

    with gr.Blocks(title="CropVLM Zero-Shot Demo") as demo:
        gr.Markdown("# CropVLM Zero-Shot Image Classification")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image")
                classes = gr.Textbox(
                    value=DEFAULT_CLASSES_TEXT,
                    lines=12,
                    label="Class names",
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=top_k,
                    step=1,
                    label="Top-k",
                )
                button = gr.Button("Classify", variant="primary")
            with gr.Column():
                label = gr.Label(num_top_classes=top_k, label="Predictions")
                score_text = gr.Textbox(
                    label="Scores",
                    lines=8,
                    interactive=False,
                )

        outputs = [label, score_text]
        button.click(classify, inputs=[image, classes, top_k_slider], outputs=outputs)
        classes.change(lambda: ({}, ""), outputs=outputs)

        if example_paths:
            gr.Examples(
                examples=[[path, DEFAULT_CLASSES_TEXT, top_k] for path in example_paths],
                inputs=[image, classes, top_k_slider],
                outputs=outputs,
                fn=classify,
                cache_examples=False,
            )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/CropVLM.pth")
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompt-template", default="{}")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_demo(
        checkpoint=args.checkpoint,
        device=args.device,
        prompt_template=args.prompt_template,
        top_k=args.top_k,
    )
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
