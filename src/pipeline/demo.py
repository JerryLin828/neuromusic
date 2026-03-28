"""
Gradio interactive demo for the Therapeutic Soundtrack Pipeline.

Launch with:
    python -m src.pipeline.demo
    python -m src.pipeline.demo --config configs/default.yaml

Provides two modes:
  1. Upload EEG file (.npy, 14 channels x 256 timepoints) → full pipeline
  2. Manual emotion sliders → prompt → generate (bypasses EEG classification)
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from src.biosignal.classifier import EmotionResult
from src.bridge.prompt_generator import EmotionState
from src.pipeline.pipeline import TherapeuticSoundtrackPipeline

logger = logging.getLogger(__name__)


def create_demo(pipeline: TherapeuticSoundtrackPipeline):
    """Build the Gradio interface."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio is required for the demo. Install: pip install gradio")

    def run_from_file(file):
        """Run full pipeline on an uploaded EEG file."""
        if file is None:
            return "No file uploaded", "", None

        path = Path(file.name)
        if path.suffix == ".npy":
            eeg_data = np.load(path)
        elif path.suffix == ".npz":
            npz = np.load(path)
            eeg_data = npz[list(npz.keys())[0]]
        else:
            return f"Unsupported format: {path.suffix}. Use .npy (14, 256).", "", None

        if eeg_data.shape[0] != 14:
            return (
                f"Expected 14 channels, got shape {eeg_data.shape}.\n"
                f"DREAMER uses 14-channel Emotiv EPOC EEG.",
                "", None,
            )

        result = pipeline.generate(eeg_data)
        emotion_text = (
            f"Valence: {result.emotion.valence:.3f} ({result.emotion.valence_class})\n"
            f"Arousal: {result.emotion.arousal:.3f} ({result.emotion.arousal_class})\n"
            f"Quadrant: {result.emotion.quadrant}\n"
            f"Confidence: {result.emotion.confidence:.3f}"
        )
        return emotion_text, result.therapeutic_prompt, (result.sample_rate, result.audio)

    def run_from_sliders(valence, arousal):
        """Run prompt generation + music from manual emotion input."""
        v_class = "high" if valence >= 0.5 else "low"
        a_class = "high" if arousal >= 0.5 else "low"
        quadrant = f"{'H' if v_class == 'high' else 'L'}V{'H' if a_class == 'high' else 'L'}A"

        emotion_state = EmotionState(valence=valence, arousal=arousal, label=quadrant)
        prompt = pipeline.prompt_generator.generate(emotion_state)
        audio = pipeline.music_generator.generate(prompt)
        sr = pipeline.music_generator.sample_rate

        emotion_text = (
            f"Valence: {valence:.2f} ({v_class})\n"
            f"Arousal: {arousal:.2f} ({a_class})\n"
            f"Quadrant: {quadrant}"
        )
        return emotion_text, prompt, (sr, audio)

    with gr.Blocks(title="Personalized Soundtrack Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Personalized Soundtrack Generator\n"
            "EEG → Emotion Classification → Therapeutic Music Generation\n\n"
            "**Dataset:** DREAMER (14-channel Emotiv EPOC, 128 Hz, 23 subjects)\n"
            "**Classifier:** TorchEEG TSCeption\n"
            "**Music:** MusicGen"
        )

        with gr.Tab("Upload EEG"):
            gr.Markdown(
                "Upload a `.npy` file with shape `(14, 256)` — 14 EEG channels, "
                "2 seconds at 128 Hz, matching DREAMER preprocessing. "
                "You can extract samples from the downloaded DREAMER dataset."
            )
            eeg_file = gr.File(label="EEG Recording (.npy)", file_types=[".npy", ".npz"])
            btn_file = gr.Button("Generate Music from EEG", variant="primary")

        with gr.Tab("Manual Emotion Input"):
            gr.Markdown(
                "Set valence and arousal manually to test the prompt generation "
                "and music generation stages without requiring EEG data.\n\n"
                "Values are probabilities of **high** class (0 = low, 1 = high)."
            )
            valence_slider = gr.Slider(0.0, 1.0, value=0.3, step=0.05,
                                       label="Valence (0=negative, 1=positive)")
            arousal_slider = gr.Slider(0.0, 1.0, value=0.7, step=0.05,
                                       label="Arousal (0=calm, 1=excited)")
            btn_slider = gr.Button("Generate Music from Emotion", variant="primary")

        gr.Markdown("---")
        gr.Markdown("### Results")

        with gr.Row():
            emotion_output = gr.Textbox(label="Detected Emotion", lines=4)
            prompt_output = gr.Textbox(label="Therapeutic Music Prompt", lines=4)

        audio_output = gr.Audio(label="Generated Music", type="numpy")

        btn_file.click(run_from_file, inputs=[eeg_file],
                      outputs=[emotion_output, prompt_output, audio_output])
        btn_slider.click(run_from_sliders, inputs=[valence_slider, arousal_slider],
                        outputs=[emotion_output, prompt_output, audio_output])

    return demo


def main():
    parser = argparse.ArgumentParser(description="Therapeutic Soundtrack Generator — Gradio Demo")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pipeline = TherapeuticSoundtrackPipeline.from_config(args.config)
    demo = create_demo(pipeline)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
