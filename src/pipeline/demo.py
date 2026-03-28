"""
Gradio interactive demo for the NeuroMusic pipeline.

Launch with:
    python -m src.pipeline.demo
    python -m src.pipeline.demo --config configs/default.yaml
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from src.biosignal.classifier import EmotionResult
from src.bridge.prompt_generator import EmotionState, THERAPEUTIC_TEMPLATES
from src.pipeline.pipeline import TherapeuticSoundtrackPipeline
from src.utils.io import load_config, add_config_arg
from evaluation.evaluate import generate_baseline_prompt

logger = logging.getLogger(__name__)

QUADRANT_INFO = {
    "HVHA": {"label": "Excited / Agitated", "color": "#e74c3c", "therapy": "Grounding, serene music"},
    "HVLA": {"label": "Content / Relaxed",  "color": "#2ecc71", "therapy": "Maintain with gentle engagement"},
    "LVHA": {"label": "Anxious / Stressed",  "color": "#e67e22", "therapy": "Calm, soothing music"},
    "LVLA": {"label": "Sad / Depressed",     "color": "#3498db", "therapy": "Uplifting, gently energizing music"},
}

DEFAULT_EXAMPLE_SAMPLES = [42, 100, 5000, 50000]


def _make_emotion_plot(valence: float, arousal: float, quadrant: str):
    """Create a valence-arousal scatter plot showing detected emotion and therapeutic target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.axhline(0.5, color="#bdc3c7", linestyle="--", linewidth=1)
    ax.axvline(0.5, color="#bdc3c7", linestyle="--", linewidth=1)

    colors = [
        ("#e67e22", 0.08),  # LVHA top-left
        ("#e74c3c", 0.08),  # HVHA top-right
        ("#3498db", 0.08),  # LVLA bottom-left
        ("#2ecc71", 0.08),  # HVLA bottom-right
    ]
    regions = [(0, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5),
               (0, 0, 0.5, 0.5), (0.5, 0, 0.5, 0.5)]
    labels_pos = [
        (0.25, 0.75, "Anxious\n(LVHA)"),
        (0.75, 0.75, "Excited\n(HVHA)"),
        (0.25, 0.25, "Sad\n(LVLA)"),
        (0.75, 0.25, "Content\n(HVLA)"),
    ]

    for (x, y, w, h), (c, a) in zip(regions, colors):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                              facecolor=c, alpha=a, edgecolor="none"))

    for lx, ly, lt in labels_pos:
        ax.text(lx, ly, lt, ha="center", va="center", fontsize=9, color="#7f8c8d", style="italic")

    # Detected emotion
    ax.scatter([valence], [arousal], s=200, c="#e74c3c", zorder=5, edgecolors="white", linewidths=2)
    ax.annotate("Detected", (valence, arousal), textcoords="offset points",
                xytext=(12, 12), fontsize=10, fontweight="bold", color="#e74c3c")

    # Therapeutic target (inverted)
    target_v = 1.0 - valence
    target_a = 1.0 - arousal
    ax.scatter([target_v], [target_a], s=200, c="#2ecc71", marker="*", zorder=5,
               edgecolors="white", linewidths=1.5)
    ax.annotate("Target", (target_v, target_a), textcoords="offset points",
                xytext=(12, -12), fontsize=10, fontweight="bold", color="#2ecc71")

    ax.annotate("", xy=(target_v, target_a), xytext=(valence, arousal),
                arrowprops=dict(arrowstyle="->", color="#95a5a6", lw=1.5, ls="--"))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Valence (negative → positive)", fontsize=11)
    ax.set_ylabel("Arousal (calm → excited)", fontsize=11)
    ax.set_title("Emotion Space — Affective Pivot", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def _format_emotion_card(valence: float, arousal: float, quadrant: str, confidence: float) -> str:
    info = QUADRANT_INFO.get(quadrant, {})
    v_bar = "█" * int(valence * 20) + "░" * (20 - int(valence * 20))
    a_bar = "█" * int(arousal * 20) + "░" * (20 - int(arousal * 20))
    return (
        f"### Detected Emotion: **{quadrant}** — {info.get('label', '')}\n\n"
        f"| Dimension | Value | |\n"
        f"|---|---|---|\n"
        f"| Valence | {valence:.3f} | {v_bar} |\n"
        f"| Arousal | {arousal:.3f} | {a_bar} |\n"
        f"| Confidence | {confidence:.3f} | |\n\n"
        f"**Therapeutic strategy:** {info.get('therapy', 'N/A')}"
    )


def create_demo(
    pipeline: TherapeuticSoundtrackPipeline,
    data_dir: str = "data/raw/dreamer",
    example_samples: list[int] | None = None,
):
    """Build the Gradio interface."""
    if example_samples is None:
        example_samples = DEFAULT_EXAMPLE_SAMPLES

    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio is required for the demo. Install: pip install gradio")

    has_dreamer = Path(data_dir, "arousal", "X.npy").exists()

    def run_from_file(file):
        if file is None:
            return "Upload a .npy file first.", "", None, None

        path = Path(file.name)
        if path.suffix == ".npy":
            eeg_data = np.load(path)
        elif path.suffix == ".npz":
            npz = np.load(path)
            eeg_data = npz[list(npz.keys())[0]]
        else:
            return f"Unsupported format: {path.suffix}. Use .npy (14, 256).", "", None, None

        if eeg_data.shape[0] != 14:
            return f"Expected 14 channels, got shape {eeg_data.shape}.", "", None, None

        result = pipeline.generate(eeg_data)
        e = result.emotion
        card = _format_emotion_card(e.valence, e.arousal, e.quadrant, e.confidence)
        fig = _make_emotion_plot(e.valence, e.arousal, e.quadrant)
        return card, result.therapeutic_prompt, (result.sample_rate, result.audio), fig

    def run_from_dreamer(sample_idx):
        if not has_dreamer:
            return "DREAMER data not found. Run: python data/scripts/download_dreamer.py", "", None, None

        X = np.load(Path(data_dir, "arousal", "X.npy"), mmap_mode="r")
        y_a = np.load(Path(data_dir, "arousal", "y.npy"))
        y_v = np.load(Path(data_dir, "valence", "y.npy"))

        idx = int(sample_idx)
        if idx >= len(X):
            return f"Index {idx} out of range (max: {len(X)-1}).", "", None, None

        eeg_data = X[idx].copy()
        gt_a = "high" if y_a[idx] == 1 else "low"
        gt_v = "high" if y_v[idx] == 1 else "low"

        result = pipeline.generate(eeg_data)
        e = result.emotion

        card = _format_emotion_card(e.valence, e.arousal, e.quadrant, e.confidence)
        card += f"\n\n---\n**Ground truth:** valence={gt_v}, arousal={gt_a}"
        fig = _make_emotion_plot(e.valence, e.arousal, e.quadrant)
        return card, result.therapeutic_prompt, (result.sample_rate, result.audio), fig

    def run_from_sliders(valence, arousal):
        v_class = "high" if valence >= 0.5 else "low"
        a_class = "high" if arousal >= 0.5 else "low"
        quadrant = f"{'H' if v_class == 'high' else 'L'}V{'H' if a_class == 'high' else 'L'}A"

        emotion_state = EmotionState(valence=valence, arousal=arousal, label=quadrant)
        prompt = pipeline.prompt_generator.generate(emotion_state)
        audio = pipeline.music_generator.generate(prompt)
        sr = pipeline.music_generator.sample_rate

        card = _format_emotion_card(valence, arousal, quadrant, 1.0)
        fig = _make_emotion_plot(valence, arousal, quadrant)
        return card, prompt, (sr, audio), fig

    def run_comparison(valence, arousal):
        """Generate therapeutic vs non-inverted vs baseline audio for comparison."""
        v_class = "high" if valence >= 0.5 else "low"
        a_class = "high" if arousal >= 0.5 else "low"
        quadrant = f"{'H' if v_class == 'high' else 'L'}V{'H' if a_class == 'high' else 'L'}A"

        emotion_state = EmotionState(valence=valence, arousal=arousal, label=quadrant)
        therapeutic_prompt = pipeline.prompt_generator.generate(emotion_state)
        noninv_prompt = generate_baseline_prompt("non_inverted", quadrant)
        baseline_prompt = generate_baseline_prompt("fixed_calm")

        therapeutic_audio = pipeline.music_generator.generate(therapeutic_prompt)
        noninv_audio = pipeline.music_generator.generate(noninv_prompt)
        baseline_audio = pipeline.music_generator.generate(baseline_prompt)
        sr = pipeline.music_generator.sample_rate

        card = _format_emotion_card(valence, arousal, quadrant, 1.0)
        fig = _make_emotion_plot(valence, arousal, quadrant)

        prompts_md = (
            f"**Therapeutic (ours):** {therapeutic_prompt}\n\n"
            f"**Non-inverted:** {noninv_prompt}\n\n"
            f"**Baseline:** {baseline_prompt}"
        )

        return (
            card, fig, prompts_md,
            (sr, therapeutic_audio),
            (sr, noninv_audio),
            (sr, baseline_audio),
        )

    with gr.Blocks(title="NeuroMusic", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# NeuroMusic\n"
            "**EEG → Emotion Classification → Affective Pivot → Therapeutic Music**\n\n"
            "Detects emotional state from brain signals and generates music that "
            "therapeutically inverts the detected emotion — anxious listeners hear "
            "calming music, low-energy listeners hear uplifting music."
        )

        with gr.Tab("Upload EEG"):
            gr.Markdown(
                "Upload a `.npy` file with shape `(14, 256)` — 14 EEG channels, "
                "2 seconds at 128 Hz (DREAMER format)."
            )
            eeg_file = gr.File(label="EEG Recording (.npy)", file_types=[".npy", ".npz"])
            btn_file = gr.Button("Generate Therapeutic Music", variant="primary")

            with gr.Row():
                emotion_out_file = gr.Markdown(label="Emotion")
                plot_out_file = gr.Plot(label="Emotion Space")
            prompt_out_file = gr.Textbox(label="Therapeutic Prompt", lines=5)
            audio_out_file = gr.Audio(label="Generated Music", type="numpy")

            btn_file.click(
                run_from_file, inputs=[eeg_file],
                outputs=[emotion_out_file, prompt_out_file, audio_out_file, plot_out_file],
            )

        if has_dreamer:
            with gr.Tab("DREAMER Samples"):
                gr.Markdown(
                    "Select a sample index from the DREAMER dataset to run the full pipeline. "
                    "Ground-truth labels are shown for comparison."
                )
                sample_idx = gr.Number(
                    value=example_samples[0] if example_samples else 42,
                    label="Sample Index",
                    info=f"Try: {', '.join(str(s) for s in example_samples)}",
                    precision=0,
                )
                btn_dreamer = gr.Button("Run Full Pipeline", variant="primary")

                with gr.Row():
                    emotion_out_dreamer = gr.Markdown(label="Emotion")
                    plot_out_dreamer = gr.Plot(label="Emotion Space")
                prompt_out_dreamer = gr.Textbox(label="Therapeutic Prompt", lines=5)
                audio_out_dreamer = gr.Audio(label="Generated Music", type="numpy")

                btn_dreamer.click(
                    run_from_dreamer, inputs=[sample_idx],
                    outputs=[emotion_out_dreamer, prompt_out_dreamer, audio_out_dreamer, plot_out_dreamer],
                )

        with gr.Tab("Manual Emotion"):
            gr.Markdown(
                "Set valence and arousal manually to test the prompt generation "
                "and music generation without EEG data."
            )
            with gr.Row():
                valence_slider = gr.Slider(0.0, 1.0, value=0.3, step=0.05,
                                           label="Valence (0 = negative, 1 = positive)")
                arousal_slider = gr.Slider(0.0, 1.0, value=0.7, step=0.05,
                                           label="Arousal (0 = calm, 1 = excited)")
            btn_slider = gr.Button("Generate Therapeutic Music", variant="primary")

            with gr.Row():
                emotion_out_slider = gr.Markdown(label="Emotion")
                plot_out_slider = gr.Plot(label="Emotion Space")
            prompt_out_slider = gr.Textbox(label="Therapeutic Prompt", lines=5)
            audio_out_slider = gr.Audio(label="Generated Music", type="numpy")

            btn_slider.click(
                run_from_sliders, inputs=[valence_slider, arousal_slider],
                outputs=[emotion_out_slider, prompt_out_slider, audio_out_slider, plot_out_slider],
            )

        with gr.Tab("Compare Conditions"):
            gr.Markdown(
                "### Side-by-Side Comparison\n"
                "Compare our **therapeutic** (affective pivot) audio against baselines "
                "to hear the difference. Set an emotion and generate all three conditions."
            )
            with gr.Row():
                cmp_valence = gr.Slider(0.0, 1.0, value=0.2, step=0.05,
                                        label="Valence (0 = negative, 1 = positive)")
                cmp_arousal = gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                        label="Arousal (0 = calm, 1 = excited)")
            btn_compare = gr.Button("Generate All Conditions", variant="primary")

            with gr.Row():
                cmp_emotion = gr.Markdown(label="Emotion")
                cmp_plot = gr.Plot(label="Emotion Space")
            cmp_prompts = gr.Markdown(label="Prompts")

            gr.Markdown("#### Therapeutic (Ours) — Inverts the detected emotion")
            cmp_audio_ther = gr.Audio(label="Therapeutic", type="numpy")
            gr.Markdown("#### Non-Inverted — Matches the detected emotion (no therapy)")
            cmp_audio_noninv = gr.Audio(label="Non-Inverted", type="numpy")
            gr.Markdown("#### Baseline — Generic calm music")
            cmp_audio_base = gr.Audio(label="Baseline", type="numpy")

            btn_compare.click(
                run_comparison, inputs=[cmp_valence, cmp_arousal],
                outputs=[cmp_emotion, cmp_plot, cmp_prompts,
                         cmp_audio_ther, cmp_audio_noninv, cmp_audio_base],
            )

        gr.Markdown(
            "---\n"
            "*NeuroMusic — MIT 6.S985 Multimodal AI, Spring 2026. "
            "Linrui Ma, Grace Yuan, Aimee Yu.*"
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="NeuroMusic — Gradio Demo")
    add_config_arg(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    demo_cfg = cfg.get("demo", {})

    pipeline = TherapeuticSoundtrackPipeline.from_config(args.config)
    demo = create_demo(
        pipeline,
        data_dir=paths.get("data_dir", "data/raw/dreamer"),
        example_samples=demo_cfg.get("example_samples", DEFAULT_EXAMPLE_SAMPLES),
    )
    demo.launch(
        share=demo_cfg.get("share", False),
        server_port=demo_cfg.get("port", 7860),
    )


if __name__ == "__main__":
    main()
