"""
Evaluation suite for the Therapeutic Soundtrack Pipeline.

Metrics:
  1. Emotion classification accuracy (if ground-truth labels available)
  2. CLAP score — semantic similarity between generated audio and target prompt
  3. Baseline comparisons (random prompt, non-inverted, fixed prompt)

Usage:
    python -m evaluation.evaluate --result-dir outputs
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Metric 1: Emotion Classification Accuracy
# ----------------------------------------------------------------

def evaluate_emotion_accuracy(
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate emotion classification performance.

    Args:
        predicted_labels: (N, 2) predicted [valence, arousal] as probabilities (0-1)
        true_labels: (N, 2) ground-truth [valence, arousal] as probabilities (0-1)
        threshold: Midpoint for binary classification (0.5 for DREAMER's probability scale)

    Returns:
        Dict with accuracy, F1, and per-class metrics.
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    results = {}
    for i, dim_name in enumerate(["valence", "arousal"]):
        pred_binary = (predicted_labels[:, i] >= threshold).astype(int)
        true_binary = (true_labels[:, i] >= threshold).astype(int)

        acc = accuracy_score(true_binary, pred_binary)
        f1 = f1_score(true_binary, pred_binary, average="binary")

        results[dim_name] = {
            "accuracy": float(acc),
            "f1_score": float(f1),
            "n_samples": int(len(pred_binary)),
            "report": classification_report(
                true_binary, pred_binary,
                target_names=["low", "high"],
                output_dict=True,
            ),
        }
        logger.info(f"  {dim_name}: accuracy={acc:.3f}, F1={f1:.3f}")

    # 4-class quadrant accuracy
    pred_quadrant = _to_quadrant(predicted_labels, threshold)
    true_quadrant = _to_quadrant(true_labels, threshold)
    quad_acc = accuracy_score(true_quadrant, pred_quadrant)
    results["quadrant"] = {
        "accuracy": float(quad_acc),
        "n_samples": int(len(pred_quadrant)),
    }
    logger.info(f"  quadrant: accuracy={quad_acc:.3f}")

    return results


def _to_quadrant(labels: np.ndarray, threshold: float = 0.5) -> list[str]:
    """Convert (N, 2) [valence, arousal] to quadrant labels."""
    quads = []
    for v, a in labels:
        vl = "H" if v >= threshold else "L"
        al = "H" if a >= threshold else "L"
        quads.append(f"{vl}V{al}A")
    return quads


# ----------------------------------------------------------------
# Metric 2: CLAP Score
# ----------------------------------------------------------------

def evaluate_clap_score(
    audio_paths: list[str | Path],
    target_prompts: list[str],
) -> dict:
    """
    Compute CLAP score: cosine similarity between audio and text embeddings.

    Requires: pip install laion-clap

    Args:
        audio_paths: Paths to generated audio WAV files.
        target_prompts: Corresponding therapeutic text prompts.

    Returns:
        Dict with mean, std, and per-sample CLAP scores.
    """
    try:
        import laion_clap
    except ImportError:
        logger.warning("laion-clap not installed. Skipping CLAP evaluation.")
        logger.warning("Install with: pip install laion-clap")
        return {"error": "laion-clap not installed"}

    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()

    audio_embeddings = model.get_audio_embedding_from_filelist(
        [str(p) for p in audio_paths], use_tensor=True,
    )
    text_embeddings = model.get_text_embedding(target_prompts, use_tensor=True)

    import torch
    audio_embeddings = torch.nn.functional.normalize(audio_embeddings, dim=-1)
    text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)

    scores = torch.sum(audio_embeddings * text_embeddings, dim=-1).detach().cpu().numpy()

    results = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "per_sample": scores.tolist(),
    }
    logger.info(f"  CLAP score: mean={results['mean']:.3f} ± {results['std']:.3f}")
    return results


# ----------------------------------------------------------------
# Baseline Comparisons
# ----------------------------------------------------------------

BASELINE_PROMPTS = {
    "random": "A lively jazz piece with saxophone and drums, energetic and improvisational",
    "fixed_calm": "Calm relaxing music",
    "fixed_generic": "A pleasant piece of background music",
}


def generate_baseline_prompt(baseline_type: str, emotion_quadrant: str = None) -> str:
    """Generate a baseline prompt for comparison."""
    if baseline_type == "random":
        return BASELINE_PROMPTS["random"]
    elif baseline_type == "fixed_calm":
        return BASELINE_PROMPTS["fixed_calm"]
    elif baseline_type == "fixed_generic":
        return BASELINE_PROMPTS["fixed_generic"]
    elif baseline_type == "non_inverted":
        # Match the emotion instead of inverting it
        non_inverted = {
            "LVHA": "An intense, agitated piece with dissonant chords and fast tempo, "
                    "conveying anxiety and tension at 140 BPM",
            "LVLA": "A slow, melancholic piece with minor key piano, somber and withdrawn, 50 BPM",
            "HVHA": "A high-energy celebratory piece with bright brass and fast drums, 130 BPM",
            "HVLA": "A soft, content ambient piece with gentle pads, relaxed and peaceful, 65 BPM",
        }
        return non_inverted.get(emotion_quadrant, BASELINE_PROMPTS["fixed_generic"])
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


# ----------------------------------------------------------------
# Full Evaluation Runner
# ----------------------------------------------------------------

def run_evaluation(
    result_dir: str | Path,
    output_path: str | Path = "evaluation/results.json",
) -> dict:
    """
    Run all evaluations on saved pipeline results.

    Args:
        result_dir: Directory containing *_meta.json and *.wav files from pipeline runs.
        output_path: Where to save the evaluation report.
    """
    result_dir = Path(result_dir)
    output_path = Path(output_path)

    meta_files = sorted(result_dir.glob("*_meta.json"))
    if not meta_files:
        logger.error(f"No result files found in {result_dir}")
        return {}

    logger.info(f"Found {len(meta_files)} pipeline results in {result_dir}")

    audio_paths = []
    prompts = []
    predicted_va = []

    for meta_path in meta_files:
        with open(meta_path) as f:
            meta = json.load(f)
        audio_path = result_dir / meta["audio_file"]
        if audio_path.exists():
            audio_paths.append(audio_path)
            prompts.append(meta["therapeutic_prompt"])
            predicted_va.append([
                meta["emotion"]["valence"],
                meta["emotion"]["arousal"],
            ])

    all_results = {"n_samples": len(audio_paths)}

    # CLAP scores
    if audio_paths and prompts:
        logger.info("Computing CLAP scores...")
        all_results["clap"] = evaluate_clap_score(audio_paths, prompts)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Evaluation results saved to {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Therapeutic Soundtrack Pipeline")
    parser.add_argument(
        "--result-dir", type=str, default="outputs",
        help="Directory with pipeline outputs (*_meta.json + *.wav)",
    )
    parser.add_argument(
        "--output", type=str, default="evaluation/results.json",
        help="Path to save evaluation report",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_evaluation(args.result_dir, args.output)


if __name__ == "__main__":
    main()
