"""
Full evaluation pipeline: generate samples across quadrants + baselines, compute metrics.

Produces:
  1. Audio samples for all 4 emotion quadrants (from real EEG data)
  2. Baseline audio (random prompt, fixed prompt, non-inverted prompt)
  3. CLAP scores comparing our pipeline vs. baselines
  4. LALM Judge scores — quadrant classification via Gemini (if enabled)
  5. Emotion classification accuracy summary
  6. A clean JSON report + human-readable summary

Usage (on GPU machine):
    python -m evaluation.run_full_eval
    python -m evaluation.run_full_eval --config configs/default.yaml
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.utils.io import load_config, add_config_arg

logger = logging.getLogger(__name__)


def find_quadrant_samples(data_dir: Path, n_per_quadrant: int = 3):
    """Find EEG samples from each emotion quadrant using ground-truth labels."""
    X = np.load(data_dir / "arousal" / "X.npy", mmap_mode="r")
    y_a = np.load(data_dir / "arousal" / "y.npy")
    y_v = np.load(data_dir / "valence" / "y.npy")

    quadrant_map = {
        "HVHA": (1, 1),
        "HVLA": (1, 0),
        "LVHA": (0, 1),
        "LVLA": (0, 0),
    }

    samples = {}
    for quad_name, (v_class, a_class) in quadrant_map.items():
        mask = (y_v == v_class) & (y_a == a_class)
        indices = np.where(mask)[0]
        chosen = np.random.choice(indices, size=min(n_per_quadrant, len(indices)), replace=False)
        samples[quad_name] = chosen.tolist()
        logger.info(f"  {quad_name}: {len(indices)} total, selected {len(chosen)}")

    return samples, X, y_a, y_v


def run_full_evaluation(cfg: dict):
    paths = cfg.get("paths", {})
    eval_cfg = cfg.get("evaluation", {})
    musicgen_cfg = cfg.get("musicgen", {})

    data_dir = Path(paths.get("data_dir", "data/raw/dreamer"))
    output_dir = Path(paths.get("eval_dir", "evaluation/full_eval"))
    n_per_quadrant = eval_cfg.get("n_per_quadrant", 3)
    duration = eval_cfg.get("duration", 10.0)
    seed = eval_cfg.get("seed", 42)

    output_dir.mkdir(parents=True, exist_ok=True)

    from src.biosignal.classifier import EmotionClassifier
    from src.bridge.prompt_generator import PromptGenerator, EmotionState
    from src.musicgen.generator import MusicGenerator
    from evaluation.evaluate import evaluate_clap_score, generate_baseline_prompt

    bio_cfg = cfg["biosignal"]
    classifier = EmotionClassifier(
        backend=bio_cfg["backend"],
        **bio_cfg.get(bio_cfg["backend"], {}),
    )
    bridge_cfg = cfg.get("bridge", {})
    bridge_backend = bridge_cfg.get("backend", "template")
    bridge_kwargs = bridge_cfg.get(bridge_backend, {})
    prompt_gen = PromptGenerator(backend=bridge_backend, **bridge_kwargs)
    music_gen = MusicGenerator(
        model_name=musicgen_cfg.get("model_name", "facebook/musicgen-small"),
        duration=duration,
        temperature=musicgen_cfg.get("temperature", 1.0),
        top_k=musicgen_cfg.get("top_k", 250),
        top_p=musicgen_cfg.get("top_p", 0.0),
        cfg_coef=musicgen_cfg.get("cfg_coef", 3.0),
    )

    # --- Find samples per quadrant ---
    logger.info("Finding samples per quadrant...")
    np.random.seed(seed)
    quadrant_samples, X, y_a, y_v = find_quadrant_samples(data_dir, n_per_quadrant)

    # --- Generate pipeline outputs + baselines ---
    results = []

    for quad_name, indices in quadrant_samples.items():
        for i, idx in enumerate(indices):
            eeg = X[idx].copy()
            gt_a = "high" if y_a[idx] == 1 else "low"
            gt_v = "high" if y_v[idx] == 1 else "low"
            gt_quad = quad_name

            logger.info(f"\n{'='*50}")
            logger.info(f"Sample {idx} | GT: {gt_quad} (V={gt_v}, A={gt_a})")

            emotion = classifier.classify(eeg)
            logger.info(f"  Predicted: {emotion.quadrant} (V={emotion.valence:.2f}, A={emotion.arousal:.2f})")

            emotion_state = emotion.to_bridge_format()
            therapeutic_prompt = prompt_gen.generate(emotion_state)

            random_prompt = generate_baseline_prompt("random")
            fixed_prompt = generate_baseline_prompt("fixed_calm")
            noninv_prompt = generate_baseline_prompt("non_inverted", emotion.quadrant)

            conditions = {
                "therapeutic": therapeutic_prompt,
                "random": random_prompt,
                "fixed_calm": fixed_prompt,
                "non_inverted": noninv_prompt,
            }

            sample_result = {
                "sample_idx": int(idx),
                "gt_quadrant": gt_quad,
                "gt_arousal": gt_a,
                "gt_valence": gt_v,
                "pred_quadrant": emotion.quadrant,
                "pred_arousal": float(emotion.arousal),
                "pred_valence": float(emotion.valence),
                "pred_confidence": float(emotion.confidence),
                "correct_quadrant": emotion.quadrant == gt_quad,
                "audio_files": {},
                "prompts": {},
            }

            for cond_name, prompt in conditions.items():
                logger.info(f"  Generating {cond_name}...")
                audio = music_gen.generate(prompt)
                fname = f"{gt_quad}_{i}_{cond_name}.wav"
                music_gen.save_audio(audio, output_dir / fname)
                sample_result["audio_files"][cond_name] = fname
                sample_result["prompts"][cond_name] = prompt

            results.append(sample_result)

    # --- Compute CLAP scores per condition ---
    logger.info(f"\n{'='*50}")
    logger.info("Computing CLAP scores...")

    clap_results = {}
    for cond_name in ["therapeutic", "random", "fixed_calm", "non_inverted"]:
        audio_paths = [
            output_dir / r["audio_files"][cond_name] for r in results
        ]
        prompts = [r["prompts"]["therapeutic"] for r in results]
        try:
            clap = evaluate_clap_score(audio_paths, prompts)
            clap_results[cond_name] = clap
        except Exception as e:
            logger.warning(f"  CLAP failed for {cond_name}: {e}")
            clap_results[cond_name] = {"error": str(e)}

    # --- LALM Judge evaluation ---
    judge_cfg = cfg.get("judge", {})
    judge_enabled = judge_cfg.get("enabled", False)
    judge_results_by_cond = {}
    judge_agg = {}

    if judge_enabled:
        logger.info(f"\n{'='*50}")
        logger.info("Running LALM Judge evaluation...")

        from evaluation.lalm_judge import GeminiJudge, TEMPLATE_PROMPT_QUADRANT
        from evaluation.evaluate import (
            compute_inversion_rate,
            compute_prompt_alignment_rate,
        )

        judge = GeminiJudge.from_config(cfg)
        max_workers = judge_cfg.get("max_concurrent", 5)
        conditions = ["therapeutic", "random", "fixed_calm", "non_inverted"]

        # Judge all conditions
        for cond_name in conditions:
            audio_paths = [
                output_dir / r["audio_files"][cond_name] for r in results
            ]
            logger.info(f"  Judging {cond_name} ({len(audio_paths)} files)...")
            cond_results = judge.judge_batch(audio_paths, max_workers=max_workers)
            judge_results_by_cond[cond_name] = cond_results

        # Compute aggregation metrics for each condition
        for cond_name in conditions:
            cond_judge = judge_results_by_cond[cond_name]
            detected_quads = [r["pred_quadrant"] for r in results]

            # Inversion rate: judged quadrant != detected emotion
            inv = compute_inversion_rate(cond_judge, detected_quads)

            # Prompt alignment: judged quadrant == prompt's intended quadrant
            # For therapeutic: use TEMPLATE_PROMPT_QUADRANT[detected] as target
            # For non_inverted: the prompt matches the detected quadrant
            # For fixed_calm: always HVLA-ish
            # For random: always HVHA-ish (jazz, energetic)
            if cond_name == "therapeutic":
                prompt_quads = [
                    TEMPLATE_PROMPT_QUADRANT.get(r["pred_quadrant"], "HVLA")
                    for r in results
                ]
            elif cond_name == "non_inverted":
                prompt_quads = [r["pred_quadrant"] for r in results]
            elif cond_name == "fixed_calm":
                prompt_quads = ["HVLA"] * len(results)
            elif cond_name == "random":
                prompt_quads = ["HVHA"] * len(results)
            else:
                prompt_quads = ["HVLA"] * len(results)

            align = compute_prompt_alignment_rate(cond_judge, prompt_quads)

            judge_agg[cond_name] = {
                "inversion_rate": inv,
                "prompt_alignment": align,
            }

        # Save per-sample judge results as JSONL
        jsonl_path = output_dir / "judge_results.jsonl"
        with open(jsonl_path, "w") as f:
            for i, r in enumerate(results):
                for cond_name in conditions:
                    jr = judge_results_by_cond[cond_name][i]
                    line = {
                        "sample_idx": r["sample_idx"],
                        "condition": cond_name,
                        "detected_quadrant": r["pred_quadrant"],
                        "predicted_quadrant": jr.predicted_quadrant,
                        "reasoning": jr.reasoning,
                        "option_order": jr.option_order,
                        "latency_ms": jr.latency_ms,
                        "error": jr.error,
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
        logger.info(f"  Judge results saved to {jsonl_path}")
    else:
        logger.info("LALM Judge disabled (set judge.enabled: true in config)")

    # --- Emotion classification summary ---
    n_correct = sum(1 for r in results if r["correct_quadrant"])
    n_total = len(results)

    # --- Compile report ---
    report = {
        "n_samples": n_total,
        "emotion_classification": {
            "quadrant_accuracy": f"{n_correct}/{n_total} ({n_correct/n_total:.1%})",
            "per_sample": [
                {
                    "idx": r["sample_idx"],
                    "gt": r["gt_quadrant"],
                    "pred": r["pred_quadrant"],
                    "correct": r["correct_quadrant"],
                }
                for r in results
            ],
        },
        "clap_scores": {
            cond: {
                "mean": c.get("mean", "N/A"),
                "std": c.get("std", "N/A"),
            }
            for cond, c in clap_results.items()
        },
        "samples": results,
    }

    if judge_agg:
        report["judge_scores"] = {
            cond: {
                "inversion_rate": agg["inversion_rate"]["overall"],
                "prompt_alignment": agg["prompt_alignment"]["overall"],
                "n_samples": agg["prompt_alignment"]["n_samples"],
                "n_errors": agg["prompt_alignment"]["n_errors"],
            }
            for cond, agg in judge_agg.items()
        }
        report["judge_detailed"] = judge_agg

    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # --- Print summary ---
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Samples: {n_total} ({n_per_quadrant} per quadrant x 4)")
    print(f"  Quadrant accuracy: {n_correct}/{n_total} ({n_correct/n_total:.1%})")
    print()
    print("  CLAP Scores (audio-text alignment):")
    print(f"  {'Condition':<20} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*36}")
    for cond, c in clap_results.items():
        mean = c.get("mean", "N/A")
        std = c.get("std", "N/A")
        mean_s = f"{mean:.3f}" if isinstance(mean, float) else mean
        std_s = f"{std:.3f}" if isinstance(std, float) else std
        marker = " <- ours" if cond == "therapeutic" else ""
        print(f"  {cond:<20} {mean_s:>8} {std_s:>8}{marker}")

    if judge_agg:
        print()
        print("  LALM Judge (emotion quadrant classification):")
        print(f"  {'Condition':<20} {'Inversion':>10} {'Alignment':>10}")
        print(f"  {'-'*40}")
        for cond in ["therapeutic", "random", "fixed_calm", "non_inverted"]:
            if cond in judge_agg:
                inv = judge_agg[cond]["inversion_rate"]["overall"]
                align = judge_agg[cond]["prompt_alignment"]["overall"]
                marker = " <- ours" if cond == "therapeutic" else ""
                print(f"  {cond:<20} {inv:>9.1%} {align:>9.1%}{marker}")

    print()
    print(f"  Report: {report_path}")
    print(f"  Audio:  {output_dir}/*.wav")
    if judge_enabled:
        print(f"  Judge:  {output_dir}/judge_results.jsonl")
    print(f"{'='*60}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Full pipeline evaluation")
    add_config_arg(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = load_config(args.config)
    run_full_evaluation(cfg)


if __name__ == "__main__":
    main()
