"""
Metric 1: Target Quadrant Alignment (standalone).

Runs Gemini LALM Judge on existing wav files to classify each clip's
emotional quadrant, then computes alignment against the therapeutic
target derived from ground-truth emotion.

Reads:   eval_report.json  (for gt_quadrant + audio file names)
Writes:  judge_results.jsonl, judge_aggregate.json

Usage:
    export GEMINI_API_KEY=...
    python3 -m evaluation.run_metric1_alignment [--eval-dir PATH] [--model MODEL]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QUADS = ["HVHA", "HVLA", "LVHA", "LVLA"]
CONDITIONS = ["therapeutic", "random", "fixed_calm", "non_inverted"]


def main():
    parser = argparse.ArgumentParser(description="Metric 1: Target Quadrant Alignment")
    parser.add_argument(
        "--eval-dir", type=str,
        default="evaluation/full_eval_with_prompts_100",
        help="Directory with eval_report.json and wav files",
    )
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--cache-dir", type=str, default="outputs/judge_cache")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    from evaluation.lalm_judge import GeminiJudge, TEMPLATE_PROMPT_QUADRANT
    from evaluation.evaluate import compute_inversion_rate, compute_prompt_alignment_rate

    # Load eval report
    with open(eval_dir / "eval_report.json") as f:
        report = json.load(f)
    samples = report["samples"]
    logger.info(f"Loaded {len(samples)} samples from {eval_dir / 'eval_report.json'}")

    # Initialize judge
    judge = GeminiJudge(
        model=args.model,
        temperature=0.0,
        max_retries=3,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )

    # Judge all conditions
    judge_results_by_cond = {}
    for cond in CONDITIONS:
        audio_paths = [eval_dir / s["audio_files"][cond] for s in samples]
        missing = [p for p in audio_paths if not p.exists()]
        if missing:
            logger.error(f"  {cond}: {len(missing)} wav files missing, skipping")
            continue
        logger.info(f"Judging {cond} ({len(audio_paths)} files)...")
        t0 = time.time()
        results = judge.judge_batch(audio_paths, max_workers=args.max_workers)
        elapsed = time.time() - t0
        n_ok = sum(1 for r in results if r.error is None)
        logger.info(f"  Done in {elapsed:.0f}s — {n_ok} ok, {len(results) - n_ok} errors")
        judge_results_by_cond[cond] = results

    # Compute aggregation (GT-based)
    judge_agg = {}
    for cond in CONDITIONS:
        if cond not in judge_results_by_cond:
            continue
        cond_results = judge_results_by_cond[cond]
        gt_quads = [s["gt_quadrant"] for s in samples]

        inv = compute_inversion_rate(cond_results, gt_quads)

        if cond == "therapeutic":
            prompt_quads = [TEMPLATE_PROMPT_QUADRANT.get(s["gt_quadrant"], "HVLA") for s in samples]
        elif cond == "non_inverted":
            prompt_quads = [s["gt_quadrant"] for s in samples]
        elif cond == "fixed_calm":
            prompt_quads = ["HVLA"] * len(samples)
        elif cond == "random":
            prompt_quads = ["HVHA"] * len(samples)
        else:
            prompt_quads = ["HVLA"] * len(samples)

        align = compute_prompt_alignment_rate(cond_results, prompt_quads)

        judge_agg[cond] = {
            "inversion_rate": inv["overall"],
            "prompt_alignment": align["overall"],
            "n_samples": align["n_samples"],
            "n_errors": align["n_errors"],
            "inversion_per_quadrant": inv.get("per_quadrant", {}),
            "alignment_per_quadrant": align.get("per_quadrant", {}),
            "confusion_matrix": inv.get("confusion_matrix", {}),
        }

    # Save judge_results.jsonl
    jsonl_path = eval_dir / "judge_results.jsonl"
    with open(jsonl_path, "w") as f:
        for i, s in enumerate(samples):
            for cond in CONDITIONS:
                if cond not in judge_results_by_cond:
                    continue
                jr = judge_results_by_cond[cond][i]
                line = {
                    "sample_idx": s["sample_idx"],
                    "condition": cond,
                    "gt_quadrant": s["gt_quadrant"],
                    "pred_quadrant": s["pred_quadrant"],
                    "predicted_quadrant": jr.predicted_quadrant,
                    "reasoning": jr.reasoning,
                    "option_order": jr.option_order,
                    "latency_ms": jr.latency_ms,
                    "error": jr.error,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    logger.info(f"Saved: {jsonl_path}")

    # Save judge_aggregate.json
    agg_path = eval_dir / "judge_aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(judge_agg, f, indent=2)
    logger.info(f"Saved: {agg_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Metric 1: Target Quadrant Alignment (GT-based)")
    print(f"{'='*60}\n")
    print(f"{'Condition':<16} {'Alignment':>10} {'Inversion':>10}")
    print("-" * 40)
    for cond in CONDITIONS:
        if cond in judge_agg:
            a = judge_agg[cond]
            marker = " <- ours" if cond == "therapeutic" else ""
            print(f"{cond:<16} {a['prompt_alignment']:>9.1%} {a['inversion_rate']:>9.1%}{marker}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
