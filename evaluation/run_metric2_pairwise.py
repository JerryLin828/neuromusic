"""
Metric 2: Pairwise Preference (standalone).

For each sample, sends 3 audio clips (therapeutic, non_inverted, fixed_calm)
to Gemini and asks which best matches the therapeutic target emotion
derived from the ground-truth quadrant.

Reads:   eval_report.json  (for gt_quadrant + audio file names)
Writes:  pairwise_results.jsonl, pairwise_aggregate.json

Usage:
    export GEMINI_API_KEY=...
    python3 -m evaluation.run_metric2_pairwise [--eval-dir PATH] [--model MODEL]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONDITIONS = ["therapeutic", "non_inverted", "fixed_calm"]


def main():
    parser = argparse.ArgumentParser(description="Metric 2: Pairwise Preference")
    parser.add_argument(
        "--eval-dir", type=str,
        default="evaluation/full_eval_with_prompts_100",
        help="Directory with eval_report.json and wav files",
    )
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--max-workers", type=int, default=5)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    from evaluation.lalm_judge import GeminiJudge, TEMPLATE_PROMPT_QUADRANT
    from evaluation.evaluate import compute_pairwise_win_rate

    # Load eval report
    with open(eval_dir / "eval_report.json") as f:
        report = json.load(f)
    samples = report["samples"]
    logger.info(f"Loaded {len(samples)} samples from {eval_dir / 'eval_report.json'}")

    # Build pairwise inputs
    audio_paths_list = []
    target_quadrants = []
    valid_samples = []

    for s in samples:
        paths = {}
        all_exist = True
        for cond in CONDITIONS:
            p = eval_dir / s["audio_files"][cond]
            if not p.exists():
                all_exist = False
                break
            paths[cond] = p

        if not all_exist:
            logger.warning(f"  Sample {s['sample_idx']}: missing wav files, skipping")
            continue

        gt_target = TEMPLATE_PROMPT_QUADRANT.get(s["gt_quadrant"], "HVLA")
        audio_paths_list.append(paths)
        target_quadrants.append(gt_target)
        valid_samples.append(s)

    logger.info(f"Prepared {len(audio_paths_list)} pairwise comparisons")

    # Initialize judge
    judge = GeminiJudge(
        model=args.model,
        temperature=0.0,
        max_retries=3,
        use_cache=False,  # Pairwise is not cached (3 clips per call)
    )

    # Run
    t0 = time.time()
    logger.info(f"Running pairwise on {len(audio_paths_list)} samples...")
    results = judge.judge_pairwise_batch(
        audio_paths_list, target_quadrants, max_workers=args.max_workers
    )
    elapsed = time.time() - t0

    n_ok = sum(1 for r in results if r.error is None)
    n_err = len(results) - n_ok
    logger.info(f"Done in {elapsed:.0f}s — {n_ok} ok, {n_err} errors")

    # Save pairwise_results.jsonl
    jsonl_path = eval_dir / "pairwise_results.jsonl"
    with open(jsonl_path, "w") as f:
        for s, tq, pr in zip(valid_samples, target_quadrants, results):
            record = {
                "sample_idx": s["sample_idx"],
                "detected_quadrant": s["gt_quadrant"],
                "target_quadrant": tq,
                "winner": pr.winner,
                "reasoning": pr.reasoning,
                "presentation_order": pr.presentation_order,
                "latency_ms": pr.latency_ms,
                "error": pr.error,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Saved: {jsonl_path}")

    # Aggregate
    agg = compute_pairwise_win_rate(results, CONDITIONS)
    agg_path = eval_dir / "pairwise_aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    logger.info(f"Saved: {agg_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Metric 2: Pairwise Preference (GT-based)")
    print(f"{'='*60}\n")
    print(f"{'Condition':<22} {'Wins':>6} {'Win Rate':>10}")
    print("-" * 40)
    for cond in CONDITIONS:
        w = agg["wins"].get(cond, 0)
        wr = agg["win_rates"].get(cond, 0)
        marker = " <- ours" if cond == "therapeutic" else ""
        print(f"{cond:<22} {w:>6} {wr:>9.1%}{marker}")
    print(f"\nTotal valid: {agg['n_samples']}, Errors: {agg['n_errors']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
