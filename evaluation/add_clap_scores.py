"""
Compute CLAP scores on an already-completed eval run and patch them into
the existing eval_report.json.

Use this when audio .wav files are already on disk from a previous
run_full_eval run but CLAP scores were skipped (laion-clap not installed).

Usage:
    python -m evaluation.add_clap_scores
    python -m evaluation.add_clap_scores --eval-dir evaluation/full_eval_with_prompts_100
    python -m evaluation.add_clap_scores --eval-dir evaluation/full_eval_continuous_v2_100
"""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def add_clap_scores(eval_dir: Path) -> None:
    report_path = eval_dir / "eval_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"No eval_report.json found in {eval_dir}")

    with open(report_path) as f:
        report = json.load(f)

    samples = report.get("samples", [])
    if not samples:
        raise ValueError("eval_report.json has no 'samples' list.")

    try:
        import laion_clap
    except ImportError:
        raise ImportError(
            "laion-clap not installed. Run: pip install laion-clap"
        )

    import numpy as np
    import torch

    logger.info("Loading CLAP model...")
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()

    conditions = ["therapeutic", "random", "fixed_calm", "non_inverted"]
    clap_results = {}

    for cond in conditions:
        audio_paths = []
        prompts = []
        missing = []

        for s in samples:
            fname = s.get("audio_files", {}).get(cond)
            prompt = s.get("prompts", {}).get("therapeutic")  # always score vs therapeutic
            if fname is None or prompt is None:
                continue
            p = eval_dir / fname
            if p.exists():
                audio_paths.append(str(p))
                prompts.append(prompt)
            else:
                missing.append(str(p))

        if missing:
            logger.warning(f"  {cond}: {len(missing)} audio files not found, skipping those.")

        if not audio_paths:
            logger.warning(f"  {cond}: no audio files found — skipping.")
            clap_results[cond] = {"mean": "N/A", "std": "N/A", "error": "no audio files"}
            continue

        batch_size = 32
        logger.info(f"  {cond}: scoring {len(audio_paths)} files (batch_size={batch_size})...")

        audio_emb_parts = []
        for i in range(0, len(audio_paths), batch_size):
            batch = audio_paths[i : i + batch_size]
            emb = model.get_audio_embedding_from_filelist(batch, use_tensor=True)
            audio_emb_parts.append(emb.detach().cpu())
            torch.cuda.empty_cache()
        audio_emb = torch.cat(audio_emb_parts, dim=0)

        text_emb_parts = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            emb = model.get_text_embedding(batch, use_tensor=True)
            text_emb_parts.append(emb.detach().cpu())
            torch.cuda.empty_cache()
        text_emb = torch.cat(text_emb_parts, dim=0)

        audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1)
        text_emb  = torch.nn.functional.normalize(text_emb, dim=-1)

        scores = torch.sum(audio_emb * text_emb, dim=-1).numpy()

        clap_results[cond] = {
            "mean": float(np.mean(scores)),
            "std":  float(np.std(scores)),
            "min":  float(np.min(scores)),
            "max":  float(np.max(scores)),
            "n":    len(scores),
        }
        logger.info(
            f"    mean={clap_results[cond]['mean']:.3f} "
            f"± {clap_results[cond]['std']:.3f}"
        )

    # Patch scores into the report
    report["clap_scores"] = clap_results

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nUpdated {report_path}")
    print(f"\n{'='*50}")
    print("CLAP SCORES")
    print(f"{'='*50}")
    print(f"  {'Condition':<20} {'Mean':>8} {'Std':>8}  {'N':>5}")
    print(f"  {'-'*45}")
    for cond, c in clap_results.items():
        mean_s = f"{c['mean']:.3f}" if isinstance(c.get('mean'), float) else "N/A"
        std_s  = f"{c['std']:.3f}"  if isinstance(c.get('std'),  float) else "N/A"
        n_s    = str(c.get('n', '?'))
        marker = " <- ours" if cond == "therapeutic" else ""
        print(f"  {cond:<20} {mean_s:>8} {std_s:>8}  {n_s:>5}{marker}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Patch CLAP scores into an existing eval_report.json"
    )
    parser.add_argument(
        "--eval-dir",
        default="evaluation/full_eval_with_prompts_100",
        help="Directory containing eval_report.json and *.wav files",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    add_clap_scores(Path(args.eval_dir))


if __name__ == "__main__":
    main()
