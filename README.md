# Neuromusic

EEG-driven therapeutic music generation. Detects a listener's emotional state from brain signals and generates music that guides them toward emotional balance via **affective inversion** — anxious listeners hear calming music, low-energy listeners hear uplifting music.

## Pipeline

```
EEG (14ch, 128 Hz)
  │
  ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  TSCeption   │ ──▶ │ Affective Bridge │ ──▶ │   MusicGen   │ ──▶ 🔊 Audio
│  (emotion)   │     │ (LLM/templates)  │     │  (text→music)│
└─────────────┘     └──────────────────┘     └──────────────┘
 valence/arousal      therapeutic prompt       waveform (32kHz)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download DREAMER dataset from HuggingFace (~5 GB)
python data/scripts/download_dreamer.py

# Train emotion classifiers (arousal + valence, ~50 min on GPU)
python -m src.biosignal.train_dreamer --dimension both --model tsception --max-epochs 50

# Run the full pipeline on a real EEG sample
python -m src.pipeline.run --dreamer-sample 42

# Launch interactive Gradio demo
python -m src.pipeline.demo
```

## Project Structure

```
neuromusic/
├── configs/default.yaml        # Pipeline configuration
├── data/
│   ├── raw/dreamer/            # DREAMER EEG data (gitignored)
│   └── scripts/                # download_dreamer.py
├── src/
│   ├── biosignal/
│   │   ├── models.py           # TSCeption, EEGNet architectures
│   │   ├── train_dreamer.py    # Training (subject-dependent splits)
│   │   └── classifier.py       # Inference (arousal + valence → quadrant)
│   ├── bridge/
│   │   └── prompt_generator.py # Emotion → therapeutic music prompt
│   ├── musicgen/
│   │   └── generator.py        # MusicGen wrapper (audiocraft)
│   └── pipeline/
│       ├── pipeline.py         # End-to-end orchestration
│       ├── run.py              # CLI entry point
│       └── demo.py             # Gradio web demo
├── evaluation/evaluate.py      # CLAP score, accuracy metrics
├── checkpoints/                # Trained weights (gitignored)
└── outputs/                    # Generated audio (gitignored)
```

## Dataset

**DREAMER** — 14-channel EEG recorded with an Emotiv EPOC headset (128 Hz) from 23 subjects during 18 emotion-eliciting video clips. Preprocessed by the [MONSTER benchmark](https://arxiv.org/abs/2502.15122) using TorchEEG. Freely available on HuggingFace:

| Variant | Task | Samples | Link |
|---------|------|---------|------|
| DREAMERA | Arousal (low/high) | 170,246 | [monster-monash/DREAMERA](https://huggingface.co/datasets/monster-monash/DREAMERA) |
| DREAMERV | Valence (low/high) | 170,246 | [monster-monash/DREAMERV](https://huggingface.co/datasets/monster-monash/DREAMERV) |

**Evaluation protocol:** Subject-dependent stratified splits (80/20 per subject). This matches published DREAMER benchmarks reporting 80–90% accuracy, unlike cross-subject splits (~55%).

## Key Components

| Stage | Model | What it does |
|-------|-------|-------------|
| **Emotion** | TSCeption (7.8K params) | Classifies 2s EEG windows into high/low arousal and valence |
| **Bridge** | Templates / Gemini API | Inverts detected emotion into a therapeutic music text prompt |
| **Music** | MusicGen (Meta) | Generates 15s audio waveform from the text prompt |

## References

- Ding et al., *TSCeption*, 2021 — [arXiv:2104.02935](https://arxiv.org/abs/2104.02935)
- Katsigiannis & Ramzan, *DREAMER*, IEEE JBHI 2018 — [DOI:10.1109/JBHI.2017.2688412](https://doi.org/10.1109/JBHI.2017.2688412)
- Copet et al., *MusicGen*, Meta 2023 — [arXiv:2306.05284](https://arxiv.org/abs/2306.05284)
- Foumani et al., *MONSTER*, 2025 — [arXiv:2502.15122](https://arxiv.org/abs/2502.15122)

## Team

- Linrui Ma (linrui@mit.edu)
- Grace Yuan (yuangc@mit.edu)
- Aimee Yu (aimeeyu@mit.edu)

MIT 6.S985 — Biological Signal Processing & AI, Spring 2026
