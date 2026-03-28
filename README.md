# NeuroMusic

EEG-driven therapeutic music generation. Detects a listener's emotional state from brain signals and generates music that guides them toward emotional balance via **affective inversion** — anxious listeners hear calming music, low-energy listeners hear uplifting music.

## Pipeline

```
EEG (14ch, 128 Hz)
  │
  ▼
┌─────────────┐      ┌──────────────────┐     ┌──────────────┐
│  TSCeption  │  ──▶ │ Affective Bridge │ ──▶ │   MusicGen   │ ──▶  Audio
│  (emotion)  │      │ (Gemini / rules) │     │  (text→music)│
└─────────────┘      └──────────────────┘     └──────────────┘
 valence/arousal      therapeutic prompt       waveform (32kHz)
```

## Results

| Metric | Value |
|---|---|
| Arousal classification | 77.2% (±8.0%) |
| Valence classification | 78.1% (±9.8%) |
| **Quadrant accuracy** | **75.0% (9/12)** |

| Condition | CLAP Score |
|---|---|
| **Therapeutic (ours)** | **0.494** |
| Fixed calm baseline | 0.343 |
| Non-inverted baseline | 0.234 |
| Random baseline | 0.062 |

Our pipeline produces audio that aligns 2x better with its therapeutic target than the non-inverted baseline, and 8x better than random.

## Setup

Install order matters due to audiocraft's strict dependency pins:

```bash
# 1. Install PyTorch (exact versions for audiocraft)
pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Install av from binary wheel
pip install av==12.0.0 --only-binary=av

# 3. Install audiocraft
pip install git+https://github.com/facebookresearch/audiocraft.git --no-deps
pip install "xformers<0.0.23" torchtext==0.16.0 "spacy>=3.6.1" \
    flashy demucs encodec julius num2words lameenc hydra-core dora-search \
    submitit cloudpickle treetable einops sentencepiece protobuf \
    "antlr4-python3-runtime==4.9.*" docopt hydra-colorlog librosa \
    pesq pystoi soundfile torchdiffeq torchmetrics "transformers==4.40.0" "numpy<2.0.0"

# 4. Install project dependencies
pip install -r requirements.txt
```

## Quick Start

All scripts read from `configs/default.yaml` — no CLI flags needed for standard runs.

```bash
# Download DREAMER dataset from HuggingFace (~5 GB)
python data/scripts/download_dreamer.py

# Train emotion classifiers (arousal + valence)
python -m src.biosignal.train_dreamer

# Run the full pipeline on a DREAMER sample
python -m src.pipeline.run --dreamer-sample 42

# Run evaluation
python -m evaluation.run_full_eval

# (Optional) Set Gemini API for richer prompts
export GEMINI_API_KEY="your-key"

# (Optional) Launch Gradio demo
python -m src.pipeline.demo
```

Edit `configs/default.yaml` to change paths, hyperparameters, model choices, etc.

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
│   ├── pipeline/
│   │   ├── pipeline.py         # End-to-end orchestration
│   │   ├── run.py              # CLI entry point
│   │   └── demo.py             # Gradio web demo
│   └── utils/
│       └── io.py               # Config loading, shared I/O helpers
├── evaluation/
│   ├── evaluate.py             # CLAP score, accuracy metrics
│   └── run_full_eval.py        # Full evaluation harness
├── checkpoints/                # Trained weights (gitignored)
└── outputs/                    # Generated audio (gitignored)
```

## Dataset

**DREAMER** — 14-channel EEG recorded with an Emotiv EPOC headset (128 Hz) from 23 subjects during 18 emotion-eliciting video clips. Preprocessed by the [MONSTER benchmark](https://arxiv.org/abs/2502.15122). Available on HuggingFace:

| Variant | Task | Samples | Link |
|---------|------|---------|------|
| DREAMERA | Arousal (low/high) | 170,246 | [monster-monash/DREAMERA](https://huggingface.co/datasets/monster-monash/DREAMERA) |
| DREAMERV | Valence (low/high) | 170,246 | [monster-monash/DREAMERV](https://huggingface.co/datasets/monster-monash/DREAMERV) |

**Evaluation protocol:** Subject-dependent stratified splits (80/20 per subject), matching published DREAMER benchmarks (80-90% accuracy).

## Key Components

| Stage | Model | What it does |
|-------|-------|-------------|
| **Emotion** | TSCeption (7.8K params) | Classifies 2s EEG windows into high/low arousal and valence |
| **Bridge** | Templates / Gemini API | Inverts detected emotion into a therapeutic music text prompt |
| **Music** | MusicGen-small (Meta) | Generates 10-15s audio waveform from the text prompt |
| **Evaluation** | CLAP (LAION) | Measures audio-text semantic alignment |

## References

- Ding et al., *TSCeption*, 2021 — [arXiv:2104.02935](https://arxiv.org/abs/2104.02935)
- Katsigiannis & Ramzan, *DREAMER*, IEEE JBHI 2018 — [DOI:10.1109/JBHI.2017.2688412](https://doi.org/10.1109/JBHI.2017.2688412)
- Copet et al., *MusicGen*, Meta 2023 — [arXiv:2306.05284](https://arxiv.org/abs/2306.05284)
- Foumani et al., *MONSTER*, 2025 — [arXiv:2502.15122](https://arxiv.org/abs/2502.15122)
- Wu et al., *CLAP*, ICASSP 2023 — [arXiv:2211.06687](https://arxiv.org/abs/2211.06687)

## Team

- Linrui Ma (linrui@mit.edu)
- Grace Yuan (yuangc@mit.edu)
- Aimee Yu (aimeeyu@mit.edu)

MIT 6.S985 — Multimodal AI, Spring 2026
