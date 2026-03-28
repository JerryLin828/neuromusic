# Personalized Soundtrack Generator

A system that reads biological signals (EEG) to detect a user's emotional state, then generates therapeutic music designed to guide them toward emotional balance.

## Pipeline

```
Raw EEG (14ch, 128Hz) → [TSCeption] → emotion (valence, arousal)
                                            ↓
                                     [LLM / Templates]  (affective inversion)
                                            ↓
                                      music text prompt
                                            ↓
                                    [Pretrained MusicGen] → audio
```

## Project Structure

```
├── configs/             # Experiment configs (YAML)
├── data/
│   ├── raw/dreamer/     # DREAMER dataset from HuggingFace (gitignored)
│   └── scripts/         # Download script
├── src/
│   ├── biosignal/       # EEG emotion classification (TSCeption, EEGNet)
│   ├── bridge/          # Emotion → music prompt (templates + Gemini)
│   ├── musicgen/        # MusicGen wrapper for audio generation
│   ├── pipeline/        # End-to-end pipeline + Gradio demo
│   └── utils/           # Shared utilities
├── evaluation/          # Metric computation (CLAP score)
├── checkpoints/         # Trained model weights (gitignored)
├── outputs/             # Generated audio samples (gitignored)
└── notebooks/           # Exploration & visualization
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Download DREAMER data from HuggingFace (~5 GB)
python data/scripts/download_dreamer.py

# 2. Train emotion classifiers (arousal + valence)
python -m src.biosignal.train_dreamer --dimension both --model tsception --max-epochs 50

# 3. Run the pipeline on a DREAMER sample
python -m src.pipeline.run --dreamer-sample 42

# 4. Launch the interactive demo
python -m src.pipeline.demo
```

## Dataset

**DREAMER** — 14-channel EEG (Emotiv EPOC, 128 Hz) from 23 subjects watching 18 emotion-eliciting videos. Preprocessed by the MONSTER benchmark using TorchEEG. Freely available on HuggingFace.

- Arousal: [monster-monash/DREAMERA](https://huggingface.co/datasets/monster-monash/DREAMERA)
- Valence: [monster-monash/DREAMERV](https://huggingface.co/datasets/monster-monash/DREAMERV)

## Team

- Grace Yuan (yuangc@mit.edu)
- Aimee Yu (aimeeyu@mit.edu)
- Linrui Ma (linrui@mit.edu)

MIT 6.S985 Project
