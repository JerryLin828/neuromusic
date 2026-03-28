"""
End-to-end Therapeutic Soundtrack Pipeline.

Wires together:
  1. EEG Emotion Classifier (src.biosignal)
  2. Prompt Generator / LLM Bridge (src.bridge)
  3. Music Generator (src.musicgen)

Usage:
    pipeline = TherapeuticSoundtrackPipeline.from_config("configs/default.yaml")
    result = pipeline.generate(eeg_data)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from src.biosignal.classifier import EmotionClassifier, EmotionResult
from src.bridge.prompt_generator import PromptGenerator
from src.musicgen.generator import MusicGenerator

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Output of one pipeline run."""
    emotion: EmotionResult
    therapeutic_prompt: str
    audio: np.ndarray
    sample_rate: int

    def save(self, output_dir: str | Path, name: str = "output"):
        """Save all outputs (audio + metadata) to files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        import torchaudio

        audio_path = out / f"{name}.wav"
        tensor = torch.from_numpy(self.audio).unsqueeze(0)
        torchaudio.save(str(audio_path), tensor, self.sample_rate)

        meta = {
            "emotion": {
                "valence": self.emotion.valence,
                "arousal": self.emotion.arousal,
                "valence_class": self.emotion.valence_class,
                "arousal_class": self.emotion.arousal_class,
                "quadrant": self.emotion.quadrant,
                "confidence": self.emotion.confidence,
            },
            "therapeutic_prompt": self.therapeutic_prompt,
            "audio_file": f"{name}.wav",
            "sample_rate": self.sample_rate,
            "duration_seconds": len(self.audio) / self.sample_rate,
        }
        with open(out / f"{name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved result to {out}/{name}.*")


class TherapeuticSoundtrackPipeline:
    """Full EEG → Emotion → Prompt → Music pipeline."""

    def __init__(
        self,
        classifier: EmotionClassifier,
        prompt_generator: PromptGenerator,
        music_generator: MusicGenerator,
        output_dir: str = "outputs",
    ):
        self.classifier = classifier
        self.prompt_generator = prompt_generator
        self.music_generator = music_generator
        self.output_dir = Path(output_dir)

    @classmethod
    def from_config(cls, config_path: str | Path) -> TherapeuticSoundtrackPipeline:
        """Build a pipeline from a YAML config file."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        bio_cfg = cfg.get("biosignal", {})
        bio_backend = bio_cfg["backend"]
        bio_kwargs = bio_cfg.get(bio_backend, {})
        classifier = EmotionClassifier(backend=bio_backend, **bio_kwargs)

        bridge_cfg = cfg.get("bridge", {})
        bridge_backend = bridge_cfg.get("backend", "template")
        bridge_kwargs = bridge_cfg.get(bridge_backend, {})
        prompt_gen = PromptGenerator(backend=bridge_backend, **bridge_kwargs)

        music_cfg = cfg.get("musicgen", {})
        music_gen = MusicGenerator(**music_cfg)

        paths_cfg = cfg.get("paths", {})

        return cls(
            classifier=classifier,
            prompt_generator=prompt_gen,
            music_generator=music_gen,
            output_dir=paths_cfg.get("output_dir", "outputs"),
        )

    def generate(self, eeg_data: np.ndarray) -> PipelineResult:
        """
        Run the full pipeline on EEG data.

        Args:
            eeg_data: Preprocessed EEG array. Shape and format must match
                the configured biosignal backend's requirements.

        Returns:
            PipelineResult with emotion, prompt, and generated audio.
        """
        logger.info("=" * 60)
        logger.info("Running Therapeutic Soundtrack Pipeline")
        logger.info("=" * 60)

        # Step 1: Classify emotion
        logger.info("[1/3] Classifying emotion from EEG...")
        emotion = self.classifier.classify(eeg_data)
        logger.info(
            f"  Detected: valence={emotion.valence:.2f}, arousal={emotion.arousal:.2f} "
            f"→ {emotion.quadrant} (confidence={emotion.confidence:.2f})"
        )

        # Step 2: Generate therapeutic prompt (with affective inversion)
        logger.info("[2/3] Generating therapeutic music prompt...")
        emotion_state = emotion.to_bridge_format()
        prompt = self.prompt_generator.generate(emotion_state)
        logger.info(f"  Prompt: {prompt[:100]}...")

        # Step 3: Generate music
        logger.info("[3/3] Generating music...")
        audio = self.music_generator.generate(prompt)
        sr = self.music_generator.sample_rate
        logger.info(f"  Generated {len(audio)/sr:.1f}s of audio at {sr} Hz")

        result = PipelineResult(
            emotion=emotion,
            therapeutic_prompt=prompt,
            audio=audio,
            sample_rate=sr,
        )

        logger.info("Pipeline complete")
        return result
