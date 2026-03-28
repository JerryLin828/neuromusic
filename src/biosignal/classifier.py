"""
EEG Emotion Classifier — DREAMER dataset with TorchEEG models.

Primary backend:
  - "dreamer": Uses trained TorchEEG models (TSCeption, EEGNet, DGCNN)
    on the DREAMER dataset. Requires two checkpoints: one for arousal
    classification and one for valence classification.
    Train with: python -m src.biosignal.train_dreamer --dimension both

Legacy backends (DEAP-based, kept for reference):
  - "mdjpt_de": mdJPT DE baseline + MLP on DEAP
  - "labram": LaBraM fine-tuned on DEAP

The classifier produces an EmotionResult with valence (low/high),
arousal (low/high), and the combined quadrant (HVHA, HVLA, LVHA, LVLA).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class EmotionResult:
    """Output of the emotion classifier."""
    valence: float          # probability of high-valence (0-1)
    arousal: float          # probability of high-arousal (0-1)
    valence_class: str      # "high" or "low"
    arousal_class: str      # "high" or "low"
    quadrant: str           # "HVHA", "HVLA", "LVHA", "LVLA"
    confidence: float       # average of both dimension confidences

    def to_bridge_format(self):
        """Convert to the format expected by the bridge module."""
        from src.bridge.prompt_generator import EmotionState
        return EmotionState(
            valence=self.valence,
            arousal=self.arousal,
            label=self.quadrant,
        )


class EmotionClassifier:
    """
    Classifies EEG data into emotion states (valence x arousal).

    Primary usage with DREAMER:
        classifier = EmotionClassifier(
            backend="dreamer",
            model_name="tsception",
            arousal_checkpoint="checkpoints/dreamer/arousal/tsception/best.ckpt",
            valence_checkpoint="checkpoints/dreamer/valence/tsception/best.ckpt",
        )
        result = classifier.classify(eeg_data)  # (14, 256) at 128 Hz
    """

    DREAMER_CHANNELS = 14
    DREAMER_TIMEPOINTS = 256
    DREAMER_FS = 128

    def __init__(self, backend: str, **kwargs):
        self.backend = backend
        self.kwargs = kwargs

        if backend == "dreamer":
            self._init_dreamer(**kwargs)
        else:
            raise ValueError(
                f"Unknown biosignal backend: '{backend}'. Available: 'dreamer'"
            )

    def _init_dreamer(
        self,
        model_name: str = "tsception",
        arousal_checkpoint: Optional[str] = None,
        valence_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        **_kwargs,
    ):
        """
        Load trained arousal and valence classifiers.

        Args:
            model_name: TorchEEG model architecture ('tsception', 'eegnet', 'dgcnn')
            arousal_checkpoint: Path to trained arousal model (.ckpt from train_dreamer.py)
            valence_checkpoint: Path to trained valence model (.ckpt)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if arousal_checkpoint is None or valence_checkpoint is None:
            raise ValueError(
                "DREAMER backend requires both arousal_checkpoint and valence_checkpoint.\n"
                "Train them first:\n"
                "  python -m src.biosignal.train_dreamer --dimension arousal --model tsception\n"
                "  python -m src.biosignal.train_dreamer --dimension valence --model tsception"
            )

        for name, path in [("arousal", arousal_checkpoint), ("valence", valence_checkpoint)]:
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{name} checkpoint not found: {path}\n"
                    f"Train it: python -m src.biosignal.train_dreamer "
                    f"--dimension {name} --model {model_name}"
                )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._model_name = model_name

        from src.biosignal.train_dreamer import build_model, DREAMERClassifier

        # Arousal model
        arousal_model = build_model(model_name)
        self._arousal_model = DREAMERClassifier.load_from_checkpoint(
            arousal_checkpoint,
            model=arousal_model,
            map_location=self._device,
        )
        self._arousal_model.to(self._device)
        self._arousal_model.eval()

        # Valence model
        valence_model = build_model(model_name)
        self._valence_model = DREAMERClassifier.load_from_checkpoint(
            valence_checkpoint,
            model=valence_model,
            map_location=self._device,
        )
        self._valence_model.to(self._device)
        self._valence_model.eval()

        logger.info(
            f"Loaded DREAMER classifiers ({model_name}):\n"
            f"  arousal: {arousal_checkpoint}\n"
            f"  valence: {valence_checkpoint}\n"
            f"  device: {self._device}"
        )

    def classify(self, eeg_data: np.ndarray) -> EmotionResult:
        """
        Classify emotion from EEG data.

        Args:
            eeg_data: shape (14, 256) — 14 channels, 256 timepoints (2s at 128 Hz).
                Already preprocessed (filtered, normalized) to match DREAMER format.
                If longer than 256 samples, uses the last 256 (most recent 2 seconds).

        Returns:
            EmotionResult with valence, arousal, quadrant, and confidence.
        """
        if self.backend == "dreamer":
            return self._classify_dreamer(eeg_data)
        raise ValueError(f"Backend '{self.backend}' does not support classify()")

    def _classify_dreamer(self, eeg_data: np.ndarray) -> EmotionResult:
        """Run arousal + valence classifiers and combine into quadrant."""
        if eeg_data.ndim != 2 or eeg_data.shape[0] != self.DREAMER_CHANNELS:
            raise ValueError(
                f"Expected shape ({self.DREAMER_CHANNELS}, T), "
                f"got {eeg_data.shape}"
            )

        # Crop/pad to expected timepoints
        if eeg_data.shape[1] > self.DREAMER_TIMEPOINTS:
            eeg_data = eeg_data[:, -self.DREAMER_TIMEPOINTS:]
        elif eeg_data.shape[1] < self.DREAMER_TIMEPOINTS:
            pad_width = self.DREAMER_TIMEPOINTS - eeg_data.shape[1]
            eeg_data = np.pad(eeg_data, ((0, 0), (pad_width, 0)), mode="constant")

        x = torch.from_numpy(eeg_data).float()

        if self._model_name in ("tsception", "eegnet"):
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, 14, 256)
        else:
            x = x.unsqueeze(0)               # (1, 14, 256)

        x = x.to(self._device)

        with torch.no_grad():
            arousal_logits = self._arousal_model(x)
            valence_logits = self._valence_model(x)

        arousal_probs = torch.softmax(arousal_logits, dim=1)[0]
        valence_probs = torch.softmax(valence_logits, dim=1)[0]

        arousal_high_prob = arousal_probs[1].item()
        valence_high_prob = valence_probs[1].item()

        arousal_class = "high" if arousal_high_prob >= 0.5 else "low"
        valence_class = "high" if valence_high_prob >= 0.5 else "low"

        quadrant = f"{'H' if valence_class == 'high' else 'L'}V{'H' if arousal_class == 'high' else 'L'}A"

        arousal_conf = max(arousal_high_prob, 1 - arousal_high_prob)
        valence_conf = max(valence_high_prob, 1 - valence_high_prob)
        avg_confidence = (arousal_conf + valence_conf) / 2

        return EmotionResult(
            valence=valence_high_prob,
            arousal=arousal_high_prob,
            valence_class=valence_class,
            arousal_class=arousal_class,
            quadrant=quadrant,
            confidence=avg_confidence,
        )
