"""
MusicGen wrapper for text-to-music generation.

Uses Meta's audiocraft library with pretrained MusicGen checkpoints.
Fully inference-only — no training required.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


class MusicGenerator:
    """Wrapper around MusicGen for therapeutic music generation."""

    def __init__(
        self,
        model_name: str = "facebook/musicgen-small",
        duration: float = 15.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        cfg_coef: float = 3.0,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: HuggingFace model ID. Options:
                - facebook/musicgen-small  (300M, ~5 GB VRAM)
                - facebook/musicgen-medium (1.5B, ~12 GB VRAM)
                - facebook/musicgen-large  (3.3B, ~24 GB VRAM)
                - facebook/musicgen-melody (1.5B, supports melody conditioning)
            duration: Length of generated audio in seconds.
            temperature: Sampling temperature (higher = more varied).
            top_k: Top-k sampling parameter.
            top_p: Top-p (nucleus) sampling parameter. 0 disables it.
            cfg_coef: Classifier-free guidance coefficient.
            device: "cuda", "cpu", or None (auto-detect).
        """
        self.model_name = model_name
        self.duration = duration
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._lock = threading.Lock()
        self._generation_params = dict(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_coef=cfg_coef,
        )

    @property
    def model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            logger.info(f"Loading MusicGen model: {self.model_name} on {self.device}")
            try:
                from audiocraft.models import MusicGen
            except ImportError:
                raise ImportError(
                    "audiocraft is required for music generation. "
                    "Install with: pip install audiocraft"
                )
            self._model = MusicGen.get_pretrained(self.model_name, device=self.device)
            self._model.set_generation_params(**self._generation_params)
            logger.info("MusicGen loaded successfully")
        return self._model

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    def generate(self, prompt: str) -> np.ndarray:
        """
        Generate audio from a text prompt.

        Args:
            prompt: Text description of the desired music.

        Returns:
            Audio waveform as numpy array, shape (num_samples,).
        """
        logger.info(f"Generating {self.duration}s of audio for prompt: {prompt[:80]}...")
        with self._lock:
            wav = self.model.generate([prompt])
        audio = wav[0, 0].cpu().numpy()
        logger.info(f"Generated audio: {len(audio)} samples at {self.sample_rate} Hz")
        return audio

    def generate_batch(self, prompts: list[str]) -> list[np.ndarray]:
        """Generate audio for multiple prompts in a single batch."""
        logger.info(f"Generating batch of {len(prompts)} clips...")
        with self._lock:
            wavs = self.model.generate(prompts)
        return [wavs[i, 0].cpu().numpy() for i in range(len(prompts))]

    def save_audio(self, audio: np.ndarray, path: str | Path):
        """Save audio to a WAV file."""
        import torchaudio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
        torchaudio.save(str(path), tensor, self.sample_rate)
        logger.info(f"Saved audio to {path}")

    def set_duration(self, duration: float):
        """Update generation duration."""
        self.duration = duration
        self._generation_params["duration"] = duration
        if self._model is not None:
            self._model.set_generation_params(**self._generation_params)
