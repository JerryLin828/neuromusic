"""
Emotion → Music Prompt generator.

Implements the "Affective Pivot": given a detected emotional state,
generate a text prompt for music that therapeutically INVERTS that state
(e.g., anxious → calm music, depressed → uplifting music).

Two backends:
  1. Rule-based templates (no dependencies, deterministic)
  2. Gemini API for richer, varied prompts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EmotionState:
    """Emotion detection output from the biosignal classifier."""
    valence: float      # probability of high-valence (0-1)
    arousal: float      # probability of high-arousal (0-1)
    label: str = ""     # discrete quadrant label, e.g. "HVHA"

    @property
    def quadrant(self) -> str:
        if self.label:
            return self.label
        v_high = self.valence >= 0.5
        a_high = self.arousal >= 0.5
        if v_high and a_high:
            return "HVHA"
        elif v_high and not a_high:
            return "HVLA"
        elif not v_high and a_high:
            return "LVHA"
        else:
            return "LVLA"


# ---------------------------------------------------------------------------
# Therapeutic inversion: map detected quadrant → target music characteristics
# ---------------------------------------------------------------------------

THERAPEUTIC_TEMPLATES: dict[str, str] = {
    # Anxious (Low Valence, High Arousal) → calm, soothing music
    "LVHA": (
        "A slow, gentle ambient piece with soft piano and warm synthesizer pads, "
        "creating a calm and peaceful atmosphere at 60 BPM in a major key. "
        "The texture is smooth and flowing, with no sudden changes."
    ),
    # Sad / Depressed (Low Valence, Low Arousal) → uplifting, gently energizing
    "LVLA": (
        "A warm, uplifting acoustic guitar melody with gentle percussion and "
        "soft strings, hopeful and bright, moderate tempo around 100 BPM. "
        "The mood gradually builds with an optimistic, encouraging feel."
    ),
    # Agitated / Manic (High Valence, High Arousal) → grounded, calming
    "HVHA": (
        "A meditative ambient soundscape with flowing strings and soft nature "
        "textures, serene and spacious, around 70 BPM. Gradually calming with "
        "gentle harmonic movement and a sense of stillness."
    ),
    # Content / Relaxed (High Valence, Low Arousal) → maintain with gentle engagement
    "HVLA": (
        "A soft, contemplative piano piece with subtle ambient pads, maintaining "
        "a peaceful and content mood at 80 BPM. Warm and introspective with "
        "delicate melodic phrases in a major key."
    ),
}

LLM_SYSTEM_PROMPT = """\
You are a music therapy AI. Given a patient's detected emotional state \
(valence and arousal as probabilities from 0 to 1, where 0.5 is the \
midpoint), generate a short music description that would therapeutically \
guide them toward emotional balance.

Rules:
- INVERT the detected emotion to produce therapeutic music:
  * Anxious (low valence, high arousal) → calm, soothing music
  * Depressed (low valence, low arousal) → gently uplifting music
  * Agitated (high valence, high arousal) → grounding, serene music
  * Content (high valence, low arousal) → maintain with gentle engagement
- Output ONLY the music description, nothing else.
- Include: tempo (BPM), instruments, mood, texture, and key/mode.
- Keep it to 2-3 sentences.
- The description will be fed directly to a text-to-music AI model, \
so make it concrete and descriptive.\
"""


class PromptGenerator:
    """Generate therapeutic music prompts from emotion states."""

    def __init__(self, backend: str = "template", **kwargs):
        """
        Args:
            backend: "template" or "gemini"
            kwargs: backend-specific options (model name, etc.)
        """
        self.backend = backend
        self.kwargs = kwargs

        if backend == "gemini":
            self._init_gemini(**kwargs)
        elif backend == "template":
            pass
        else:
            raise ValueError(f"Unknown bridge backend: '{backend}'. Available: 'template', 'gemini'")

    def _init_gemini(self, model: str = "gemini-2.0-flash", **_kwargs):
        """
        Initialize Gemini API client.

        Requires GEMINI_API_KEY or GOOGLE_API_KEY environment variable.
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required for the Gemini backend. "
                "Install with: pip install google-generativeai"
            )

        import os
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY env var."
            )

        genai.configure(api_key=api_key)
        self._gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=LLM_SYSTEM_PROMPT,
        )
        self._llm_model_name = model
        logger.info(f"Gemini bridge initialized with model: {model}")

    def generate(self, emotion: EmotionState) -> str:
        """Generate a therapeutic music prompt for the given emotion state."""
        if self.backend == "template":
            return self._generate_template(emotion)
        elif self.backend == "gemini":
            return self._generate_gemini(emotion)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _generate_template(self, emotion: EmotionState) -> str:
        quadrant = emotion.quadrant
        prompt = THERAPEUTIC_TEMPLATES.get(quadrant)
        if prompt is None:
            logger.warning(f"No template for quadrant {quadrant}, defaulting to LVHA (calming)")
            prompt = THERAPEUTIC_TEMPLATES["LVHA"]
        return prompt

    def _generate_gemini(self, emotion: EmotionState) -> str:
        user_msg = (
            f"Patient's detected emotional state: "
            f"valence = {emotion.valence:.2f}, arousal = {emotion.arousal:.2f} "
            f"(quadrant: {emotion.quadrant})."
        )
        response = self._gemini_model.generate_content(
            user_msg,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 200,
            },
        )
        return response.text.strip()
