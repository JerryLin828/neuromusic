"""
Emotion → Music Prompt generator.

Implements the "Affective Pivot": given a detected emotional state,
generate a text prompt for music that therapeutically INVERTS that state
(e.g., anxious → calm music, depressed → uplifting music).

Three backends:
  1. "template"   — Rule-based templates keyed on the discrete quadrant
                    (deterministic, no dependencies, but discards the
                    continuous valence/arousal magnitudes).
  2. "gemini"     — Gemini API for richer, varied prompts.
  3. "continuous" — Parametric prompt builder that preserves the float
                    (valence, arousal) signal end-to-end. Computes a
                    therapeutic target (target_v, target_a) by reflecting
                    the detected point across (0.5, 0.5) with an
                    aggressiveness coefficient α, then maps target_v/target_a
                    to BPM, key, mood vocabulary, and instrumentation.
                    Deterministic, no API calls.
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
midpoint), generate a music description that would therapeutically \
guide them toward emotional balance.

Rules:
- INVERT the detected emotion to produce therapeutic music:
  * Anxious (low valence, high arousal) → calm, soothing music
  * Depressed (low valence, low arousal) → gently uplifting music
  * Agitated (high valence, high arousal) → grounding, serene music
  * Content (high valence, low arousal) → maintain with gentle engagement
- Output ONLY the music description, nothing else. No preamble, no labels.
- You MUST include: tempo (BPM), instruments, mood, texture, and key/mode.
- Write 3-5 sentences. Be vivid, specific, and descriptive.
- Example output format: "A slow, gentle ambient piece with soft piano \
and warm synthesizer pads, creating a calm and peaceful atmosphere at \
60 BPM in a major key. The texture is smooth and flowing, with no sudden \
changes. Delicate string harmonics drift in the background, adding warmth \
and emotional depth."
- The description will be fed directly to a text-to-music AI model, \
so make it concrete and rich with musical detail.\
"""


class PromptGenerator:
    """Generate therapeutic music prompts from emotion states."""

    def __init__(self, backend: str = "template", **kwargs):
        """
        Args:
            backend: "template", "gemini", or "continuous"
            kwargs: backend-specific options (model name, alpha, etc.)
        """
        self.backend = backend
        self.kwargs = kwargs

        if backend == "gemini":
            self._init_gemini(**kwargs)
        elif backend == "template":
            pass
        elif backend == "continuous":
            self._init_continuous(**kwargs)
        else:
            raise ValueError(
                f"Unknown bridge backend: '{backend}'. "
                f"Available: 'template', 'gemini', 'continuous'"
            )

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
        elif self.backend == "continuous":
            return self._generate_continuous(emotion)
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
                "max_output_tokens": 2048,
            },
        )
        return response.text.strip()

    # ---------------------------------------------------------------
    # Continuous backend
    # ---------------------------------------------------------------
    #
    # Preserves the float (valence, arousal) magnitudes that the
    # template and Gemini backends collapse into a discrete quadrant.
    #
    # The therapeutic target is computed per-quadrant to respect the
    # asymmetric clinical goal (don't push happy people sad, don't push
    # calm content people into anxiety, etc.). Within each quadrant,
    # the magnitude of the target is a continuous function of the
    # detected magnitude — i.e., a more deeply sad EEG reading produces
    # a more aggressively uplifting target.
    #
    # `alpha` ∈ [1.0, 1.5] is the inversion aggressiveness. α=1.0 reflects
    # gently across (0.5, 0.5); larger α pushes targets further toward the
    # extremes.

    def _init_continuous(self, alpha: float = 1.2, **_kwargs):
        """
        Args:
            alpha: aggressiveness of the affective inversion. 1.0 is a pure
                reflection across the midpoint; 1.5 pushes harder toward the
                extreme. Values outside [1.0, 1.5] are clipped at runtime.
        """
        self._alpha = float(alpha)
        logger.info(f"Continuous bridge initialized with alpha={self._alpha}")

    @staticmethod
    def _compute_target(detected_v: float, detected_a: float, alpha: float
                        ) -> tuple[float, float]:
        """
        Compute a therapeutic (target_v, target_a) from detected (v, a).

        The mapping is asymmetric across quadrants to follow the existing
        clinical inversion logic:
            HVHA (excited/agitated) → keep V high, drop A
            HVLA (content)          → maintain (no inversion needed)
            LVHA (anxious/angry)    → lift V, drop A
            LVLA (sad/depressed)    → lift V, lift A   (this is the case
                                      that motivated the continuous backend)

        The magnitude of each push is proportional to how far the detected
        value is from the midpoint, scaled by alpha.
        """
        a = max(1.0, min(1.5, alpha))
        v_high = detected_v >= 0.5
        a_high = detected_a >= 0.5

        if v_high and a_high:                 # HVHA → calm but still positive
            target_v = max(detected_v, 0.65)
            target_a = max(0.0, 0.5 - a * (detected_a - 0.5))
        elif v_high and not a_high:           # HVLA → maintain (slight nudge)
            target_v = detected_v
            target_a = detected_a
        elif (not v_high) and a_high:         # LVHA → calm and bright
            target_v = min(1.0, 0.5 + a * (0.5 - detected_v))
            target_a = max(0.0, 0.5 - a * (detected_a - 0.5))
        else:                                 # LVLA → uplift both, push hard
            target_v = min(1.0, 0.5 + a * (0.5 - detected_v))
            target_a = min(1.0, 0.5 + a * (0.5 - detected_a))
        return target_v, target_a

    @staticmethod
    def _bin(value: float, vocab: list[str]) -> str:
        """Pick a word from `vocab` based on where `value` falls in [0, 1]."""
        idx = min(len(vocab) - 1, max(0, int(value * len(vocab))))
        return vocab[idx]

    def _generate_continuous(self, emotion: EmotionState) -> str:
        target_v, target_a = self._compute_target(
            emotion.valence, emotion.arousal, self._alpha
        )

        # BPM scales linearly with target arousal: 60 → 140
        bpm = round(60 + 80 * target_a)

        # Key/mode vocabulary on the valence axis
        if target_v >= 0.7:
            key = "bright major key"
        elif target_v >= 0.55:
            key = "warm major key"
        elif target_v >= 0.45:
            key = "modal/neutral key"
        else:
            key = "minor key"

        arousal_words = [
            "deeply serene",      # [0.0, 0.2)
            "calm and spacious",  # [0.2, 0.4)
            "warm and steady",    # [0.4, 0.6)
            "uplifting",          # [0.6, 0.8)
            "energetic",          # [0.8, 1.0]
        ]
        valence_words = [
            "somber",
            "melancholic",
            "contemplative",
            "hopeful",
            "joyful",
        ]
        arousal_word = self._bin(target_a, arousal_words)
        valence_word = self._bin(target_v, valence_words)

        # Instrumentation graded by target arousal
        if target_a < 0.3:
            instruments = (
                "soft piano, warm synthesizer pads, and gentle string harmonics"
            )
        elif target_a < 0.55:
            instruments = (
                "fingerpicked acoustic guitar, mellow piano, and subtle ambient strings"
            )
        elif target_a < 0.75:
            instruments = (
                "strummed acoustic guitar, bright piano, light percussion, "
                "and rising string lines"
            )
        else:
            instruments = (
                "driving acoustic guitar, bouncing bass, clapping percussion, "
                "soaring strings, and bright brass accents"
            )

        # Energy-front-loaded opener: critical for short MusicGen clips,
        # which don't have time to "build". Skip evolution language.
        if target_a >= 0.7:
            opener = (
                f"An exuberant, {valence_word} and {arousal_word} indie-pop piece "
                f"in a {key}, immediately full and celebratory from the first beat."
            )
        elif target_a >= 0.5:
            opener = (
                f"A {valence_word}, {arousal_word} piece in a {key}, with a clear "
                f"forward-moving pulse from the opening bar."
            )
        else:
            opener = (
                f"A {arousal_word}, {valence_word} ambient piece in a {key}, "
                f"with smooth, flowing texture and no sudden changes."
            )

        body = (
            f"Tempo around {bpm} BPM, featuring {instruments}. "
            f"The mood is {valence_word} and {arousal_word} throughout."
        )

        # Diagnostics tail (kept short; helps reproducibility but can be
        # stripped at eval time if desired).
        meta = (
            f"[detected v={emotion.valence:.2f}, a={emotion.arousal:.2f} "
            f"→ target v={target_v:.2f}, a={target_a:.2f}, α={self._alpha}]"
        )
        logger.debug(meta)

        return f"{opener} {body}"
