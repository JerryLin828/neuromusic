"""
Improved continuous prompt generator (v2).

Fixes over v1 (_generate_continuous in prompt_generator.py):
  1. Genre tags anchored to (target_v, target_a) — MusicGen responds well to
     genre vocabulary (e.g. "indie-pop", "cinematic orchestral").
  2. Non-repeating mood words — opener and body draw from different word pools
     so the same adjective doesn't appear twice in one prompt.
  3. HVLA case now nudges arousal slightly upward (v1 was a no-op).
  4. Richer instrumentation: 6 levels instead of 4.
  5. Texture/rhythm descriptors added for mid- and high-arousal targets.
  6. Diagnostic meta tag removed from output (was already only logged, but
     clarified here for cleanliness).

Drop-in replacement: same __init__ signature as PromptGenerator(backend="continuous").
Usage:
    from src.bridge.prompt_generator_v2 import PromptGeneratorV2
    gen = PromptGeneratorV2(alpha=1.2)
    prompt = gen.generate(emotion_state)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.bridge.prompt_generator import EmotionState  # reuse the dataclass

logger = logging.getLogger(__name__)


class PromptGeneratorV2:
    """
    Improved therapeutic music prompt generator.

    Accepts the same (alpha) keyword as the v1 continuous backend so it can
    be swapped in without touching configs.
    """

    def __init__(self, alpha: float = 1.2, **_kwargs):
        self._alpha = max(1.0, min(1.5, float(alpha)))
        logger.info(f"PromptGeneratorV2 initialized with alpha={self._alpha}")

    # ------------------------------------------------------------------
    # Public API (matches PromptGenerator)
    # ------------------------------------------------------------------

    def generate(self, emotion: EmotionState) -> str:
        return self._generate(emotion)

    # ------------------------------------------------------------------
    # Target computation (identical to v1 — do not change without a
    # corresponding eval run)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_target(v: float, a: float, alpha: float) -> tuple[float, float]:
        al = max(1.0, min(1.5, alpha))
        v_high = v >= 0.5
        a_high = a >= 0.5

        if v_high and a_high:           # HVHA → calm but positive
            tv = max(v, 0.65)
            ta = max(0.0, 0.5 - al * (a - 0.5))
        elif v_high and not a_high:     # HVLA → gentle engagement nudge (v2 fix)
            tv = v
            ta = min(0.65, a + al * 0.15)
        elif (not v_high) and a_high:   # LVHA → calm + bright
            tv = min(1.0, 0.5 + al * (0.5 - v))
            ta = max(0.0, 0.5 - al * (a - 0.5))
        else:                           # LVLA → uplift both
            tv = min(1.0, 0.5 + al * (0.5 - v))
            ta = min(1.0, 0.5 + al * (0.5 - a))
        return tv, ta

    # ------------------------------------------------------------------
    # Vocabulary tables
    # ------------------------------------------------------------------

    # Genre tags indexed by (valence_bin, arousal_bin) where bins are 0-2
    # (low / mid / high).  Chosen to be concrete genre signals for MusicGen.
    _GENRE: dict[tuple[int, int], str] = {
        (0, 0): "dark ambient",
        (0, 1): "melancholic indie folk",
        (0, 2): "dramatic cinematic orchestral",
        (1, 0): "ambient new-age",
        (1, 1): "mellow lo-fi acoustic",
        (1, 2): "cinematic folk-pop",
        (2, 0): "peaceful ambient piano",
        (2, 1): "warm indie folk",
        (2, 2): "uplifting indie-pop",
    }

    # Opener mood descriptors (used in the first sentence)
    _OPENER_MOOD: list[list[str]] = [
        # arousal low → high
        ["still and contemplative", "gentle and unhurried", "quietly flowing"],   # valence low
        ["warm and grounded",       "easy and unhurried",   "bright and moving"],  # valence mid
        ["serene and hopeful",      "light and forward-moving", "joyful and radiant"],  # valence high
    ]

    # Body mood descriptors (different pool — avoids repeating opener words)
    _BODY_MOOD: list[list[str]] = [
        ["introspective", "subdued",   "tense yet expressive"],
        ["balanced",      "settled",   "gently energised"],
        ["optimistic",    "buoyant",   "exuberant"],
    ]

    # Instrumentation at 6 arousal levels
    _INSTRUMENTS: list[str] = [
        # target_a ∈ [0.0, 0.17)
        "solo piano with long reverb tails and sparse, breathy synthesizer pads",
        # [0.17, 0.33)
        "soft piano, warm cello sustains, and delicate ambient textures",
        # [0.33, 0.5)
        "fingerpicked acoustic guitar, mellow piano, and subtle ambient strings",
        # [0.5, 0.67)
        "strummed acoustic guitar, bright piano chords, light brushed drums, "
        "and warm string lines",
        # [0.67, 0.83)
        "driving acoustic guitar, melodic bass, mid-tempo percussion, "
        "and rising violin lines",
        # [0.83, 1.0]
        "driving electric guitar, punchy bass, energetic drum kit, "
        "bright brass stabs, and soaring string swells",
    ]

    # Texture / rhythm riders (appended for mid–high arousal to give MusicGen
    # more rhythmic guidance, which it uses strongly)
    _RHYTHM_RIDERS: list[str] = [
        "",   # low — no extra rider needed
        "",
        "The rhythm is steady and legato with long phrase arcs.",
        "A clear, walking quarter-note pulse drives the track.",
        "Syncopated rhythmic feel with a pronounced backbeat.",
        "Propulsive, forward-pushing groove with strong downbeats.",
    ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bin3(x: float) -> int:
        """Map [0,1] → {0, 1, 2}."""
        if x < 0.4:
            return 0
        if x < 0.7:
            return 1
        return 2

    @staticmethod
    def _bin6(x: float) -> int:
        """Map [0,1] → {0…5}."""
        return min(5, int(x * 6))

    def _key(self, tv: float) -> str:
        if tv >= 0.72:
            return "bright major key"
        if tv >= 0.57:
            return "warm major key"
        if tv >= 0.43:
            return "modal/neutral key"
        return "minor key"

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _generate(self, emotion: EmotionState) -> str:
        tv, ta = self._compute_target(emotion.valence, emotion.arousal, self._alpha)

        bpm = round(60 + 80 * ta)
        key = self._key(tv)

        vb = self._bin3(tv)
        ab = self._bin3(ta)

        genre       = self._GENRE[(vb, ab)]
        opener_mood = self._OPENER_MOOD[vb][ab]
        body_mood   = self._BODY_MOOD[vb][ab]
        instruments = self._INSTRUMENTS[self._bin6(ta)]
        rhythm      = self._RHYTHM_RIDERS[self._bin6(ta)]

        article = "An" if opener_mood[0].lower() in "aeiou" else "A"
        opener = (
            f"{article} {opener_mood} {genre} piece in a {key}, "
            f"immediately establishing its character from the first bar."
        )
        body = (
            f"Tempo around {bpm} BPM, featuring {instruments}. "
            f"The overall feel is {body_mood} and sustained throughout."
        )

        parts = [opener, body]
        if rhythm:
            parts.append(rhythm)

        return " ".join(parts)
