"""
LALM (Large Audio Language Model) Judge for music emotion evaluation.

Sends generated audio to an LLM with audio understanding (e.g., Gemini 2.5 Pro)
and asks it to classify the music's emotional quadrant on the valence-arousal
circumplex. This provides a semantic emotion-level evaluation that complements
the lexical audio-text matching of CLAP scores.

Usage:
    judge = GeminiJudge(model="gemini-2.5-pro")
    result = judge.judge("path/to/audio.wav")

    # Batch with concurrency:
    results = judge.judge_batch(["a.wav", "b.wav", ...], max_workers=5)
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quadrant definitions (must match src/biosignal/classifier.py)
# ---------------------------------------------------------------------------

QUADRANT_OPTIONS = {
    "HVHA": (
        "High arousal, positive valence — excited, joyful, energetic music "
        "(e.g., upbeat dance, triumphant fanfare)"
    ),
    "HVLA": (
        "Low arousal, positive valence — calm, serene, peaceful music "
        "(e.g., gentle piano, soft ambient)"
    ),
    "LVHA": (
        "Low valence, high arousal — anxious, angry, agitated music "
        "(e.g., dissonant strings, aggressive percussion)"
    ),
    "LVLA": (
        "Low valence, low arousal — sad, melancholic, depressed music "
        "(e.g., slow minor-key ballad)"
    ),
}

VALID_QUADRANTS = set(QUADRANT_OPTIONS.keys())

# What quadrant each therapeutic template is actually describing.
# This is used by compute_prompt_alignment_rate() in evaluate.py.
# Review with the team — update if templates change.
TEMPLATE_PROMPT_QUADRANT = {
    "LVHA": "HVLA",  # anxious → calm, soothing (low arousal, positive valence)
    "LVLA": "HVHA",  # sad → uplifting, energizing (high arousal, positive valence)
    "HVHA": "HVLA",  # agitated → grounding, serene (low arousal, positive valence)
    "HVLA": "HVLA",  # content → maintain calm (low arousal, positive valence)
}

SYSTEM_INSTRUCTION = (
    "You are an expert music psychologist evaluating audio clips on the "
    "valence-arousal circumplex (Russell, 1980). You will hear a short music "
    "clip and classify its emotional character into one of four quadrants. "
    "Listen carefully to the audio itself — do not rely on metadata or "
    "assumptions. Output only a JSON object matching the requested schema."
)

PAIRWISE_SYSTEM_INSTRUCTION = (
    "You are an expert music psychologist. You will hear multiple short music "
    "clips and judge which one best matches a given emotional target. "
    "Listen carefully to each clip's tempo, instrumentation, dynamics, and "
    "harmony. Output only a JSON object matching the requested schema."
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class JudgeResult:
    """Result from a single LALM judge call."""
    predicted_quadrant: str          # one of HVHA, HVLA, LVHA, LVLA
    raw_response: str                # full JSON string from the model
    reasoning: str                   # model's one-sentence justification
    option_order: list[str]          # e.g. ["LVLA", "HVHA", "LVHA", "HVLA"]
    latency_ms: float
    error: Optional[str] = None      # non-None if the call failed

    @staticmethod
    def make_error(error_msg: str) -> JudgeResult:
        return JudgeResult(
            predicted_quadrant="",
            raw_response="",
            reasoning="",
            option_order=[],
            latency_ms=0.0,
            error=error_msg,
        )


@dataclass
class PairwiseResult:
    """Result from a pairwise/listwise preference judge call."""
    winner: str                     # condition name: "therapeutic", "non_inverted", "fixed_calm"
    raw_response: str
    reasoning: str
    presentation_order: list[str]   # randomized order of conditions
    latency_ms: float
    error: Optional[str] = None

    @staticmethod
    def make_error(error_msg: str) -> PairwiseResult:
        return PairwiseResult(
            winner="",
            raw_response="",
            reasoning="",
            presentation_order=[],
            latency_ms=0.0,
            error=error_msg,
        )


# ---------------------------------------------------------------------------
# Option randomization
# ---------------------------------------------------------------------------

def _randomize_options(seed: int) -> tuple[dict[str, str], list[str]]:
    """
    Randomize the A/B/C/D assignment of quadrant options.

    Returns:
        letter_to_quad: {"A": "LVLA", "B": "HVHA", ...}
        order: ["LVLA", "HVHA", ...]  (for logging)
    """
    quads = list(QUADRANT_OPTIONS.keys())
    rng = random.Random(seed)
    rng.shuffle(quads)
    letters = ["A", "B", "C", "D"]
    letter_to_quad = {letter: quad for letter, quad in zip(letters, quads)}
    return letter_to_quad, quads


def _build_user_prompt(letter_to_quad: dict[str, str]) -> str:
    """Build the MCQ user prompt with randomized option order."""
    lines = [
        "Listen to the attached audio clip. Which quadrant best describes "
        "the music's emotional character?\n"
    ]
    for letter in ["A", "B", "C", "D"]:
        quad = letter_to_quad[letter]
        lines.append(f"({letter}) {QUADRANT_OPTIONS[quad]}")
    lines.append(
        "\nRespond with the letter of your choice and a one-sentence "
        "justification grounded in audio features (tempo, instrumentation, "
        "dynamics, harmony)."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

class _DiskCache:
    """Simple file-based cache for judge results."""

    def __init__(self, cache_dir: str | Path, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, audio_path: str, model: str, prompt_version: str,
             option_order: list[str]) -> str:
        audio_bytes = Path(audio_path).read_bytes()
        blob = (
            hashlib.sha256(audio_bytes).hexdigest()
            + model + prompt_version + ",".join(option_order)
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, audio_path: str, model: str, prompt_version: str,
            option_order: list[str]) -> Optional[JudgeResult]:
        if not self.enabled:
            return None
        key = self._key(audio_path, model, prompt_version, option_order)
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return JudgeResult(**data)
        except Exception:
            return None

    def put(self, audio_path: str, model: str, prompt_version: str,
            option_order: list[str], result: JudgeResult):
        if not self.enabled:
            return
        key = self._key(audio_path, model, prompt_version, option_order)
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps(asdict(result), ensure_ascii=False))


# ---------------------------------------------------------------------------
# GeminiJudge
# ---------------------------------------------------------------------------

class GeminiJudge:
    """
    Classifies music emotion via Gemini's audio understanding.

    Sends audio inline (no File API) and uses structured JSON output
    to get a reliable quadrant classification.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        temperature: float = 0.0,
        max_retries: int = 3,
        cache_dir: str | Path = "outputs/judge_cache",
        use_cache: bool = True,
        prompt_version: str = "v1",
    ):
        self.model_name = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.prompt_version = prompt_version

        self._cache = _DiskCache(cache_dir, enabled=use_cache)
        self._init_client()

    def _init_client(self):
        """Initialize the Gemini API client."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required for GeminiJudge. "
                "Install with: pip install google-generativeai"
            )

        import os
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. "
                "Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=SYSTEM_INSTRUCTION,
        )
        logger.info(f"GeminiJudge initialized: model={self.model_name}")

    def judge(self, audio_path: str | Path) -> JudgeResult:
        """
        Judge a single audio file's emotional quadrant.

        Args:
            audio_path: Path to a WAV file.

        Returns:
            JudgeResult with predicted_quadrant and reasoning.
        """
        audio_path = str(audio_path)

        # Deterministic seed from file path for reproducible option order
        seed = int(hashlib.md5(audio_path.encode()).hexdigest(), 16) % (2**31)
        letter_to_quad, option_order = _randomize_options(seed)

        # Check cache
        cached = self._cache.get(
            audio_path, self.model_name, self.prompt_version, option_order
        )
        if cached is not None:
            logger.debug(f"Cache hit: {audio_path}")
            return cached

        # Build prompt
        user_prompt = _build_user_prompt(letter_to_quad)

        # Read audio bytes inline
        audio_bytes = Path(audio_path).read_bytes()

        # Call Gemini with retries
        result = self._call_with_retry(
            audio_bytes, user_prompt, letter_to_quad, option_order
        )

        # Cache successful results
        if result.error is None:
            self._cache.put(
                audio_path, self.model_name, self.prompt_version,
                option_order, result
            )

        return result

    def _call_with_retry(
        self,
        audio_bytes: bytes,
        user_prompt: str,
        letter_to_quad: dict[str, str],
        option_order: list[str],
    ) -> JudgeResult:
        """Call Gemini API with exponential backoff retry."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                response = self._model.generate_content(
                    [
                        {"mime_type": "audio/wav", "data": audio_bytes},
                        user_prompt,
                    ],
                    generation_config={
                        "temperature": self.temperature,
                        "response_mime_type": "application/json",
                        "response_schema": {
                            "type": "object",
                            "properties": {
                                "choice": {
                                    "type": "string",
                                    "enum": ["A", "B", "C", "D"],
                                },
                                "reasoning": {"type": "string"},
                            },
                            "required": ["choice", "reasoning"],
                        },
                    },
                )
                latency_ms = (time.time() - t0) * 1000

                raw_text = response.text.strip()
                parsed = json.loads(raw_text)
                choice_letter = parsed["choice"].upper()
                reasoning = parsed.get("reasoning", "")

                if choice_letter not in letter_to_quad:
                    return JudgeResult(
                        predicted_quadrant="",
                        raw_response=raw_text,
                        reasoning=reasoning,
                        option_order=option_order,
                        latency_ms=latency_ms,
                        error=f"Invalid choice letter: {choice_letter}",
                    )

                predicted_quadrant = letter_to_quad[choice_letter]

                return JudgeResult(
                    predicted_quadrant=predicted_quadrant,
                    raw_response=raw_text,
                    reasoning=reasoning,
                    option_order=option_order,
                    latency_ms=latency_ms,
                )

            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    f"Gemini call failed (attempt {attempt+1}/{self.max_retries}): "
                    f"{e}. Retrying in {wait}s..."
                )
                time.sleep(wait)

        return JudgeResult.make_error(f"All {self.max_retries} retries failed: {last_error}")

    def judge_batch(
        self,
        audio_paths: list[str | Path],
        max_workers: int = 5,
    ) -> list[JudgeResult]:
        """
        Judge multiple audio files concurrently.

        Args:
            audio_paths: List of paths to WAV files.
            max_workers: Maximum concurrent API calls.

        Returns:
            List of JudgeResults in the same order as audio_paths.
        """
        results: list[Optional[JudgeResult]] = [None] * len(audio_paths)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.judge, path): i
                for i, path in enumerate(audio_paths)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = JudgeResult.make_error(str(e))

        n_ok = sum(1 for r in results if r and r.error is None)
        n_err = len(results) - n_ok
        logger.info(
            f"Judge batch complete: {n_ok} succeeded, {n_err} failed "
            f"out of {len(audio_paths)}"
        )
        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Pairwise preference
    # ------------------------------------------------------------------

    def judge_pairwise(
        self,
        audio_paths: dict[str, str | Path],
        target_quadrant: str,
    ) -> PairwiseResult:
        """
        Given multiple audio clips, pick the one that best matches a target emotion.

        Args:
            audio_paths: {condition_name: wav_path}, e.g.
                {"therapeutic": "a.wav", "non_inverted": "b.wav", "fixed_calm": "c.wav"}
            target_quadrant: The target emotion quadrant (e.g. "HVLA").

        Returns:
            PairwiseResult with the winning condition name.
        """
        conditions = list(audio_paths.keys())

        # Deterministic shuffle from sorted paths
        path_str = "|".join(str(audio_paths[c]) for c in sorted(conditions))
        seed = int(hashlib.md5(path_str.encode()).hexdigest(), 16) % (2**31)
        rng = random.Random(seed)
        presentation_order = conditions.copy()
        rng.shuffle(presentation_order)

        letters = ["A", "B", "C"][:len(presentation_order)]
        letter_to_cond = {l: c for l, c in zip(letters, presentation_order)}

        # Build prompt
        target_desc = QUADRANT_OPTIONS.get(target_quadrant, QUADRANT_OPTIONS["HVLA"])
        prompt_lines = [
            f"You will hear {len(letters)} music clips labeled {', '.join(letters)}. "
            f"Which clip best expresses the following emotional target?\n",
            f"Target: {target_desc}\n",
            "Respond with the letter of the clip that best matches the target, "
            "and a one-sentence justification grounded in audio features "
            "(tempo, instrumentation, dynamics, harmony).",
        ]
        user_prompt = "\n".join(prompt_lines)

        # Build content: interleave audio clips with labels
        content = []
        for letter in letters:
            cond = letter_to_cond[letter]
            audio_bytes = Path(audio_paths[cond]).read_bytes()
            content.append(f"Clip {letter}:")
            content.append({"mime_type": "audio/wav", "data": audio_bytes})
        content.append(user_prompt)

        # Call with retry
        result = self._call_pairwise_with_retry(
            content, letter_to_cond, presentation_order, letters
        )
        return result

    def _call_pairwise_with_retry(
        self,
        content: list,
        letter_to_cond: dict[str, str],
        presentation_order: list[str],
        valid_letters: list[str],
    ) -> PairwiseResult:
        """Call Gemini for pairwise preference with retries."""
        import google.generativeai as genai

        pairwise_model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=PAIRWISE_SYSTEM_INSTRUCTION,
        )

        last_error = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                response = pairwise_model.generate_content(
                    content,
                    generation_config={
                        "temperature": self.temperature,
                        "response_mime_type": "application/json",
                        "response_schema": {
                            "type": "object",
                            "properties": {
                                "choice": {
                                    "type": "string",
                                    "enum": valid_letters,
                                },
                                "reasoning": {"type": "string"},
                            },
                            "required": ["choice", "reasoning"],
                        },
                    },
                )
                latency_ms = (time.time() - t0) * 1000

                raw_text = response.text.strip()
                parsed = json.loads(raw_text)
                choice_letter = parsed["choice"].upper()
                reasoning = parsed.get("reasoning", "")

                if choice_letter not in letter_to_cond:
                    return PairwiseResult(
                        winner="",
                        raw_response=raw_text,
                        reasoning=reasoning,
                        presentation_order=presentation_order,
                        latency_ms=latency_ms,
                        error=f"Invalid choice: {choice_letter}",
                    )

                return PairwiseResult(
                    winner=letter_to_cond[choice_letter],
                    raw_response=raw_text,
                    reasoning=reasoning,
                    presentation_order=presentation_order,
                    latency_ms=latency_ms,
                )

            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    f"Pairwise call failed (attempt {attempt+1}/{self.max_retries}): "
                    f"{e}. Retrying in {wait}s..."
                )
                time.sleep(wait)

        return PairwiseResult.make_error(
            f"All {self.max_retries} retries failed: {last_error}"
        )

    def judge_pairwise_batch(
        self,
        audio_paths_list: list[dict[str, str | Path]],
        target_quadrants: list[str],
        max_workers: int = 5,
    ) -> list[PairwiseResult]:
        """
        Run pairwise preference on multiple samples concurrently.

        Args:
            audio_paths_list: List of {condition: wav_path} dicts, one per sample.
            target_quadrants: Target quadrant per sample.
            max_workers: Max concurrent API calls.
        """
        results: list[Optional[PairwiseResult]] = [None] * len(audio_paths_list)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.judge_pairwise, paths, tq): i
                for i, (paths, tq) in enumerate(
                    zip(audio_paths_list, target_quadrants)
                )
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = PairwiseResult.make_error(str(e))

        n_ok = sum(1 for r in results if r and r.error is None)
        n_err = len(results) - n_ok
        logger.info(
            f"Pairwise batch complete: {n_ok} succeeded, {n_err} failed "
            f"out of {len(audio_paths_list)}"
        )
        return results  # type: ignore[return-value]

    @classmethod
    def from_config(cls, cfg: dict) -> GeminiJudge:
        """Create a GeminiJudge from the judge section of the config."""
        judge_cfg = cfg.get("judge", {})
        return cls(
            model=judge_cfg.get("model", "gemini-2.5-pro"),
            temperature=judge_cfg.get("temperature", 0.0),
            max_retries=judge_cfg.get("max_retries", 3),
            cache_dir=judge_cfg.get("cache_dir", "outputs/judge_cache"),
            use_cache=judge_cfg.get("use_cache", True),
            prompt_version=judge_cfg.get("prompt_version", "v1"),
        )
