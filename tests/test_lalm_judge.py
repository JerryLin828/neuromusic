"""
Tests for evaluation/lalm_judge.py

Run:
    pytest tests/test_lalm_judge.py -v
    pytest tests/test_lalm_judge.py -v -k "not api"   # skip live API tests
"""

import json
import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evaluation.lalm_judge import (
    GeminiJudge,
    JudgeResult,
    QUADRANT_OPTIONS,
    TEMPLATE_PROMPT_QUADRANT,
    VALID_QUADRANTS,
    _build_user_prompt,
    _randomize_options,
    _DiskCache,
)
from evaluation.evaluate import (
    compute_inversion_rate,
    compute_prompt_alignment_rate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path: str | Path, duration_s: float = 1.0, sr: int = 32000):
    """Create a minimal valid WAV file (silence)."""
    n_samples = int(duration_s * sr)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        # WAV header
        data_size = n_samples * 2  # 16-bit mono
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))       # chunk size
        f.write(struct.pack("<H", 1))        # PCM
        f.write(struct.pack("<H", 1))        # mono
        f.write(struct.pack("<I", sr))       # sample rate
        f.write(struct.pack("<I", sr * 2))   # byte rate
        f.write(struct.pack("<H", 2))        # block align
        f.write(struct.pack("<H", 16))       # bits per sample
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)
    return str(path)


# ---------------------------------------------------------------------------
# Test: Option randomization
# ---------------------------------------------------------------------------

class TestOptionRandomization:
    def test_all_quadrants_present(self):
        letter_to_quad, order = _randomize_options(seed=42)
        assert set(letter_to_quad.values()) == VALID_QUADRANTS
        assert set(order) == VALID_QUADRANTS

    def test_deterministic(self):
        a1, o1 = _randomize_options(seed=123)
        a2, o2 = _randomize_options(seed=123)
        assert a1 == a2
        assert o1 == o2

    def test_different_seeds_different_order(self):
        _, o1 = _randomize_options(seed=1)
        _, o2 = _randomize_options(seed=2)
        # With 4! = 24 permutations, very unlikely to collide
        # but not impossible — just check they're valid
        assert set(o1) == VALID_QUADRANTS
        assert set(o2) == VALID_QUADRANTS

    def test_four_letters(self):
        letter_to_quad, _ = _randomize_options(seed=0)
        assert set(letter_to_quad.keys()) == {"A", "B", "C", "D"}


# ---------------------------------------------------------------------------
# Test: Prompt construction
# ---------------------------------------------------------------------------

class TestPromptConstruction:
    def test_prompt_contains_all_options(self):
        letter_to_quad, _ = _randomize_options(seed=42)
        prompt = _build_user_prompt(letter_to_quad)
        for letter in ["A", "B", "C", "D"]:
            assert f"({letter})" in prompt
        for quad, desc in QUADRANT_OPTIONS.items():
            assert desc in prompt

    def test_prompt_has_instruction(self):
        letter_to_quad, _ = _randomize_options(seed=42)
        prompt = _build_user_prompt(letter_to_quad)
        assert "Listen to the attached audio clip" in prompt
        assert "justification" in prompt


# ---------------------------------------------------------------------------
# Test: JudgeResult
# ---------------------------------------------------------------------------

class TestJudgeResult:
    def test_make_error(self):
        r = JudgeResult.make_error("timeout")
        assert r.error == "timeout"
        assert r.predicted_quadrant == ""

    def test_fields(self):
        r = JudgeResult(
            predicted_quadrant="HVLA",
            raw_response='{"choice": "A", "reasoning": "calm"}',
            reasoning="calm",
            option_order=["HVLA", "HVHA", "LVHA", "LVLA"],
            latency_ms=1234.5,
        )
        assert r.predicted_quadrant == "HVLA"
        assert r.error is None


# ---------------------------------------------------------------------------
# Test: Disk cache
# ---------------------------------------------------------------------------

class TestDiskCache:
    def test_cache_miss_then_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = _DiskCache(tmpdir, enabled=True)
            wav = _make_wav(Path(tmpdir) / "test.wav")

            # Miss
            result = cache.get(wav, "gemini-2.5-pro", "v1", ["HVHA", "HVLA", "LVHA", "LVLA"])
            assert result is None

            # Put
            jr = JudgeResult(
                predicted_quadrant="HVLA",
                raw_response="{}",
                reasoning="test",
                option_order=["HVHA", "HVLA", "LVHA", "LVLA"],
                latency_ms=100.0,
            )
            cache.put(wav, "gemini-2.5-pro", "v1", ["HVHA", "HVLA", "LVHA", "LVLA"], jr)

            # Hit
            cached = cache.get(wav, "gemini-2.5-pro", "v1", ["HVHA", "HVLA", "LVHA", "LVLA"])
            assert cached is not None
            assert cached.predicted_quadrant == "HVLA"
            assert cached.reasoning == "test"

    def test_cache_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = _DiskCache(tmpdir, enabled=False)
            wav = _make_wav(Path(tmpdir) / "test.wav")

            jr = JudgeResult(
                predicted_quadrant="HVLA",
                raw_response="{}",
                reasoning="test",
                option_order=[],
                latency_ms=100.0,
            )
            cache.put(wav, "model", "v1", [], jr)
            assert cache.get(wav, "model", "v1", []) is None

    def test_different_model_different_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = _DiskCache(tmpdir, enabled=True)
            wav = _make_wav(Path(tmpdir) / "test.wav")
            order = ["HVHA", "HVLA", "LVHA", "LVLA"]

            jr = JudgeResult(
                predicted_quadrant="HVLA",
                raw_response="{}",
                reasoning="test",
                option_order=order,
                latency_ms=100.0,
            )
            cache.put(wav, "gemini-2.5-pro", "v1", order, jr)

            # Different model → miss
            assert cache.get(wav, "gemini-2.5-flash", "v1", order) is None
            # Same model → hit
            assert cache.get(wav, "gemini-2.5-pro", "v1", order) is not None


# ---------------------------------------------------------------------------
# Test: GeminiJudge with mocked API
# ---------------------------------------------------------------------------

class TestGeminiJudgeMocked:
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"})
    @patch("evaluation.lalm_judge.GeminiJudge._init_client")
    def test_judge_parses_response(self, mock_init):
        judge = GeminiJudge(use_cache=False)

        # Mock the model
        mock_response = MagicMock()
        mock_response.text = '{"choice": "B", "reasoning": "calm piano, slow tempo"}'
        judge._model = MagicMock()
        judge._model.generate_content.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _make_wav(Path(tmpdir) / "test.wav")
            result = judge.judge(wav)

        assert result.error is None
        assert result.predicted_quadrant in VALID_QUADRANTS
        assert result.reasoning == "calm piano, slow tempo"
        assert len(result.option_order) == 4

    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"})
    @patch("evaluation.lalm_judge.GeminiJudge._init_client")
    def test_judge_handles_api_failure(self, mock_init):
        judge = GeminiJudge(use_cache=False, max_retries=2)
        judge._model = MagicMock()
        judge._model.generate_content.side_effect = RuntimeError("API down")

        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _make_wav(Path(tmpdir) / "test.wav")
            result = judge.judge(wav)

        assert result.error is not None
        assert "retries failed" in result.error

    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"})
    @patch("evaluation.lalm_judge.GeminiJudge._init_client")
    def test_judge_batch(self, mock_init):
        judge = GeminiJudge(use_cache=False)

        mock_response = MagicMock()
        mock_response.text = '{"choice": "A", "reasoning": "energetic"}'
        judge._model = MagicMock()
        judge._model.generate_content.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            wavs = [_make_wav(Path(tmpdir) / f"test_{i}.wav") for i in range(3)]
            results = judge.judge_batch(wavs, max_workers=2)

        assert len(results) == 3
        assert all(r.error is None for r in results)


# ---------------------------------------------------------------------------
# Test: Aggregation functions
# ---------------------------------------------------------------------------

class TestAggregation:
    def _make_results(self, quadrants: list[str]) -> list[JudgeResult]:
        return [
            JudgeResult(
                predicted_quadrant=q,
                raw_response="{}",
                reasoning="test",
                option_order=[],
                latency_ms=0,
            )
            for q in quadrants
        ]

    def test_inversion_rate_all_inverted(self):
        judge_results = self._make_results(["HVLA", "HVHA", "LVLA", "LVHA"])
        detected = ["LVHA", "LVLA", "HVHA", "HVLA"]
        result = compute_inversion_rate(judge_results, detected)
        assert result["overall"] == 1.0
        assert result["n_inverted"] == 4

    def test_inversion_rate_none_inverted(self):
        judge_results = self._make_results(["HVHA", "LVLA"])
        detected = ["HVHA", "LVLA"]
        result = compute_inversion_rate(judge_results, detected)
        assert result["overall"] == 0.0

    def test_inversion_rate_skips_errors(self):
        judge_results = [
            JudgeResult.make_error("fail"),
            *self._make_results(["HVLA"]),
        ]
        detected = ["LVHA", "LVHA"]
        result = compute_inversion_rate(judge_results, detected)
        assert result["n_samples"] == 1
        assert result["n_errors"] == 1
        assert result["overall"] == 1.0

    def test_prompt_alignment_all_aligned(self):
        judge_results = self._make_results(["HVLA", "HVHA", "HVLA"])
        prompt_quads = ["HVLA", "HVHA", "HVLA"]
        result = compute_prompt_alignment_rate(judge_results, prompt_quads)
        assert result["overall"] == 1.0

    def test_prompt_alignment_none_aligned(self):
        judge_results = self._make_results(["LVHA", "LVLA"])
        prompt_quads = ["HVLA", "HVHA"]
        result = compute_prompt_alignment_rate(judge_results, prompt_quads)
        assert result["overall"] == 0.0

    def test_confusion_matrix_structure(self):
        judge_results = self._make_results(["HVLA", "HVHA"])
        detected = ["LVHA", "LVLA"]
        result = compute_inversion_rate(judge_results, detected)
        cm = result["confusion_matrix"]
        assert set(cm.keys()) == VALID_QUADRANTS
        for row in cm.values():
            assert set(row.keys()) == VALID_QUADRANTS


# ---------------------------------------------------------------------------
# Test: TEMPLATE_PROMPT_QUADRANT consistency
# ---------------------------------------------------------------------------

class TestTemplateMapping:
    def test_all_quadrants_mapped(self):
        assert set(TEMPLATE_PROMPT_QUADRANT.keys()) == VALID_QUADRANTS

    def test_all_targets_valid(self):
        for target in TEMPLATE_PROMPT_QUADRANT.values():
            assert target in VALID_QUADRANTS


# ---------------------------------------------------------------------------
# Test: Live API (only runs if GEMINI_API_KEY is set)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set — skipping live API test",
)
class TestGeminiJudgeLive:
    def test_smoke_single_call(self):
        """Smoke test: judge a short silence WAV, just verify we get a valid response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _make_wav(Path(tmpdir) / "silence.wav", duration_s=3.0)
            judge = GeminiJudge(
                model="gemini-2.5-flash",  # cheaper for smoke test
                use_cache=False,
            )
            result = judge.judge(wav)

        assert result.error is None, f"API call failed: {result.error}"
        assert result.predicted_quadrant in VALID_QUADRANTS
        assert len(result.reasoning) > 0
        assert result.latency_ms > 0
