# Continuous Bridge Backend

> Branch: `emotion_rep`. Adds a third option to the Affective Bridge that preserves the float `(valence, arousal)` magnitudes end-to-end instead of collapsing them into a discrete quadrant.

## Summary of changes

Two files modified, no existing behavior touched:

| File | Change |
|---|---|
| `src/bridge/prompt_generator.py` | Added a new `"continuous"` backend (`_init_continuous`, `_compute_target`, `_generate_continuous`, helpers). The `template` and `gemini` backends are byte-identical to before. |
| `configs/default.yaml` | Added a `bridge.continuous:` config block with an `alpha` knob. Default `bridge.backend` is unchanged (`"gemini"`). |

## Why this exists

The `template` and `gemini` backends both flatten the continuous `(valence, arousal)` signal into one of four quadrant labels before deciding what music to generate. That means a shallow-LVLA reading at `(v=0.45, a=0.45)` and a deeply-LVLA reading at `(v=0.05, a=0.10)` produce the *same* prompt — even though the second person needs a far more aggressively uplifting target.

This was the dominant cause of the `sad → uplifting` quadrant matching at only **1.7%** on Metric 1: every LVLA sample was being mapped to a single calm-leaning prompt full of "gentle / soft / gradually builds" language, which MusicGen-small interprets as ambient. With only 10–15 seconds of audio, the model never has time to "build" out of the calm opening.

## How the continuous mapping works

The therapeutic target is computed by reflecting the detected point across `(0.5, 0.5)` with an aggressiveness coefficient `α`, then mapped to BPM, key, mood vocabulary, and instrumentation parametrically.

```
target_v = 0.5 + α · (0.5 − detected_v)      # for inversion-target quadrants
target_a = 0.5 + α · (0.5 − detected_a)
BPM      = 60 + 80 · target_a                # 60–140 BPM range
key      = bright major | warm major | modal | minor   # by target_v
mood     = somber | melancholic | contemplative | hopeful | joyful   # by target_v
energy   = serene | calm | warm | uplifting | energetic              # by target_a
```

The exact mapping is **asymmetric per detected quadrant** so it follows the existing clinical inversion logic:

| Detected | Target |
|---|---|
| `HVHA` (excited / agitated) | Stays positive while becoming calmer |
| `HVLA` (content) | Maintains |
| `LVHA` (anxious / angry) | Calmed and brightened |
| `LVLA` (sad / depressed) | Aggressively uplifted on **both** axes |

## How to use it

**One-line config change.** Edit `configs/default.yaml`:

```yaml
bridge:
  backend: "continuous"   # was "gemini"
```

Optionally tune `α`:

```yaml
bridge:
  continuous:
    alpha: 1.2   # 1.0 = midpoint reflection, 1.5 = strongly pushed toward extreme
```

Everything else (the eval harness, the demo, the pipeline runner) reads from the same config and picks up the new backend automatically. No other code changes needed.

## What you should see

Three different LVLA inputs at α=1.2 produce three different prompts, with a BPM spread of 33 and no calm-leaning vocabulary:

| Detected `(v, a)` | Magnitude | Generated prompt |
|---|---|---|
| `(0.45, 0.45)` | shallow | *"A contemplative, warm and steady piece in a warm major key... 105 BPM... fingerpicked acoustic guitar, mellow piano, and subtle ambient strings."* |
| `(0.30, 0.30)` | mid | *"An exuberant, hopeful and uplifting indie-pop piece in a bright major key, immediately full and celebratory from the first beat. Tempo around 119 BPM..."* |
| `(0.05, 0.10)` | deep | *"An exuberant, joyful and energetic indie-pop piece in a bright major key, immediately full and celebratory from the first beat. Tempo around 138 BPM, featuring driving acoustic guitar, bouncing bass, clapping percussion, soaring strings, and bright brass accents..."* |

The `LVHA → calm` and `HVHA → calm` paths still produce ambient/serene outputs but preserve positive valence (e.g., *"deeply serene, joyful"*), and `HVLA` correctly maintains.

The `α` coefficient gives you a per-run dial. On the deep-LVLA case:

| α | Tempo |
|---|---|
| 1.0 | 132 BPM |
| 1.2 | 138 BPM |
| 1.5 | 140 BPM |

## A/B comparing against the existing eval

Because `template` and `gemini` are unchanged, you can compare the new backend against the existing baselines without touching any other code.

```bash
# 1. Smoke test the prompt generator only (no GPU needed, ~2 seconds)
python -c "
from src.bridge.prompt_generator import PromptGenerator, EmotionState
pg = PromptGenerator(backend='continuous', alpha=1.2)
for v, a, label in [(0.45, 0.45, 'shallow'), (0.30, 0.30, 'mid'), (0.05, 0.10, 'deep')]:
    print(f'\n--- LVLA {label} (v={v}, a={a}) ---')
    print(pg.generate(EmotionState(valence=v, arousal=a)))
"

# 2. Switch the backend in configs/default.yaml:
#       bridge:
#         backend: "continuous"

# 3. Sanity-check on one DREAMER sample
python -m src.pipeline.run --dreamer-sample 42

# 4. Run the full evaluation (requires GPU for MusicGen)
python -m evaluation.run_full_eval

# 5. Compare evaluation/full_eval_with_prompts_100/eval_report.json
#    against the previous "gemini"-backend report. The cell to watch is
#    the LVLA → uplifting (sad → joyful) row.
```

## PR / commit notes

The diff is two files only — `src/bridge/prompt_generator.py` and `configs/default.yaml`. Existing baselines (`template` and `gemini`) are byte-identical to `origin/main`, so reproducibility of all current results is preserved.

Suggested commit message:

```
Add continuous bridge backend that preserves float V/A magnitudes

The template and gemini backends collapse the (valence, arousal) floats
into a discrete quadrant before generating a prompt, which means a
shallow-LVLA reading and a deep-LVLA reading produce identical music.
The continuous backend reflects the detected point through (0.5, 0.5)
with an aggressiveness coefficient alpha, then maps the resulting
target to BPM, key, mood vocabulary, and instrumentation parametrically.

Targeted at the sad->uplifting failure mode (1.7% on Metric 1).
```

```bash
git add src/bridge/prompt_generator.py configs/default.yaml CONTINUOUS_BRIDGE.md
git commit -m "Add continuous bridge backend that preserves float V/A magnitudes"
git push origin emotion_rep
```
