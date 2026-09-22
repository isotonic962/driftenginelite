import math


class DriftScorer:
    """
    Scores drift as deviation from the Moberg texture profile.

    Corridors derived from p10-p90 across 88 full chapters spanning three
    books of the Emigrants series: The Emigrants (26), Unto a Good Land (26),
    and The Settlers (36). Sista brevet till Sverige is deliberately excluded
    -- treated as a separate work, not part of the same arc. Measured with the
    original lexicon.py PHYSICAL_VERBS set and the patched _split_sentences()
    (curly-quote lookahead + min-length filter -- see texture.py).

    These supersede an earlier 16-chapter single-book calibration, which was
    systematically too narrow: measured against the full three-book sample it
    rejected roughly a third of Moberg's own chapters (action_pct 34/88
    outside, neutral_pct 34/88, sentence_rhythm 38/88, interiority_pct 31/88,
    figurative ceiling exceeded by 40/88). The interiority floor was the worst
    offender at 7.0 when the real p10 across the series is 3.2 -- the engine
    was penalising sparser interiority than book one, which is exactly what
    Moberg himself writes later on.

    Interiority declines measurably across the series: mean 8.90 (Emigrants),
    7.98 (Unto a Good Land), 6.75 (The Settlers); Welch t = 3.04 comparing
    first to third. Dialogue density (11.2-11.7) and figurative density
    (0.14-0.15) stay flat throughout. So "Moberg's register" is not one fixed
    target across the arc -- a future position-aware corridor pass could use
    this, since current_chapter() already exists in scene_counter.py.

    Still not validated against actual Qwen3-14B model output -- only against
    real Moberg prose.

    All texture metrics are corridor-scored: penalize only when a
    chapter falls outside Moberg's own observed range for that metric.

    Figurative density remains the one true one-sided metric: Moberg's
    own p90 across 16 chapters is 0.15, and there's no scene type where
    heavy figurative language is expected, so anything meaningfully
    above that ceiling is a real anchor violation, not scene variance.

    Prior corridors (10-chapter sample, pre-lexicon-fix) are superseded.
    The old neutral_pct corridor in particular was measuring a lexicon bug:
    an earlier benchmark pass had misclassified speech/perception verbs
    (said, told, asked, looked...) as physical action, which collapsed
    neutral_pct and inflated action_pct. Fixed by keeping lexicon.py's
    original ~80-verb PHYSICAL_VERBS set untouched.

    Entropy floor set from the 16-chapter p10 (8.05); both old and new
    sentence-splitter classifiers agree on this within noise.
    """

    CORRIDORS = {
        "action_pct":       (15.0, 29.6),
        "dialogue_density": (0.0,  21.7),
        "neutral_pct":      (63.0, 78.2),
        "sentence_rhythm":  (8.8,  12.7),
        "interiority_pct":  (3.2,  11.5),
    }

    ONE_SIDED = {
        "figurative_density": 0.2,
    }

    WEIGHTS = {
        "interiority_pct":    1.0,
        "figurative_density": 1.5,
        "action_pct":         0.5,
        "dialogue_density":   0.5,
        "neutral_pct":        0.2,
        "sentence_rhythm":    0.2,
    }

    ENTROPY_FLOOR = 7.8

    # --- turn-scale corrections (additive; inert when texture dict carries
    # no length metadata, so chapter-scale behavior is bit-identical) -------
    #
    # 1. Entropy is bounded above by log2(n_tokens), so the 8.05 chapter
    #    floor is mathematically unclearable below ~265 tokens and penalizes
    #    short turns for being short. When n_tokens is known, the floor
    #    becomes min(8.05, log2(n_tokens) - ENTROPY_LEN_MARGIN).
    #    Margin provenance: on 50 curated short passages (real Haruf/
    #    Lindgren/Niemi + annotation-corpus synthetics), healthy prose sits
    #    0.06-1.33 bits below its cap (p90 = 1.10). 1.6 passes all of them
    #    while degenerate looped output falls much further below cap.
    #    PROVISIONAL until re-fit on real Qwen3 output -- kaggle/calibrate.py
    #    section 5 emits the fitted margin for each run.
    ENTROPY_LEN_MARGIN = 1.6

    # 2. Below 9 sentences the interiority corridor (7-12%) is unsatisfiable
    #    on granularity grounds alone: 0/8 = 0%, 1/8 = 12.5% -- both outside.
    #    When n_sentences is known and < 9, counts of 0 or 1 interior
    #    sentences score zero penalty; 2+ keep the normal ceiling penalty.
    GRANULARITY_MIN_SENTENCES = 9

    def __init__(self):
        pass

    def _one_sided_penalty(self, actual, target):
        """Penalize only when actual exceeds target. Below target = 0."""
        return max(0.0, actual - target)

    def _corridor_penalty(self, actual, low, high):
        """Penalize only when actual falls outside [low, high]. Inside = 0."""
        if actual < low:
            return low - actual
        elif actual > high:
            return actual - high
        return 0.0

    def score(self, texture, entropy=0.0):
        """
        Args:
            texture: dict from TextureAnalyzer.analyze()
            entropy: float from EntropyCalculator.analyze()["entropy"]

        Returns dict with per-component deviations and final drift_score.
        """
        components = {}
        drift = 0.0

        for key, target in self.ONE_SIDED.items():
            actual = texture.get(key, 0.0)
            penalty = self._one_sided_penalty(actual, target)
            weight = self.WEIGHTS[key]
            components[f"{key}_dev"] = round(penalty, 3)
            drift += penalty * weight

        n_sentences = texture.get("n_sentences")
        n_tokens = texture.get("n_tokens")

        for key, (low, high) in self.CORRIDORS.items():
            actual = texture.get(key, 0.0)
            if (key == "interiority_pct" and n_sentences
                    and n_sentences < self.GRANULARITY_MIN_SENTENCES):
                # short-turn dead zone: 0 or 1 interior sentences is fine;
                # 2+ keeps the normal ceiling penalty
                interior_count = round(actual * n_sentences / 100.0)
                penalty = max(0.0, actual - high) if interior_count >= 2 else 0.0
            else:
                penalty = self._corridor_penalty(actual, low, high)
            weight = self.WEIGHTS[key]
            components[f"{key}_dev"] = round(penalty, 3)
            drift += penalty * weight

        effective_floor = self.ENTROPY_FLOOR
        if n_tokens:
            effective_floor = min(
                self.ENTROPY_FLOOR,
                math.log2(max(2, n_tokens)) - self.ENTROPY_LEN_MARGIN,
            )
        components["entropy_floor_effective"] = round(effective_floor, 3)
        entropy_penalty = max(0.0, effective_floor - entropy)
        entropy_component = round(entropy_penalty * 0.3, 4)
        components["entropy_component"] = entropy_component
        drift += entropy_penalty * 0.3

        components["drift_score"] = round(drift, 4)
        return components