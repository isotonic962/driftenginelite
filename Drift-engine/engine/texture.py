import re
import math
from collections import Counter


class TextureAnalyzer:
    """
    Measures prose quality signals rather than violation counts.
    Replaces sentiment/volatility scoring for the 14B model.
    """

    def __init__(self):
        self.simile_patterns = [
            r'\blike a \w+',
            r'\blike \w+ from',
            r'\bas if \w+',
            r'\bas though \w+',
        ]

        self.interiority_verbs = {
            "thought", "felt", "knew", "remembered", "recalled",
            "wondered", "imagined", "wished", "hoped", "feared",
            "believed", "supposed", "considered", "reflected",
            "thinks", "feels", "knows", "remembers", "wonders",
            "imagines", "wishes", "hopes", "fears", "believes",
        }

        from .lexicon import PHYSICAL_VERBS
        self.physical_verbs = PHYSICAL_VERBS

    def _split_sentences(self, text):
        # Lookahead includes curly open quote (dialogue starts), and a
        # min-length filter (>2 words) drops fragments that were skewing
        # sentence-level classification. See drift.py docstring for the
        # 16-chapter recalibration this patch was verified against.
        raw = re.split(r'(?<=[.!?])\s+(?=[A-Z\u201c])', text)
        return [s.strip() for s in raw if s.strip() and len(s.split()) > 2]

    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def figurative_density(self, text):
        """Similes and figurative comparisons per 100 tokens."""
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0
        count = sum(len(re.findall(p, text.lower())) for p in self.simile_patterns)
        return round((count / len(tokens)) * 100, 2)

    def action_interiority_ratio(self, text):
        """
        Ratio of physical action sentences to interiority sentences.
        Returns (action_pct, interiority_pct, neutral_pct).
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return 0.0, 0.0, 0.0

        action = 0
        interiority = 0
        neutral = 0

        for s in sentences:
            tokens = set(self._tokenize(s))
            has_physical = bool(tokens & self.physical_verbs)
            has_interior = bool(tokens & self.interiority_verbs)

            if has_physical and not has_interior:
                action += 1
            elif has_interior:
                interiority += 1
            else:
                neutral += 1

        total = len(sentences)
        return (
            round(action / total * 100, 1),
            round(interiority / total * 100, 1),
            round(neutral / total * 100, 1),
        )

    def dialogue_density(self, text):
        """Percentage of sentences that contain dialogue."""
        sentences = self._split_sentences(text)
        if not sentences:
            return 0.0
        dialogue_sents = sum(1 for s in sentences if any(q in s for q in [chr(34), chr(8220), chr(8221)]))
        return round(dialogue_sents / len(sentences) * 100, 1)

    def sentence_rhythm(self, text):
        """
        Standard deviation of sentence lengths (in words).
        Higher = more rhythmic variation. Zero = flat.
        """
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return 0.0
        lengths = [len(s.split()) for s in sentences]
        mean = sum(lengths) / len(lengths)
        variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
        return round(math.sqrt(variance), 2)

    def prompt_echo(self, prompt, output):
        """
        Token overlap between prompt and output.
        High overlap = model is echoing rather than generating.
        """
        prompt_tokens = set(self._tokenize(prompt))
        output_tokens = self._tokenize(output)
        if not output_tokens:
            return 0.0
        overlap = sum(1 for t in output_tokens if t in prompt_tokens)
        return round(overlap / len(output_tokens) * 100, 1)

    def analyze(self, text, prompt=""):
        """Full texture analysis."""
        action_pct, interior_pct, neutral_pct = self.action_interiority_ratio(text)
        return {
            "figurative_density": self.figurative_density(text),
            "action_pct": action_pct,
            "interiority_pct": interior_pct,
            "neutral_pct": neutral_pct,
            "dialogue_density": self.dialogue_density(text),
            "sentence_rhythm": self.sentence_rhythm(text),
            "prompt_echo": self.prompt_echo(prompt, text),
            # length metadata for turn-scale corrections in DriftScorer.
            # Additive keys only -- downstream .get() consumers unaffected.
            "n_tokens": len(self._tokenize(text)),
            "n_sentences": len(self._split_sentences(text)),
        }
