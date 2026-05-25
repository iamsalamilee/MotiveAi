"""
Nigerian Pidgin-adapted VADER sentiment analyzer.

Corrects systematic semantic shifts in Nigerian Pidgin before scoring,
e.g. 'ginger' (motivation/enthusiasm) vs. the root plant interpretation.
"""
import json
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Minimum required overrides — extend with full lexicon file
DEFAULT_OVERRIDES = {
    "ginger":  0.7,   # motivation/energy, not root plant
    "hammer":  0.8,   # succeeded/rich, not a tool
    "wash":   -0.6,   # cheated/disappointed, not clean
    "sabi":    0.5,   # knowledgeable/skilled
    "vex":    -0.7,   # angry/irritated
    "sharp":   0.6,   # clever/smart
    "mumu":   -0.5,   # foolish/gullible
    "yarn":   -0.1,   # chatting (near-neutral)
    "wire":    0.3,   # to bribe/tip (context-dependent, mild positive)
}

class PidginVader:
    def __init__(self, lexicon_path: str | None = None):
        self._analyzer = SentimentIntensityAnalyzer()
        self._overrides: dict[str, float] = dict(DEFAULT_OVERRIDES)
        if lexicon_path and Path(lexicon_path).exists():
            self._load_lexicon(lexicon_path)
        # Inject overrides into VADER's internal lexicon
        for token, score in self._overrides.items():
            self._analyzer.lexicon[token] = score

    def _load_lexicon(self, path: str) -> None:
        """Load additional overrides from JSON file."""
        with open(path) as f:
            extra = json.load(f)
        self._overrides.update(extra)

    def score_sentiment(self, text: str) -> dict:
        """
        Score sentiment of text using Pidgin-corrected VADER.

        Returns:
            dict with keys: neg, neu, pos, compound (same as VADER)
        """
        return self._analyzer.polarity_scores(text)

    def batch_score(self, texts: list[str]) -> list[dict]:
        """Score a list of texts."""
        return [self.score_sentiment(t) for t in texts]

# Module-level singleton
pidgin_vader = PidginVader()
