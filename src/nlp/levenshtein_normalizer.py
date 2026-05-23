"""
Phonologically-weighted Levenshtein normalizer for Nigerian Pidgin.

Normalizes chaotic Pidgin spelling variants to canonical forms,
e.g. 'abii', 'abe', 'aby' → 'abi'.
"""
import json
import re
from pathlib import Path
from functools import lru_cache

# Phonological edit costs (lower = cheaper = more likely to be same word)
VOWEL_SUBSTITUTION_COST  = 0.5
CONSONANT_DOUBLE_COST    = 0.3
STANDARD_SUB_COST        = 1.0
MAX_NORMALIZE_COST       = 1.5   # don't normalize if best match > this

VOWELS = set("aeiouAEIOU")

# Minimal bootstrap canonical vocab — extend with pidgin_canonical.json
BOOTSTRAP_VOCAB = {
    "abi", "sha", "sef", "na", "dey", "sabi", "wahala", "ginger",
    "hammer", "yarn", "mumu", "vex", "sharp", "chop", "comot",
    "enter", "follow", "from", "dem", "them", "una", "una",
}

class LevenshteinNormalizer:
    def __init__(self, canonical_path: str | None = None):
        self._vocab: set[str] = set(BOOTSTRAP_VOCAB)
        if canonical_path and Path(canonical_path).exists():
            with open(canonical_path) as f:
                data = json.load(f)
            self._vocab.update(data.get("vocab", []))
        self._vocab_list = sorted(self._vocab)

    def normalize(self, text: str) -> str:
        """Normalize all tokens in text. Returns normalized string."""
        tokens = re.findall(r"\w+|\W+", text)
        return "".join(
            self.normalize_token(t) if re.match(r"\w+", t) else t
            for t in tokens
        )

    def normalize_token(self, token: str) -> str:
        """
        Find closest canonical match using weighted Levenshtein.
        Returns original token if no match within MAX_NORMALIZE_COST.
        """
        lower = token.lower()
        if lower in self._vocab:
            return token  # already canonical

        best_match, best_cost = token, float("inf")
        for candidate in self._vocab_list:
            cost = self._weighted_distance(lower, candidate)
            if cost < best_cost:
                best_cost, best_match = cost, candidate

        if best_cost <= MAX_NORMALIZE_COST:
            # Preserve original casing style
            return best_match if token.islower() else best_match.capitalize()
        return token

    @lru_cache(maxsize=8192)
    def _weighted_distance(self, a: str, b: str) -> float:
        """
        Levenshtein distance with phonological weights for Pidgin.
        Vowel substitutions and consonant doublings are cheaper.
        """
        m, n = len(a), len(b)
        dp = [[float("inf")] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1): dp[i][0] = float(i)
        for j in range(n + 1): dp[0][j] = float(j)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    sub_cost = (
                        VOWEL_SUBSTITUTION_COST
                        if a[i-1] in VOWELS and b[j-1] in VOWELS
                        else STANDARD_SUB_COST
                    )
                    double_cost = (
                        CONSONANT_DOUBLE_COST
                        if a[i-1] == b[j-1] and a[i-1] not in VOWELS
                        else 1.0
                    )
                    dp[i][j] = min(
                        dp[i-1][j] + 1.0,          # deletion
                        dp[i][j-1] + 1.0,          # insertion
                        dp[i-1][j-1] + sub_cost,   # substitution
                        dp[i-1][j-1] + double_cost  # consonant doubling
                    )
        return dp[m][n]

# Module-level singleton
levenshtein = LevenshteinNormalizer()
