"""
APG4RecSim — Task-Aware Automated Profile Generation for Recommendation Simulation.

Extracts structured UserProfile from raw review history.
Profile traits are injected into the LLM generation prompt to increase
n-gram overlap (ROUGE) and semantic similarity (BERTScore) with ground truth.
"""
from dataclasses import dataclass, field
from statistics import mean, stdev

@dataclass
class UserProfile:
    user_id: str
    price_sensitivity: float        # 0.0 (insensitive) → 1.0 (very sensitive)
    quality_threshold: float        # min acceptable quality floor
    rating_skew: float              # mean historical star rating
    sentiment_volatility: float     # std dev of compound sentiment scores
    verbosity_mean: float           # mean review word count
    verbosity_std: float            # std dev of review word count
    domain_vocabulary: list[str] = field(default_factory=list)  # top-40 n-grams
    behavioral_policy: dict = field(default_factory=dict)
    temporal_weight: float = 1.0   # set by SASRec module

class ProfileGenerator:
    def __init__(self):
        from src.nlp.pidgin_vader import pidgin_vader
        from src.nlp.levenshtein_normalizer import levenshtein
        self._vader = pidgin_vader
        self._normalizer = levenshtein

    def extract_profile(self, user_id: str, history: list[dict]) -> UserProfile:
        """
        Extract structured UserProfile from user review history.

        Args:
            user_id: user identifier
            history: list of review dicts with keys: item_id, rating, review_text, timestamp

        Returns:
            UserProfile dataclass
        """
        if not history:
            return self._default_profile(user_id)

        texts = [self._normalizer.normalize(r["review_text"]) for r in history]
        ratings = [r["rating"] for r in history]
        sentiments = [self._vader.score_sentiment(t)["compound"] for t in texts]
        word_counts = [len(t.split()) for t in texts]

        return UserProfile(
            user_id=user_id,
            price_sensitivity=self._derive_price_sensitivity(history, texts),
            quality_threshold=self._derive_quality_threshold(ratings),
            rating_skew=mean(ratings),
            sentiment_volatility=stdev(sentiments) if len(sentiments) > 1 else 0.0,
            verbosity_mean=mean(word_counts),
            verbosity_std=stdev(word_counts) if len(word_counts) > 1 else 0.0,
            domain_vocabulary=self._derive_vocab(texts),
            behavioral_policy={
                "rating_distribution": self._rating_dist(ratings),
                "avg_review_length": mean(word_counts),
            }
        )

    def to_prompt_string(self, profile: UserProfile) -> str:
        """Serialize profile to LLM system prompt injection string."""
        vocab_sample = ", ".join(profile.domain_vocabulary[:15])
        return (
            f"User traits: price_sensitivity={profile.price_sensitivity:.2f}, "
            f"quality_threshold={profile.quality_threshold:.2f}, "
            f"avg_rating={profile.rating_skew:.1f}/5, "
            f"typical_review_length={int(profile.verbosity_mean)} words, "
            f"characteristic_vocabulary=[{vocab_sample}]"
        )

    def _derive_price_sensitivity(self, history: list[dict], texts: list[str]) -> float:
        """Heuristic: if user mentions price/cost/cheap/expensive frequently, higher sensitivity."""
        price_keywords = {"price", "cost", "cheap", "expensive", "worth", "value", "money", "dear"}
        mentions = sum(
            1 for t in texts
            if any(w in t.lower() for w in price_keywords)
        )
        return min(1.0, mentions / max(len(texts), 1))

    def _derive_quality_threshold(self, ratings: list[int]) -> float:
        """Normalized quality floor: mean / 5."""
        return mean(ratings) / 5.0

    def _derive_vocab(self, texts: list[str]) -> list[str]:
        """
        Extract top-40 characteristic tokens via simple TF scoring.
        TODO Day 3: replace with TF-IDF using full corpus as IDF base.
        """
        from collections import Counter
        import re
        stop = {"the","a","an","is","it","in","of","to","and","was","i","my","this","for","on"}
        tokens = []
        for t in texts:
            tokens.extend(w.lower() for w in re.findall(r"\b\w{3,}\b", t) if w.lower() not in stop)
        return [w for w, _ in Counter(tokens).most_common(40)]

    def _rating_dist(self, ratings: list[int]) -> dict:
        from collections import Counter
        total = len(ratings)
        return {str(k): round(v/total, 2) for k, v in Counter(ratings).items()}

    def _default_profile(self, user_id: str) -> UserProfile:
        return UserProfile(
            user_id=user_id,
            price_sensitivity=0.5, quality_threshold=0.7,
            rating_skew=4.0, sentiment_volatility=0.3,
            verbosity_mean=80.0, verbosity_std=30.0,
        )
