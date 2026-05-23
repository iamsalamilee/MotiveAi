"""
SASRec-inspired sequential temporal attention module.

Encodes user review history with position-aware self-attention and
applies exponential recency decay to model preference drift over time.
"""
import math
import numpy as np
from datetime import datetime

LAMBDA_DECAY = 0.001  # decay rate per day; tune if reviews are very sparse

class TemporalAttentionModule:
    def encode_sequence(self, history: list[dict], embeddings: np.ndarray) -> np.ndarray:
        """
        Encode ordered history into a single weighted representation vector.

        Args:
            history:    list of review dicts sorted by timestamp ascending
            embeddings: (N, 768) NaijaBERT embeddings of each review

        Returns:
            (768,) weighted mean embedding
        """
        if len(history) == 0 or embeddings.shape[0] == 0:
            return np.zeros(768, dtype=np.float32)

        history_sorted = sorted(history, key=lambda r: r.get("timestamp",""))
        recency_weights = self._recency_weights(history_sorted)
        attention_weights = self._positional_attention(embeddings)

        combined = recency_weights * attention_weights
        combined /= combined.sum() + 1e-9  # normalize

        return (embeddings * combined[:, None]).sum(axis=0)

    def _recency_weights(self, history: list[dict]) -> np.ndarray:
        """Exponential decay: recent reviews weighted higher."""
        now = datetime.utcnow()
        weights = []
        for r in history:
            try:
                ts = datetime.fromisoformat(r["timestamp"].replace("Z",""))
                delta_days = (now - ts).days
            except Exception:
                delta_days = 365
            weights.append(math.exp(-LAMBDA_DECAY * delta_days))
        arr = np.array(weights, dtype=np.float32)
        return arr

    def _positional_attention(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Simplified self-attention: cosine similarity of each embedding
        to the mean embedding as a proxy for relevance score.
        """
        mean_emb = embeddings.mean(axis=0)
        norm_emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        norm_mean = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)
        scores = (norm_emb @ norm_mean).astype(np.float32)
        scores = np.exp(scores)  # softmax-style
        return scores

# Module-level singleton
temporal_attention = TemporalAttentionModule()
