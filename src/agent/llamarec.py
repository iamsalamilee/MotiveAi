"""
LlamaRec Verbalizer Reranker — Stage 2 of the recommendation pipeline.

Scores top-100 candidates from ChromaDB retrieval by extracting
probability distributions from the LLM's final hidden layer.
NO autoregressive decoding — single forward pass per batch.
"""
import numpy as np
from loguru import logger

RERANKER_PROMPT_TEMPLATE = """\
You are a recommendation scoring model.

User preferences:
  Price sensitivity: {price_sensitivity:.2f}
  Quality threshold: {quality_threshold:.2f}
  Characteristic vocabulary: {vocab}

Rate the relevance of this item for this user (0 = irrelevant, 1 = perfect match):
  Item: {item_title}
  Category: {item_category}
  Attributes: {attributes}

Score:"""

class VerbalReranker:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self, model_path: str) -> None:
        """Load LLM for verbalizer scoring. Called once at startup."""
        # TODO: implement real LLM loading with optional quantization
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self._model = AutoModelForCausalLM.from_pretrained(model_path, ...)
        self._loaded = True
        logger.info(f"LlamaRec verbalizer loaded from {model_path}")

    def rerank(self, profile, candidates: list[dict]) -> list[dict]:
        """
        Rerank candidates using verbalizer scoring.
        Returns candidates sorted by score descending.

        Args:
            profile:    UserProfile from APG4RecSim
            candidates: list of dicts from ChromaDB retriever

        Returns:
            sorted list with 'verbalizer_score' field added
        """
        if not candidates:
            return []

        prompts = [self._build_prompt(profile, c) for c in candidates]
        scores = self._score_batch(prompts)
        scores = self._softmax_normalize(scores)

        for c, s in zip(candidates, scores):
            c["verbalizer_score"] = float(s)

        return sorted(candidates, key=lambda x: x["verbalizer_score"], reverse=True)

    def _build_prompt(self, profile, item: dict) -> str:
        """Build ranking prompt for a single candidate item."""
        vocab = ", ".join((profile.domain_vocabulary or [])[:10])
        return RERANKER_PROMPT_TEMPLATE.format(
            price_sensitivity=profile.price_sensitivity,
            quality_threshold=profile.quality_threshold,
            vocab=vocab,
            item_title=item.get("item_title",""),
            item_category=item.get("item_category",""),
            attributes=str(item.get("item_metadata",{}))[:200],
        )

    def _score_batch(self, prompts: list[str]) -> list[float]:
        """
        Single forward pass over all prompts.
        Extract logit at the score token position from final hidden layer.

        TODO Day 3: implement real LLM forward pass.
        Current stub returns similarity-based placeholder scores.
        """
        if not self._loaded:
            # Stub: return uniform scores during development
            return [1.0] * len(prompts)

        # Real implementation:
        # inputs = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        # with torch.no_grad():
        #     outputs = self._model(**inputs, output_hidden_states=True)
        # Extract logit at [SCORE] token position from outputs.logits
        return [1.0] * len(prompts)

    def _softmax_normalize(self, scores: list[float]) -> list[float]:
        """Normalize raw logits to probability distribution."""
        arr = np.array(scores, dtype=np.float64)
        arr -= arr.max()  # numerical stability
        exp = np.exp(arr)
        return (exp / exp.sum()).tolist()

# Module-level singleton
verbalizer = VerbalReranker()
