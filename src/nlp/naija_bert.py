"""
NaijaBERT embedding wrapper.

Uses airesearch/wangchanberta-base-att-spm-uncased as a proxy;
replace model name with actual NaijaBERT checkpoint when available.
Mean-pools the final hidden layer for sentence-level embeddings.
"""
import numpy as np
import torch
from loguru import logger

class NaijaBertEmbedder:
    def __init__(self, model_name: str = "airesearch/wangchanberta-base-att-spm-uncased"):
        self._model_name = model_name
        self._tokenizer = None
        self._model = None
        self._loaded = False

    def warmup(self) -> None:
        """Load model into memory. Call once at startup."""
        from transformers import AutoTokenizer, AutoModel
        logger.info(f"Loading NaijaBERT: {self._model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)
        self._model.eval()
        self._loaded = True
        logger.info("NaijaBERT loaded.")

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Batch embed texts. Returns (N, 768) float32 array.

        Args:
            texts: list of strings to embed

        Returns:
            numpy array of shape (len(texts), hidden_size)
        """
        if not self._loaded:
            self.warmup()

        inputs = self._tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self._model(**inputs)

        return self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single string. Returns (768,) array."""
        return self.embed([text])[0]

    def _mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
        """Mean-pool final hidden layer over non-padding tokens."""
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled.numpy().astype(np.float32)

# Module-level singleton
naija_bert = NaijaBertEmbedder()
