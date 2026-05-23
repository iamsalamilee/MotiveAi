"""
ChromaDB vector retriever — Stage 1 of the Two-Stage recommendation pipeline.

ANN search over NaijaBERT-embedded item catalogue.
Returns top-100 candidates for LlamaRec verbalizer reranking.
"""
import numpy as np
from loguru import logger

class VectorRetriever:
    def __init__(self):
        self._client = None
        self._collection = None
        self._connected = False

    def connect(self, host: str, port: int, collection_name: str = "items") -> None:
        """Connect to ChromaDB and get/create the items collection."""
        import chromadb
        self._client = chromadb.HttpClient(host=host, port=port)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._connected = True
        logger.info(f"ChromaDB connected. Items indexed: {self.get_collection_size()}")

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 100) -> list[dict]:
        """
        ANN search for top_k most similar items.

        Args:
            query_embedding: (768,) float32 array from NaijaBERT
            top_k: number of candidates to retrieve (default 100 for Stage 1)

        Returns:
            list of dicts with keys: item_id, item_title, metadata, similarity_score
        """
        if not self._connected:
            raise RuntimeError("Retriever not connected. Call connect() first.")

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.get_collection_size()),
            include=["metadatas", "distances"],
        )

        items = []
        for i, item_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i]
            items.append({
                "item_id": item_id,
                "item_title": meta.get("title", ""),
                "item_category": meta.get("category", ""),
                "item_metadata": meta,
                "similarity_score": 1.0 - dist,  # cosine distance → similarity
            })
        return items

    def upsert_items(self, items: list[dict]) -> None:
        """
        Insert or update items in ChromaDB.

        Args:
            items: list of dicts with keys: item_id, embedding, metadata
        """
        if not self._connected:
            raise RuntimeError("Retriever not connected.")

        self._collection.upsert(
            ids=[it["item_id"] for it in items],
            embeddings=[it["embedding"].tolist() for it in items],
            metadatas=[it["metadata"] for it in items],
        )

    def get_collection_size(self) -> int:
        """Return number of items currently indexed."""
        if not self._connected:
            return 0
        return self._collection.count()

    def is_ready(self) -> bool:
        return self._connected and self.get_collection_size() > 0

# Module-level singleton
retriever = VectorRetriever()
