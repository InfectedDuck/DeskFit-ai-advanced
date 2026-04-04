"""Cross-encoder re-ranker for DeskFit AI RAG pipeline."""

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Re-ranks retrieval candidates using a cross-encoder model.

    Cross-encoders process query-document pairs jointly, producing much more
    accurate relevance scores than bi-encoder cosine similarity.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model: CrossEncoder | None = None

    def _load_model(self) -> None:
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            self._model = CrossEncoder(self._model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Score each (query, document) pair and return top-k sorted by relevance.

        Args:
            query: The user query.
            candidates: List of dicts with at least 'id' and 'document' keys.
            top_k: Number of top results to return.

        Returns:
            Top-k candidates sorted by cross-encoder relevance score (descending).
        """
        if not candidates:
            return []

        self._load_model()

        # Build pairs for scoring
        pairs = [(query, c["document"]) for c in candidates]
        scores = self._model.predict(pairs)

        # Attach scores and sort
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        sorted_candidates = sorted(
            candidates, key=lambda c: c["rerank_score"], reverse=True
        )

        return sorted_candidates[:top_k]
