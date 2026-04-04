"""Embeddings client using sentence-transformers for DeskFit AI."""

from sentence_transformers import SentenceTransformer


class EmbeddingsClient:
    """Creates vector embeddings using a local sentence-transformers model.

    Uses all-MiniLM-L6-v2 by default, producing 384-dimensional vectors.
    The model is loaded lazily on first use (~80MB download on first run).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model: SentenceTransformer | None = None
        self._model_name = model_name

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformers model on first access."""
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a vector."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed multiple texts efficiently in one batch call."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self.model.get_sentence_embedding_dimension()
