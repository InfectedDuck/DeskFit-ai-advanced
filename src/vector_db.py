"""ChromaDB vector database wrapper for DeskFit AI."""

from pathlib import Path

import chromadb


class VectorDatabase:
    """Wrapper around ChromaDB for storing and querying fitness knowledge base entries."""

    def __init__(self, persist_directory: str | Path, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self) -> chromadb.Collection:
        """Access the underlying ChromaDB collection."""
        return self._collection

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """Add documents with pre-computed embeddings to the collection.

        Handles batching automatically for large datasets.
        """
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self._collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings[i:end],
            )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        """Query the collection by embedding vector with optional metadata filters.

        Returns dict with keys: ids, documents, metadatas, distances.
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def delete_collection(self) -> None:
        """Delete the entire collection and recreate it empty."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_all_ids(self) -> list[str]:
        """Return all document IDs in the collection."""
        result = self._collection.get(include=[])
        return result["ids"]
