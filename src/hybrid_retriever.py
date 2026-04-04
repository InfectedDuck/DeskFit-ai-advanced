"""Hybrid retriever combining vector search, BM25, and cross-encoder re-ranking."""

import config
from src.bm25_search import BM25Search
from src.embeddings import EmbeddingsClient
from src.reranker import CrossEncoderReranker
from src.vector_db import VectorDatabase


class HybridRetriever:
    """Combines vector search + BM25 keyword search + cross-encoder re-ranking.

    Pipeline:
    1. Over-fetch candidates from ChromaDB vector search (top vector_k)
    2. Over-fetch candidates from BM25 keyword search (top bm25_k)
    3. Merge and deduplicate candidates
    4. Re-rank using cross-encoder
    5. Return final top-k results
    """

    def __init__(
        self,
        embeddings_client: EmbeddingsClient | None = None,
        vector_db: VectorDatabase | None = None,
        bm25_search: BM25Search | None = None,
        reranker: CrossEncoderReranker | None = None,
        vector_k: int = 15,
        bm25_k: int = 15,
        final_k: int = 5,
    ):
        self._embeddings = embeddings_client or EmbeddingsClient(config.EMBEDDING_MODEL_NAME)
        self._vector_db = vector_db or VectorDatabase(config.CHROMA_DB_DIR, config.COLLECTION_NAME)
        self._bm25 = bm25_search or BM25Search()
        self._reranker = reranker or CrossEncoderReranker()
        self._vector_k = vector_k
        self._bm25_k = bm25_k
        self._final_k = final_k

    def retrieve(self, query: str, k: int | None = None, filters: dict | None = None) -> list[dict]:
        """Execute hybrid retrieval pipeline.

        Returns list of dicts with keys: id, document, metadata, distance, rerank_score.
        """
        final_k = k or self._final_k

        # Step 1: Vector search
        query_embedding = self._embeddings.embed_text(query)
        vector_results = self._vector_db.query(
            query_embedding=query_embedding,
            n_results=self._vector_k,
            where=filters,
        )

        # Step 2: BM25 search
        bm25_results = self._bm25.search(query, top_k=self._bm25_k)

        # Step 3: Merge and deduplicate
        candidates = {}

        # Add vector results
        if vector_results and vector_results.get("ids") and vector_results["ids"][0]:
            for doc_id, document, metadata, distance in zip(
                vector_results["ids"][0],
                vector_results["documents"][0],
                vector_results["metadatas"][0],
                vector_results["distances"][0],
            ):
                candidates[doc_id] = {
                    "id": doc_id,
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                    "source": "vector",
                }

        # Add BM25 results (merge, don't overwrite)
        for doc_id, bm25_score in bm25_results:
            if doc_id in candidates:
                candidates[doc_id]["bm25_score"] = bm25_score
                candidates[doc_id]["source"] = "both"
            else:
                doc_text = self._bm25.get_document_text(doc_id)
                if doc_text:
                    candidates[doc_id] = {
                        "id": doc_id,
                        "document": doc_text,
                        "metadata": {},
                        "distance": None,
                        "bm25_score": bm25_score,
                        "source": "bm25",
                    }

        candidate_list = list(candidates.values())

        if not candidate_list:
            return []

        # Step 4: Re-rank with cross-encoder
        reranked = self._reranker.rerank(query, candidate_list, top_k=final_k)

        return reranked

    def retrieve_ids(self, query: str, k: int = 5) -> list[str]:
        """Convenience method returning only document IDs."""
        results = self.retrieve(query, k=k)
        return [r["id"] for r in results]
