"""RAG pipeline orchestrator for DeskFit AI."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.embeddings import EmbeddingsClient
from src.llm_client import LLMClient
from src.prompts import build_rag_prompt
from src.vector_db import VectorDatabase

if TYPE_CHECKING:
    from src.hybrid_retriever import HybridRetriever


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""

    answer: str
    sources: list[dict] = field(default_factory=list)


class RAGPipeline:
    """Orchestrates the full RAG flow: embed query -> search -> augment -> generate."""

    def __init__(
        self,
        embeddings_client: EmbeddingsClient,
        vector_db: VectorDatabase,
        llm_client: LLMClient,
        hybrid_retriever: HybridRetriever | None = None,
    ) -> None:
        self._embeddings = embeddings_client
        self._db = vector_db
        self._llm = llm_client
        self._hybrid = hybrid_retriever

    def _retrieve(
        self,
        user_question: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """Retrieve context chunks using hybrid or baseline retrieval."""
        if self._hybrid is not None:
            return self._hybrid.retrieve(user_question, k=top_k, filters=filters)

        query_embedding = self._embeddings.embed_text(user_question)
        results = self._db.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=filters,
        )
        return self._format_results(results)

    def query(
        self,
        user_question: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> RAGResponse:
        """Execute the full RAG pipeline and return a complete response."""
        context_chunks = self._retrieve(user_question, top_k, filters)
        messages = build_rag_prompt(user_question, context_chunks)
        answer = self._llm.chat(messages)
        return RAGResponse(answer=answer, sources=context_chunks)

    def query_stream(
        self,
        user_question: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> tuple[Generator[str, None, None], list[dict]]:
        """Streaming version of the RAG pipeline.

        Returns a tuple of (stream_generator, source_chunks).
        The sources are available immediately since retrieval happens before LLM call.
        """
        context_chunks = self._retrieve(user_question, top_k, filters)
        messages = build_rag_prompt(user_question, context_chunks)
        stream = self._llm.chat_stream(messages)
        return stream, context_chunks

    @staticmethod
    def _format_results(results: dict) -> list[dict]:
        """Convert ChromaDB query results into a list of context chunk dicts."""
        chunks = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return chunks

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            chunks.append({
                "id": doc_id,
                "document": document,
                "metadata": metadata,
                "distance": distance,
            })
        return chunks
