"""BM25 keyword search for DeskFit AI knowledge base."""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

import config
from src.populate_db import (
    prepare_exercise_text,
    prepare_posture_text,
    prepare_wellness_text,
)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercasing tokenizer."""
    return re.findall(r"\w+", text.lower())


class BM25Search:
    """BM25 keyword search over the DeskFit knowledge base."""

    def __init__(self, data_dir: Path | None = None):
        if data_dir is None:
            data_dir = config.DATA_DIR

        self._doc_ids: list[str] = []
        self._doc_texts: list[str] = []
        self._build_index(data_dir)

    def _build_index(self, data_dir: Path) -> None:
        """Load JSON data and build the BM25 index."""
        # Load exercises
        with open(data_dir / "exercises.json", encoding="utf-8") as f:
            exercises = json.load(f)
        for ex in exercises:
            self._doc_ids.append(ex["id"])
            self._doc_texts.append(prepare_exercise_text(ex))

        # Load posture tips
        with open(data_dir / "posture_tips.json", encoding="utf-8") as f:
            posture_tips = json.load(f)
        for pt in posture_tips:
            self._doc_ids.append(pt["id"])
            self._doc_texts.append(prepare_posture_text(pt))

        # Load wellness advice
        with open(data_dir / "wellness_advice.json", encoding="utf-8") as f:
            wellness_advice = json.load(f)
        for wa in wellness_advice:
            self._doc_ids.append(wa["id"])
            self._doc_texts.append(prepare_wellness_text(wa))

        # Build BM25 index
        tokenized_docs = [_tokenize(text) for text in self._doc_texts]
        self._bm25 = BM25Okapi(tokenized_docs)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search for documents matching the query.

        Returns list of (doc_id, bm25_score) sorted by score descending.
        """
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score
        scored_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [(self._doc_ids[i], float(scores[i])) for i in scored_indices]

    def get_document_text(self, doc_id: str) -> str | None:
        """Get the text content for a document ID."""
        try:
            idx = self._doc_ids.index(doc_id)
            return self._doc_texts[idx]
        except ValueError:
            return None

    @property
    def doc_ids(self) -> list[str]:
        return list(self._doc_ids)

    @property
    def doc_texts(self) -> dict[str, str]:
        return dict(zip(self._doc_ids, self._doc_texts))
