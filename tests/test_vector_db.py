"""Tests for the ChromaDB vector database wrapper."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_db import VectorDatabase

# Use a fixed test directory to avoid Windows temp cleanup issues
TEST_DB_DIR = Path(__file__).parent / "_test_chroma_db"


def get_db(collection_name: str = "test") -> VectorDatabase:
    """Create a test database, cleaning up any existing test collection."""
    db = VectorDatabase(str(TEST_DB_DIR), collection_name)
    db.delete_collection()
    return db


def test_add_and_count():
    db = get_db("test_add")
    assert db.count() == 0

    db.add_documents(
        ids=["doc_1", "doc_2"],
        documents=["neck stretch exercise", "back pain relief"],
        metadatas=[{"type": "exercise"}, {"type": "exercise"}],
        embeddings=[[0.1] * 384, [0.2] * 384],
    )
    assert db.count() == 2


def test_query():
    db = get_db("test_query")
    db.add_documents(
        ids=["doc_1", "doc_2", "doc_3"],
        documents=["neck stretch", "shoulder shrug", "eye palming"],
        metadatas=[
            {"type": "exercise", "body_area": "neck"},
            {"type": "exercise", "body_area": "shoulders"},
            {"type": "exercise", "body_area": "eyes"},
        ],
        embeddings=[
            [1.0] + [0.0] * 383,
            [0.0, 1.0] + [0.0] * 382,
            [0.0, 0.0, 1.0] + [0.0] * 381,
        ],
    )

    results = db.query(query_embedding=[1.0] + [0.0] * 383, n_results=2)
    assert len(results["ids"][0]) == 2
    assert "doc_1" in results["ids"][0]


def test_query_with_filter():
    db = get_db("test_filter")
    db.add_documents(
        ids=["doc_1", "doc_2"],
        documents=["neck stretch", "eye exercise"],
        metadatas=[
            {"type": "exercise", "body_area": "neck"},
            {"type": "exercise", "body_area": "eyes"},
        ],
        embeddings=[[0.5] * 384, [0.5] * 384],
    )

    results = db.query(
        query_embedding=[0.5] * 384,
        n_results=5,
        where={"body_area": "eyes"},
    )
    assert len(results["ids"][0]) == 1
    assert results["ids"][0][0] == "doc_2"


def test_delete_collection():
    db = get_db("test_delete")
    db.add_documents(
        ids=["doc_1"],
        documents=["test"],
        metadatas=[{"type": "test"}],
        embeddings=[[0.1] * 384],
    )
    assert db.count() == 1

    db.delete_collection()
    assert db.count() == 0


def test_get_all_ids():
    db = get_db("test_ids")
    db.add_documents(
        ids=["a", "b", "c"],
        documents=["doc a", "doc b", "doc c"],
        metadatas=[{"type": "t"}] * 3,
        embeddings=[[0.1] * 384] * 3,
    )
    ids = db.get_all_ids()
    assert set(ids) == {"a", "b", "c"}


if __name__ == "__main__":
    test_add_and_count()
    print("PASS: test_add_and_count")
    test_query()
    print("PASS: test_query")
    test_query_with_filter()
    print("PASS: test_query_with_filter")
    test_delete_collection()
    print("PASS: test_delete_collection")
    test_get_all_ids()
    print("PASS: test_get_all_ids")
    print("\nAll vector DB tests passed!")
