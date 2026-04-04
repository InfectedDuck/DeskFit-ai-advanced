"""Tests for the embeddings client."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import EmbeddingsClient


def test_embed_single_text():
    client = EmbeddingsClient()
    vector = client.embed_text("neck pain from sitting at desk")
    assert isinstance(vector, list)
    assert len(vector) == 384
    assert all(isinstance(v, float) for v in vector)


def test_embed_batch():
    client = EmbeddingsClient()
    texts = ["neck stretch", "back pain", "eye fatigue"]
    vectors = client.embed_batch(texts)
    assert len(vectors) == 3
    assert all(len(v) == 384 for v in vectors)


def test_dimension_property():
    client = EmbeddingsClient()
    assert client.dimension == 384


def test_similar_texts_have_close_embeddings():
    client = EmbeddingsClient()
    v1 = client.embed_text("neck pain from computer work")
    v2 = client.embed_text("neck hurts after using laptop")
    v3 = client.embed_text("best pizza recipe")

    # Cosine similarity (vectors are normalized, so dot product = cosine sim)
    sim_related = sum(a * b for a, b in zip(v1, v2))
    sim_unrelated = sum(a * b for a, b in zip(v1, v3))

    assert sim_related > sim_unrelated, "Related texts should have higher similarity"


if __name__ == "__main__":
    test_embed_single_text()
    print("PASS: test_embed_single_text")
    test_embed_batch()
    print("PASS: test_embed_batch")
    test_dimension_property()
    print("PASS: test_dimension_property")
    test_similar_texts_have_close_embeddings()
    print("PASS: test_similar_texts_have_close_embeddings")
    print("\nAll embedding tests passed!")
