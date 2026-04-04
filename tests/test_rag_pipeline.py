"""Tests for the RAG pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts import build_rag_prompt, format_context


def test_format_context_empty():
    result = format_context([])
    assert "No relevant context" in result


def test_format_context_with_chunks():
    chunks = [
        {
            "document": "Exercise: Neck Rolls\nSteps: Roll your neck gently.",
            "metadata": {"type": "exercise", "title": "Neck Rolls"},
            "distance": 0.2,
        }
    ]
    result = format_context(chunks)
    assert "Neck Rolls" in result
    assert "80%" in result  # 1 - 0.2 = 0.8 = 80%


def test_build_rag_prompt_structure():
    chunks = [
        {
            "document": "Test document content",
            "metadata": {"type": "exercise", "title": "Test"},
            "distance": 0.1,
        }
    ]
    messages = build_rag_prompt("my neck hurts", chunks)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "DeskFit AI" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "my neck hurts" in messages[1]["content"]
    assert "Test document content" in messages[1]["content"]


def test_build_rag_prompt_no_context():
    messages = build_rag_prompt("random question", [])
    assert "No relevant context" in messages[1]["content"]


if __name__ == "__main__":
    test_format_context_empty()
    print("PASS: test_format_context_empty")
    test_format_context_with_chunks()
    print("PASS: test_format_context_with_chunks")
    test_build_rag_prompt_structure()
    print("PASS: test_build_rag_prompt_structure")
    test_build_rag_prompt_no_context()
    print("PASS: test_build_rag_prompt_no_context")
    print("\nAll RAG pipeline tests passed!")
