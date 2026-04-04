"""Database population script for DeskFit AI.

Loads the JSON dataset, creates embedding vectors, and stores them in ChromaDB.
Run with: python -m src.populate_db [--force]
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.embeddings import EmbeddingsClient
from src.vector_db import VectorDatabase


def load_json(filepath: Path) -> list[dict]:
    """Load and parse a JSON data file."""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def prepare_exercise_text(exercise: dict) -> str:
    """Convert an exercise dict into a searchable text representation."""
    parts = [
        f"Exercise: {exercise['title']}",
        f"Category: {exercise['category']}",
        f"Body area: {exercise['body_area']}",
        f"Difficulty: {exercise['difficulty']}",
        f"Duration: {exercise['duration_seconds']} seconds",
        f"Description: {exercise['description']}",
        f"Steps: {' '.join(exercise['steps'])}",
        f"Benefits: {', '.join(exercise['benefits'])}",
        f"Best for: {', '.join(exercise['best_for'])}",
    ]
    if exercise.get("precautions"):
        parts.append(f"Precautions: {', '.join(exercise['precautions'])}")
    return "\n".join(parts)


def prepare_posture_text(tip: dict) -> str:
    """Convert a posture tip dict into a searchable text representation."""
    parts = [
        f"Posture Tip: {tip['title']}",
        f"Category: {tip['category']}",
        f"Applies to: {tip['applies_to']}",
        f"Description: {tip['description']}",
        f"Quick fix: {tip['quick_fix']}",
        f"Signs of problem: {', '.join(tip['signs_of_problem'])}",
    ]
    return "\n".join(parts)


def prepare_wellness_text(advice: dict) -> str:
    """Convert a wellness advice dict into a searchable text representation."""
    parts = [
        f"Wellness Advice: {advice['title']}",
        f"Category: {advice['category']}",
        f"Context: {advice['context']}",
        f"Description: {advice['description']}",
        f"Why it works: {advice['why_it_works']}",
        f"When to use: {', '.join(advice['when_to_use'])}",
    ]
    return "\n".join(parts)


def build_exercise_metadata(exercise: dict) -> dict:
    """Build ChromaDB-compatible metadata for an exercise."""
    return {
        "type": "exercise",
        "title": exercise["title"],
        "category": exercise["category"],
        "body_area": exercise["body_area"],
        "difficulty": exercise["difficulty"],
        "duration_seconds": exercise["duration_seconds"],
        "can_do_at_desk": str(exercise["can_do_at_desk"]),
        "requires_equipment": str(exercise["requires_equipment"]),
    }


def build_posture_metadata(tip: dict) -> dict:
    """Build ChromaDB-compatible metadata for a posture tip."""
    return {
        "type": "posture_tip",
        "title": tip["title"],
        "category": tip["category"],
        "applies_to": tip["applies_to"],
    }


def build_wellness_metadata(advice: dict) -> dict:
    """Build ChromaDB-compatible metadata for wellness advice."""
    return {
        "type": "wellness_advice",
        "title": advice["title"],
        "category": advice["category"],
        "context": advice["context"],
    }


def populate(force: bool = False) -> None:
    """Main population function."""
    db = VectorDatabase(config.CHROMA_DB_DIR, config.COLLECTION_NAME)

    current_count = db.count()
    if current_count > 0 and not force:
        print(f"Database already contains {current_count} documents.")
        print("Use --force to rebuild from scratch.")
        return

    if force and current_count > 0:
        print(f"Clearing existing {current_count} documents...")
        db.delete_collection()

    # Load all data files
    print("Loading dataset files...")
    exercises = load_json(config.DATA_DIR / "exercises.json")
    posture_tips = load_json(config.DATA_DIR / "posture_tips.json")
    wellness_advice = load_json(config.DATA_DIR / "wellness_advice.json")
    print(f"  Exercises: {len(exercises)}")
    print(f"  Posture tips: {len(posture_tips)}")
    print(f"  Wellness advice: {len(wellness_advice)}")

    # Prepare all documents
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for ex in exercises:
        ids.append(ex["id"])
        documents.append(prepare_exercise_text(ex))
        metadatas.append(build_exercise_metadata(ex))

    for pt in posture_tips:
        ids.append(pt["id"])
        documents.append(prepare_posture_text(pt))
        metadatas.append(build_posture_metadata(pt))

    for wa in wellness_advice:
        ids.append(wa["id"])
        documents.append(prepare_wellness_text(wa))
        metadatas.append(build_wellness_metadata(wa))

    print(f"\nTotal documents to embed: {len(documents)}")

    # Create embeddings
    print("Creating embeddings (this may take a moment on first run)...")
    embeddings_client = EmbeddingsClient(config.EMBEDDING_MODEL_NAME)
    embeddings = embeddings_client.embed_batch(documents)
    print(f"  Embedding dimension: {len(embeddings[0])}")

    # Store in database
    print("Storing in ChromaDB...")
    db.add_documents(ids, documents, metadatas, embeddings)

    final_count = db.count()
    print(f"\nDone! Database now contains {final_count} documents.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate DeskFit AI vector database")
    parser.add_argument("--force", action="store_true", help="Rebuild database from scratch")
    args = parser.parse_args()
    populate(force=args.force)


if __name__ == "__main__":
    main()
