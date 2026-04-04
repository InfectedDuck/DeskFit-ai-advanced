"""Automated evaluation runner for DeskFit AI RAG system.

Measures retrieval quality metrics (Hit Rate@K, MRR) against a ground-truth test set.
Supports baseline (vector-only) and enhanced (hybrid) retrieval modes.

Usage:
    python -m evaluation.run_evaluation --mode baseline
    python -m evaluation.run_evaluation --mode enhanced
    python -m evaluation.run_evaluation --mode all
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from evaluation.metrics import compute_all_metrics
from src.embeddings import EmbeddingsClient
from src.vector_db import VectorDatabase


def load_test_queries(path: Path | None = None) -> list[dict]:
    """Load the ground-truth test query set."""
    if path is None:
        path = Path(__file__).parent / "test_queries.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def baseline_retrieval_fn(
    embeddings_client: EmbeddingsClient,
    vector_db: VectorDatabase,
):
    """Return a retrieval function that uses vector-only search."""
    def retrieve(query: str, k: int = 5) -> list[str]:
        embedding = embeddings_client.embed_text(query)
        results = vector_db.query(query_embedding=embedding, n_results=k)
        if results and results.get("ids") and results["ids"][0]:
            return results["ids"][0]
        return []
    return retrieve


def enhanced_retrieval_fn():
    """Return a retrieval function that uses hybrid search + re-ranking."""
    from src.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever()
    def retrieve(query: str, k: int = 5) -> list[str]:
        return retriever.retrieve_ids(query, k=k)
    return retrieve


def run_evaluation(
    retrieval_fn,
    test_queries: list[dict],
    k: int = 5,
    label: str = "baseline",
) -> dict:
    """Run evaluation over all test queries and compute metrics."""
    results = []
    print(f"\n{'='*60}")
    print(f"  Running evaluation: {label} (top-{k})")
    print(f"{'='*60}")

    start_time = time.time()

    for i, tq in enumerate(test_queries):
        query = tq["query"]
        expected_ids = tq["expected_ids"]

        retrieved_ids = retrieval_fn(query, k=k)

        hit = any(eid in retrieved_ids[:k] for eid in expected_ids)
        status = "HIT" if hit else "MISS"

        print(f"  [{i+1:2d}/{len(test_queries)}] {status}  {query[:55]:<55}  -> {retrieved_ids[:3]}")

        results.append({
            "query_id": tq["id"],
            "query": query,
            "category": tq.get("category", ""),
            "retrieved_ids": retrieved_ids,
            "expected_ids": expected_ids,
        })

    elapsed = time.time() - start_time
    metrics = compute_all_metrics(results, k=k)

    print(f"\n{'-'*60}")
    print(f"  Results for: {label}")
    print(f"{'-'*60}")
    print(f"  Hit Rate@1:   {metrics['hit_rate_at_1']:.1%}")
    print(f"  Hit Rate@3:   {metrics['hit_rate_at_3']:.1%}")
    print(f"  Hit Rate@5:   {metrics['hit_rate_at_5']:.1%}")
    print(f"  MRR:          {metrics['mrr']:.4f}")
    print(f"  Precision@5:  {metrics['precision_at_5']:.4f}")
    print(f"  Queries:      {metrics['num_queries']}")
    print(f"  Time:         {elapsed:.2f}s")
    print(f"{'-'*60}\n")

    # Build output
    output = {
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "elapsed_seconds": round(elapsed, 2),
        "metrics": {
            "hit_rate_at_1": round(metrics["hit_rate_at_1"], 4),
            "hit_rate_at_3": round(metrics["hit_rate_at_3"], 4),
            "hit_rate_at_5": round(metrics["hit_rate_at_5"], 4),
            "mrr": round(metrics["mrr"], 4),
            "precision_at_5": round(metrics["precision_at_5"], 4),
        },
        "num_queries": metrics["num_queries"],
        "per_query": metrics["per_query"],
    }

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / f"{label}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Run RAG retrieval evaluation")
    parser.add_argument(
        "--mode",
        choices=["baseline", "enhanced", "all"],
        default="all",
        help="Which retrieval mode to evaluate",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-K for retrieval")
    args = parser.parse_args()

    test_queries = load_test_queries()
    print(f"Loaded {len(test_queries)} test queries")

    all_results = {}

    if args.mode in ("baseline", "all"):
        print("\nInitializing baseline components...")
        embeddings_client = EmbeddingsClient(config.EMBEDDING_MODEL_NAME)
        vector_db = VectorDatabase(config.CHROMA_DB_DIR, config.COLLECTION_NAME)
        print(f"  Vector DB documents: {vector_db.count()}")

        retrieve_fn = baseline_retrieval_fn(embeddings_client, vector_db)
        all_results["baseline"] = run_evaluation(
            retrieve_fn, test_queries, k=args.k, label="baseline"
        )

    if args.mode in ("enhanced", "all"):
        print("\nInitializing enhanced (hybrid) components...")
        retrieve_fn = enhanced_retrieval_fn()
        all_results["enhanced"] = run_evaluation(
            retrieve_fn, test_queries, k=args.k, label="enhanced"
        )

    # Print comparison if both modes ran
    if "baseline" in all_results and "enhanced" in all_results:
        b = all_results["baseline"]["metrics"]
        e = all_results["enhanced"]["metrics"]

        print(f"\n{'='*60}")
        print(f"  COMPARISON: Baseline vs Enhanced")
        print(f"{'='*60}")
        print(f"  {'Metric':<15} {'Baseline':>10} {'Enhanced':>10} {'Change':>10}")
        print(f"  {'-'*45}")
        for metric in ["hit_rate_at_1", "hit_rate_at_3", "hit_rate_at_5", "mrr", "precision_at_5"]:
            bv = b[metric]
            ev = e[metric]
            if bv > 0:
                change = (ev - bv) / bv * 100
                change_str = f"{change:+.1f}%"
            else:
                change_str = "N/A"
            print(f"  {metric:<15} {bv:>10.4f} {ev:>10.4f} {change_str:>10}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
