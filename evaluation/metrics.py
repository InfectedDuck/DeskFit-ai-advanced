"""Retrieval quality metrics for DeskFit AI RAG evaluation."""


def hit_rate_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Return 1.0 if any expected ID appears in the top-k retrieved IDs, else 0.0."""
    top_k = retrieved_ids[:k]
    return 1.0 if any(eid in top_k for eid in expected_ids) else 0.0


def reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """Return 1/rank of the first relevant result, or 0.0 if none found."""
    for i, rid in enumerate(retrieved_ids):
        if rid in expected_ids:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Return the fraction of top-k retrieved IDs that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for rid in top_k if rid in expected_ids)
    return relevant / len(top_k)


def compute_all_metrics(results: list[dict], k: int = 5) -> dict:
    """Compute aggregate metrics over a list of query results.

    Each result dict must have keys: query_id, retrieved_ids, expected_ids.
    Returns aggregate metrics and per-query details.
    """
    per_query = []
    for r in results:
        rid = r["retrieved_ids"]
        eid = r["expected_ids"]
        pq = {
            "query_id": r["query_id"],
            "query": r.get("query", ""),
            "hit_rate_at_1": hit_rate_at_k(rid, eid, 1),
            "hit_rate_at_3": hit_rate_at_k(rid, eid, 3),
            "hit_rate_at_5": hit_rate_at_k(rid, eid, k),
            "mrr": reciprocal_rank(rid, eid),
            "precision_at_5": precision_at_k(rid, eid, k),
            "retrieved_ids": rid,
            "expected_ids": eid,
        }
        per_query.append(pq)

    n = len(per_query)
    return {
        "hit_rate_at_1": sum(pq["hit_rate_at_1"] for pq in per_query) / n if n else 0,
        "hit_rate_at_3": sum(pq["hit_rate_at_3"] for pq in per_query) / n if n else 0,
        "hit_rate_at_5": sum(pq["hit_rate_at_5"] for pq in per_query) / n if n else 0,
        "mrr": sum(pq["mrr"] for pq in per_query) / n if n else 0,
        "precision_at_5": sum(pq["precision_at_5"] for pq in per_query) / n if n else 0,
        "num_queries": n,
        "per_query": per_query,
    }
