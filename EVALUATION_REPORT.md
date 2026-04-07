# DeskFit AI RAG Evaluation Report

**Generated:** 2026-04-06 13:47 UTC

## 1. Metrics Definition and Selection

### Why These Metrics?

In a RAG system, retrieval quality is the foundation. If the retriever fails to find
relevant documents, the LLM has no grounded context and will either hallucinate or
give generic advice. For a wellness assistant like DeskFit AI, this could mean
recommending inappropriate exercises or missing safety precautions.

We selected two complementary retrieval metrics:

### Primary Metric: Hit Rate@K

**Definition:** The proportion of queries where at least one relevant document
appears in the top-K retrieved results.

```
Hit Rate@K = (# queries with at least 1 relevant doc in top-K) / (total queries)
```

**Why it matters:** This is a binary measure of retrieval success. A hit rate of 90%
means 10% of user queries get zero relevant context -- those users receive purely
hallucinated or generic responses. For a health/wellness application, this directly
impacts user trust and safety.

We measure at K=1, K=3, and K=5:
- **Hit Rate@1** captures whether the *single best* result is relevant (critical for
  ranking quality)
- **Hit Rate@5** captures overall retrieval coverage (matches the system's default top-K)

### Secondary Metric: Mean Reciprocal Rank (MRR)

**Definition:** The average of 1/rank of the first relevant result across all queries.

```
MRR = (1/N) * SUM(1 / rank_of_first_relevant_result)
```

**Why it matters:** MRR captures *ranking quality*. Two systems might both have 90%
hit rate, but if System A places the relevant doc at rank 1 while System B places it
at rank 5, System A provides better context to the LLM. Higher MRR = the LLM sees
the most relevant information first in its context window.

### Additional Metric: Precision@5

**Definition:** The fraction of top-5 results that are relevant.

```
Precision@5 = (# relevant docs in top-5) / 5
```

This measures how "clean" the context is -- higher precision means less noise
for the LLM to filter through.

## 2. Evaluation Setup

### Test Set

- **30 test queries** covering 5 categories:
  - `direct_match`: queries using domain terms (e.g., "carpal tunnel prevention")
  - `symptom_based`: describing symptoms (e.g., "my neck is stiff")
  - `scenario`: situational queries (e.g., "quick stretch between meetings")
  - `cross_category`: spanning multiple content types (e.g., "stress and anxiety")
  - `vague`: imprecise queries (e.g., "I feel stiff")

- Each query has 1-6 **expected relevant document IDs**, manually curated against
  the 100-document knowledge base

### Baseline System

- **Embedding model:** `all-MiniLM-L6-v2` (384-dim, sentence-transformers)
- **Vector database:** ChromaDB with cosine similarity, HNSW index
- **Retrieval:** Pure vector search, top-5
- **No re-ranking, no keyword search**

## 3. Baseline Results

**Timestamp:** 2026-04-06T13:46:38.951276+00:00
**Retrieval time:** 9.88s for 30 queries

| Metric | Value |
|--------|-------|
| Hit Rate@1 | 56.7% |
| Hit Rate@3 | 83.3% |
| Hit Rate@5 | 90.0% |
| MRR | 0.7056 |
| Precision@5 | 0.3733 |

### Baseline Analysis

**Strengths:**
- Hit Rate@5 at 90% shows the vector embeddings capture semantic meaning well
- Direct-match queries (domain terms) perform best

**Weaknesses:**
- Hit Rate@1 at only 56.7% means the *most relevant* document often isn't ranked first
- MRR of 0.71 confirms ranking quality needs improvement
- Symptom-based and vague queries show the most misses
- Bi-encoder cosine similarity is a rough ranking signal -- it embeds query and
  document *independently*, missing fine-grained relevance cues

**Missed Queries (no relevant doc in top-5):**

- `My neck is stiff from looking at my monitor all day` -- retrieved: ['pt_001', 'pt_010', 'pt_011', 'pt_013', 'pt_009']
- `Upper back pain between shoulder blades` -- retrieved: ['pt_009', 'pt_005', 'pt_017', 'ex_016', 'pt_003']
- `Simple exercises that need no equipment at my desk` -- retrieved: ['ex_032', 'ex_007', 'ex_030', 'ex_039', 'ex_034']

### Per-Query Baseline Results

| # | Query | Hit@5 | MRR | Top-3 Retrieved |
|---|-------|-------|-----|-----------------|
| 1 | My neck is stiff from looking at my monitor all da | **N** | 0.00 | pt_001, pt_010, pt_011 |
| 2 | My wrists hurt from typing too much | Y | 1.00 | pt_007, pt_003, ex_003 |
| 3 | Quick stretch I can do between meetings | Y | 0.50 | pt_016, ex_050, ex_041 |
| 4 | Breathing exercise for stress before a presentatio | Y | 1.00 | ex_006, ex_037, ex_044 |
| 5 | How should I set up my desk and monitor? | Y | 0.33 | pt_021, pt_010, pt_006 |
| 6 | I feel sleepy every afternoon at work | Y | 1.00 | wa_003, wa_009, wa_002 |
| 7 | My eyes are tired from staring at screens | Y | 0.50 | pt_018, wa_001, ex_020 |
| 8 | Lower back pain from sitting all day | Y | 1.00 | pt_005, ex_017, pt_025 |
| 9 | Shoulder pain and rounded posture fix | Y | 1.00 | pt_009, ex_004, pt_017 |
| 10 | I feel stiff after sitting for hours | Y | 1.00 | pt_019, pt_025, pt_009 |
| 11 | How to stop slouching at my desk | Y | 0.33 | pt_015, pt_016, pt_005 |
| 12 | Exercises for better posture | Y | 0.50 | ex_010, ex_023, ex_022 |
| 13 | Carpal tunnel prevention for programmers | Y | 1.00 | ex_003, ex_045, pt_007 |
| 14 | Stress and anxiety relief at work | Y | 1.00 | wa_015, wa_008, wa_018 |
| 15 | What snacks should I eat at my desk? | Y | 1.00 | wa_006, wa_021, wa_014 |
| 16 | My legs feel numb from sitting too long | Y | 0.25 | pt_002, pt_004, pt_019 |
| 17 | How to sleep better as a workaholic | Y | 1.00 | wa_012, wa_003, wa_025 |
| 18 | Hip tightness from prolonged sitting | Y | 0.50 | pt_019, ex_009, pt_020 |
| 19 | 2 minute break ideas at my desk | Y | 0.25 | wa_011, wa_006, wa_024 |
| 20 | Upper back pain between shoulder blades | **N** | 0.00 | pt_009, pt_005, pt_017 |
| 21 | How much water should I drink at work | Y | 1.00 | wa_002, wa_006, wa_023 |
| 22 | Tingling in my fingers when typing | Y | 1.00 | pt_007, pt_003, pt_014 |
| 23 | Best position for dual monitors | Y | 1.00 | pt_010, pt_001, pt_006 |
| 24 | I can't focus and my mind is foggy | Y | 0.50 | ex_020, wa_010, wa_001 |
| 25 | Standing desk tips and proper usage | Y | 0.50 | pt_002, pt_012, pt_021 |
| 26 | Calming exercise before an important meeting | Y | 1.00 | wa_010, wa_013, ex_006 |
| 27 | My jaw is clenched and tight from stress | Y | 1.00 | wa_008, wa_015, ex_044 |
| 28 | How to take regular breaks using pomodoro | Y | 1.00 | wa_007, wa_024, wa_011 |
| 29 | Simple exercises that need no equipment at my desk | **N** | 0.00 | ex_032, ex_007, ex_030 |
| 30 | Weekend recovery after intense work week | Y | 1.00 | wa_025, wa_012, wa_017 |

## 4. Enhancement: Hybrid Search + Cross-Encoder Re-ranking

### Rationale

The baseline's main weakness is **ranking quality** (low Hit Rate@1, moderate MRR).
This stems from two limitations:

1. **Bi-encoder limitation:** `all-MiniLM-L6-v2` encodes query and document
   independently. It captures broad semantic similarity but misses fine-grained
   relevance signals (e.g., which specific exercise best matches a symptom).

2. **Vector-only search:** Some queries contain keywords that matter for relevance
   (e.g., "pomodoro", "carpal tunnel", "dual monitors") but may not be well
   captured by the embedding space.

### Enhancement Approach

We implement a three-stage hybrid retrieval pipeline:

```
User Query
    |
    +---> [Vector Search (top-15)] ---+
    |     (all-MiniLM-L6-v2)         |
    |                                 +---> [Merge & Deduplicate]
    +---> [BM25 Search (top-15)]  ---+         |
          (keyword matching)                   v
                                    [Cross-Encoder Re-ranking]
                                    (ms-marco-MiniLM-L-6-v2)
                                          |
                                          v
                                    [Top-5 Results]
```

**Component 1: BM25 Keyword Search**
- Uses `rank-bm25` (BM25Okapi) for term-frequency-based scoring
- Catches exact keyword matches the bi-encoder might miss
- Built from the same text representations used for embeddings
- Index built in-memory (100 docs = milliseconds)

**Component 2: Cross-Encoder Re-ranker**
- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers)
- Processes each (query, document) pair *jointly* through BERT
- Produces a single relevance score per pair
- Much more accurate than bi-encoder cosine similarity for relevance ranking
- Trained on MS MARCO passage ranking dataset

**Component 3: Hybrid Retriever**
- Over-fetches 15 candidates from each source (vector + BM25)
- Merges and deduplicates candidates
- Re-ranks the merged pool with the cross-encoder
- Returns top-5 final results

### Why This Should Work

- **BM25 + Vector = broader recall:** BM25 excels at keyword matching;
  vector search excels at semantic matching. Together they cover more ground.
- **Cross-encoder = precise ranking:** By processing query-document pairs jointly,
  the cross-encoder can assess fine-grained relevance that bi-encoders miss.
  This directly targets Hit Rate@1 and MRR improvement.
- **Well-established technique:** This hybrid approach is standard in modern IR
  systems and consistently outperforms either method alone.

## 5. Enhanced Results (Iteration 1: Hybrid + Re-ranking)

**Timestamp:** 2026-04-06T13:47:26.093713+00:00
**Retrieval time:** 27.89s for 30 queries

| Metric | Value |
|--------|-------|
| Hit Rate@1 | 76.7% |
| Hit Rate@3 | 93.3% |
| Hit Rate@5 | 96.7% |
| MRR | 0.8400 |
| Precision@5 | 0.4467 |

## 6. Comparison: Baseline vs Enhanced

| Metric | Baseline | Enhanced | Absolute Change | Relative Change |
|--------|----------|----------|-----------------|-----------------|
| Hit Rate@1 | 0.5667 | 0.7667 | +0.2000 | **+35.3%** |
| Hit Rate@3 | 0.8333 | 0.9333 | +0.1000 | **+12.0%** |
| Hit Rate@5 | 0.9000 | 0.9667 | +0.0667 | **+7.4%** |
| MRR | 0.7056 | 0.8400 | +0.1344 | **+19.0%** |
| Precision@5 | 0.3733 | 0.4467 | +0.0734 | **+19.7%** |

### Key Findings

1. **Hit Rate@1 improved by +35.3%** (from 56.7% to
   76.7%). This exceeds the 30% improvement threshold. The cross-encoder
   re-ranking dramatically improves the precision of the top-ranked result.

2. **MRR improved by +19.0%** (from 0.7056 to 0.8400).
   Relevant documents are consistently ranked higher.

3. **Hit Rate@5 improved from 90.0% to 96.7%.**
   The hybrid approach (vector + BM25) finds relevant documents that vector-only search missed.

4. **Precision@5 improved by +19.7%.**
   The context provided to the LLM is cleaner (more relevant, less noise).

### Latency Impact

- Baseline: 9.88s total (0.33s/query)
- Enhanced: 27.89s total (0.93s/query)
- The cross-encoder adds ~0.60s per query
  -- acceptable for a chat interface where LLM generation takes 1-3s anyway.

### Per-Query Improvement Analysis

| # | Query | Baseline MRR | Enhanced MRR | Change |
|---|-------|-------------|-------------|--------|
| tq_001 | My neck is stiff from looking at my monitor a | 0.00 | 0.20 | Improved |
| tq_002 | My wrists hurt from typing too much | 1.00 | 1.00 | Same |
| tq_003 | Quick stretch I can do between meetings | 0.50 | 1.00 | Improved |
| tq_004 | Breathing exercise for stress before a presen | 1.00 | 1.00 | Same |
| tq_005 | How should I set up my desk and monitor? | 0.33 | 0.33 | Same |
| tq_006 | I feel sleepy every afternoon at work | 1.00 | 0.33 | Degraded |
| tq_007 | My eyes are tired from staring at screens | 0.50 | 1.00 | Improved |
| tq_008 | Lower back pain from sitting all day | 1.00 | 0.33 | Degraded |
| tq_009 | Shoulder pain and rounded posture fix | 1.00 | 1.00 | Same |
| tq_010 | I feel stiff after sitting for hours | 1.00 | 1.00 | Same |
| tq_011 | How to stop slouching at my desk | 0.33 | 0.50 | Improved |
| tq_012 | Exercises for better posture | 0.50 | 0.50 | Same |
| tq_013 | Carpal tunnel prevention for programmers | 1.00 | 1.00 | Same |
| tq_014 | Stress and anxiety relief at work | 1.00 | 1.00 | Same |
| tq_015 | What snacks should I eat at my desk? | 1.00 | 1.00 | Same |
| tq_016 | My legs feel numb from sitting too long | 0.25 | 1.00 | Improved |
| tq_017 | How to sleep better as a workaholic | 1.00 | 1.00 | Same |
| tq_018 | Hip tightness from prolonged sitting | 0.50 | 1.00 | Improved |
| tq_019 | 2 minute break ideas at my desk | 0.25 | 1.00 | Improved |
| tq_020 | Upper back pain between shoulder blades | 0.00 | 1.00 | Improved |
| tq_021 | How much water should I drink at work | 1.00 | 1.00 | Same |
| tq_022 | Tingling in my fingers when typing | 1.00 | 1.00 | Same |
| tq_023 | Best position for dual monitors | 1.00 | 1.00 | Same |
| tq_024 | I can't focus and my mind is foggy | 0.50 | 1.00 | Improved |
| tq_025 | Standing desk tips and proper usage | 0.50 | 1.00 | Improved |
| tq_026 | Calming exercise before an important meeting | 1.00 | 1.00 | Same |
| tq_027 | My jaw is clenched and tight from stress | 1.00 | 1.00 | Same |
| tq_028 | How to take regular breaks using pomodoro | 1.00 | 1.00 | Same |
| tq_029 | Simple exercises that need no equipment at my | 0.00 | 0.00 | Same |
| tq_030 | Weekend recovery after intense work week | 1.00 | 1.00 | Same |

**Summary:** 10 queries improved, 2 degraded, 18 unchanged.

## 7. Implementation Details

### New Components

| File | Description |
|------|-------------|
| `src/bm25_search.py` | BM25Okapi keyword search over knowledge base |
| `src/reranker.py` | Cross-encoder re-ranker (ms-marco-MiniLM-L-6-v2) |
| `src/hybrid_retriever.py` | Orchestrates vector + BM25 + re-ranking pipeline |
| `evaluation/metrics.py` | Hit Rate@K, MRR, Precision@K computation |
| `evaluation/run_evaluation.py` | Automated evaluation runner |
| `evaluation/test_queries.json` | 30-query ground-truth test set |
| `evaluation/generate_report.py` | This report generator |

### Modified Components

| File | Change |
|------|--------|
| `config.py` | Added hybrid retrieval configuration constants |
| `src/rag_pipeline.py` | Added optional `hybrid_retriever` parameter |
| `app.py` | Wired up hybrid retriever when `USE_HYBRID_RETRIEVAL=True` |
| `requirements.txt` | Added `rank-bm25>=0.2.2` |

### New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `rank-bm25` | >=0.2.2 | BM25Okapi keyword search algorithm |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | (model) | Cross-encoder for re-ranking |

Note: The cross-encoder model is loaded via `sentence-transformers` which is
already a project dependency. No new Python package is needed for it.

## 8. Conclusion

The hybrid search + cross-encoder re-ranking enhancement achieved:

- **+35.3% improvement in Hit Rate@1** (primary target metric, above 30% threshold)
- **+19.0% improvement in MRR** (secondary metric)
- **Hit Rate@5 from 90.0% to 96.7%** (near-perfect retrieval coverage)

The enhancement qualitatively improves the system by:

1. **Adding a fundamentally different retrieval signal** (BM25 keyword matching)
   alongside the existing semantic vector search
2. **Replacing approximate bi-encoder ranking** with precise cross-encoder scoring
   that processes query-document pairs jointly
3. **Maintaining backward compatibility** -- the enhancement is an optional pipeline
   component controlled by `config.USE_HYBRID_RETRIEVAL`

### Reproducibility

To reproduce these results:
```bash
# Install dependencies
pip install -r requirements.txt

# Populate the vector database
python -m src.populate_db

# Run evaluation (baseline + enhanced)
python -m evaluation.run_evaluation --mode all

# Generate this report
python -m evaluation.generate_report

# Launch the enhanced application
streamlit run app.py
```
