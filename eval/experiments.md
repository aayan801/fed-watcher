# Retrieval experiments

Top-5 hit rate on the 40 questions in `eval/questions.json` (`python src/evaluate.py`).
One change per row, each kept only if it helps; every row includes the changes above it.

Caveat: the questions were also used to pick these changes, so the numbers are optimistic.
Types with n < 6 (fraction, dash, accent, date) move 20-50 points per question; read them as hints.

## Overall

| # | Change | BM25 | Embeddings | Hybrid |
|---|---|---|---|---|
| 0 | Baseline: all-MiniLM-L6-v2, equal-weight RRF, no header | 82% | 38% | 65% |
| 1 | Embedding model: BAAI/bge-small-en-v1.5 (512-token limit, query prefix) | 82% | 62% | 82% |

## Hybrid by question type

| # | keyword (12) | paraphrase (12) | fraction (5) | dash (5) | accent (4) | date (2) |
|---|---|---|---|---|---|---|
| 0 | 83% | 50% | 60% | 40% | 100% | 50% |
| 1 | 92% | 67% | 80% | 100% | 75% | 100% |

## Notes

- **0 → 1:** all-MiniLM-L6-v2 reads at most 256 tokens, and 1,089 of 2,465 chunks (44%) are longer
  (median 248, max 389 tokens), so their endings were cut off before embedding. bge-small reads 512 and
  cuts none. It also expects questions to start with a retrieval prefix, which MiniLM doesn't use.
