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
| 2 | Weighted hybrid: BM25 counts 2x in RRF | 82% | 62% | 88% |
| 3 | Date header "FOMC minutes, 2024-09-18:" on each chunk (**not kept**: off by default) | 82% | 62% | 82% |

## Hybrid by question type

| # | keyword (12) | paraphrase (12) | fraction (5) | dash (5) | accent (4) | date (2) |
|---|---|---|---|---|---|---|
| 0 | 83% | 50% | 60% | 40% | 100% | 50% |
| 1 | 92% | 67% | 80% | 100% | 75% | 100% |
| 2 | 100% | 67% | 80% | 100% | 100% | 100% |
| 3 | 92% | 58% | 80% | 100% | 100% | 100% |

## Notes

- **0 → 1:** all-MiniLM-L6-v2 reads at most 256 tokens, and 1,089 of 2,465 chunks (44%) are longer
  (median 248, max 389 tokens), so their endings were cut off before embedding. bge-small reads 512 and
  cuts none. It also expects questions to start with a retrieval prefix, which MiniLM doesn't use.
- **1 → 2:** BM25 weight sweep, hybrid overall: 0.5 → 78%, 1.0 → 82%, 1.5 → 82%, **2.0 → 88%**,
  3.0 → 82%, 5.0 → 88%. Not smooth (each step is 2-3 questions), so this is noise-level tuning on the
  test set itself. Picked 2.0: smallest weight at the top, and it keeps paraphrase at 67% (5.0 drops it to 58%).
- **2 → 3:** the header lost q11 (debt limit, 2023) and q23 (repo facility) and gained nothing: the
  2 date questions were already hits. Embeddings alone moved within noise (62% both, date 50% → 100%).
  Kept as `--date-header`, off by default. This eval can't really judge it: only 2 date questions, and
  ISO dates ("2024-09-18") don't contain the month names people type ("September 2024").
  Re-test with more date questions and a header like "FOMC minutes, September 18, 2024".
