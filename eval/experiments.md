# Retrieval experiments

Top-5 hit rate on the 40 questions in `eval/questions.json` (`python src/evaluate.py`).
One change per row, each kept only if it helps; every row includes the kept changes above it.

- **Document level:** a top-5 chunk comes from a document that answers the question.
- **Chunk level:** a top-5 chunk actually *contains* the answer (matches the question's `answer_pattern`).
  This is what the answer step needs: the LLM only sees the chunks. Added at #4; rows 0-3 were re-scored.

Caveat: the questions were also used to pick these changes, so the numbers are optimistic.
Types with n < 6 (fraction, dash, accent, date) move 20-50 points per question; read them as hints.

## Overall

| # | Change | Doc: BM25 | Doc: Emb | Doc: Hybrid | **Chunk: BM25** | **Chunk: Emb** | **Chunk: Hybrid** |
|---|---|---|---|---|---|---|---|
| 0 | Baseline: all-MiniLM-L6-v2, equal-weight RRF, no header | 82% | 38% | 65% | 57% | 25% | 38% |
| 1 | Embedding model: BAAI/bge-small-en-v1.5 (512-token limit, query prefix) | 82% | 62% | 82% | 57% | 32% | 48% |
| 2 | Weighted hybrid: BM25 counts 2x in RRF | 82% | 62% | 88% | 57% | 32% | 52% |
| 3 | Date header "FOMC minutes, 2024-09-18:" on each chunk (**not kept**: off by default) | 82% | 62% | 82% | 60% | 32% | 50% |
| 4 | Chunk-level eval added (no retrieval change; same as #2) | 82% | 62% | 88% | 57% | 32% | 52% |
| 5 | Split: top 2 statements + top 3 minutes (**not kept**: off by default) | 80% | 57% | 78% | 62% | 38% | 52% |
| 6 | Date filter: a question naming a month and year searches only that meeting (on #4) | 85% | 72% | 90% | 68% | 42% | 68% |
| 7 | Date filter + 2 statements & 3 minutes (**current default**) | 82% | 72% | 80% | 80% | 62% | **78%** |

## Hybrid by question type

Document level:

| # | keyword (12) | paraphrase (12) | fraction (5) | dash (5) | accent (4) | date (2) |
|---|---|---|---|---|---|---|
| 0 | 83% | 50% | 60% | 40% | 100% | 50% |
| 1 | 92% | 67% | 80% | 100% | 75% | 100% |
| 2 | 100% | 67% | 80% | 100% | 100% | 100% |
| 3 | 92% | 58% | 80% | 100% | 100% | 100% |
| 5 | 92% | 42% | 80% | 100% | 100% | 100% |
| 6 | 100% | 67% | 100% | 100% | 100% | 100% |
| 7 | 92% | 42% | 100% | 100% | 100% | 100% |

Chunk level:

| # | keyword (12) | paraphrase (12) | fraction (5) | dash (5) | accent (4) | date (2) |
|---|---|---|---|---|---|---|
| 0 | 50% | 25% | 40% | 0% | 100% | 0% |
| 1 | 67% | 33% | 60% | 20% | 75% | 0% |
| 2 | 75% | 33% | 60% | 20% | 100% | 0% |
| 3 | 67% | 25% | 60% | 40% | 100% | 0% |
| 5 | 83% | 25% | 60% | 20% | 100% | 0% |
| 6 | 75% | 42% | 80% | 80% | 100% | 50% |
| 7 | 92% | 42% | 80% | 100% | 100% | 100% |

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
- **#4, chunk level:** hybrid finds a right *document* for 88% of questions but a chunk *containing the
  answer* for only 52%. Of its 19 chunk-level misses, 14 had the right document and the wrong chunk.
  That matches the answer run, where 2 of 4 answerable questions came back "not found in the documents".
  Dissents and decisions sit in short statements (120 of 2,465 chunks), which minutes chunks crowd out.
- **4 → 5, statement slots:** reserving 2 of the 5 places for statements gained the questions that only a
  statement answers well (q03 end of runoff, q04 redemption cap, q12 Middle East) but lost three answered
  only in the minutes (q09 Bullard's reasons, q11 debt limit, q23 repo facility), because minutes now get
  3 places instead of 5. Hybrid chunk level stays 52%; doc level drops 88% → 78%. With 1 statement slot:
  doc 82%, chunk 50%. Not kept on its own (`--statement-slots`, default 0); re-tested with the date filter.
- **4 → 6, date filter** (built on #4, not #5): 14 of the 40 questions name a month and year ("September
  2024", "July 28-29, 2026"); for those, only chunks from that meeting are searched. A meeting stored
  under its last day also counts for the previous month if it ends on the 1st (January 31-February 1).
  If no meeting matches, everything is searched. Hybrid chunk level 52% → 68%: gained q16, q25, q30, q32,
  q34, q39, lost none. All gains come from the 14 dated questions, so a real user's share of dated
  questions decides how much this helps. Kept (`--date-filter`, on by default).
- **6 → 7, date filter + statement slots:** inside one meeting, reserving 2 places for its statement no
  longer pushes out the answer, so the split now pays off: hybrid chunk level 68% → 78% (vs #4: gained 12
  questions, lost q11 debt limit and q23 repo facility, both answered only in minutes and naming no
  month). With 1 statement slot: chunk 75%, doc 85%. Doc level drops 90% → 80%, but chunk level is what
  the answer step needs, so this is the new default. Its weak spot: undated questions answered only in
  minutes lose 2 of their 5 places. Paraphrase (42%) is still the weakest type.
