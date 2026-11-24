# Fed Watcher

Ask questions about Federal Reserve (FOMC) policy statements and minutes, and get answers with citations to the exact meeting.

**Status:** download, chunking, retrieval, evaluation and the LLM answer step are done. The web app (FastAPI + Streamlit) is next.

## Plan

1. ✅ Download FOMC statements and minutes (2021 onwards) from federalreserve.gov
2. ✅ Chunk documents (paragraph-aware, with overlap)
3. ✅ Retrieval: BM25 vs embeddings vs hybrid, evaluated on a 40-question set
4. ✅ Answer step with an LLM (Claude Sonnet 5.5), citing the meetings it used
5. ⬜ FastAPI endpoint + small web UI
6. ⬜ Deploy

## Example

```bash
python src/answer.py "Who dissented at the July 2026 meeting and what did they want instead?"
```

> Beth M. Hammack, Neel Kashkari, and Lorie K. Logan dissented at the July 2026 meeting. They preferred to raise the target range for the federal funds rate by 1/4 percentage point, rather than maintain it at 3-1/2 to 3-3/4 percent [statement 2026-07-29][minutes 2026-07-29].

Every claim cites a meeting as `[kind date]`. The model sees only the top-5 retrieved chunks; if they don't answer the question, it replies "not found in the documents".

## Results

Top-5 hit rate of hybrid retrieval on 40 questions. **Chunk level** counts a hit only if a retrieved chunk contains the answer; document level counts any chunk from the right document. Full log: [`eval/experiments.md`](eval/experiments.md).

| Setup | Document level | Chunk level |
|---|---|---|
| Hybrid (BM25 weighted 2x + bge-small embeddings) | 88% | 52% |
| + 2 statement and 3 minutes chunks (not kept) | 78% | 52% |
| + only search the meeting named in the question (month and year) | 90% | 68% |
| + both (current default) | 80% | **78%** |

These numbers were tuned on the same 40 questions they are measured on, so they are optimistic.

## Run

```bash
pip install -r requirements.txt
python src/download.py
python src/retrieve.py "How worried were officials about AI-related debt?" --method hybrid
python src/evaluate.py      # top-5 hit rate (document and chunk level) on eval/questions.json
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env   # your key; .env is gitignored
python src/answer.py "Why did the Fed cut rates in September 2024?"
pytest -q
```
