# Fed Watcher

Ask questions about Federal Reserve (FOMC) policy statements and minutes, and get answers with citations to the exact meeting.

**Status: in progress.** Data pipeline and chunking first; retrieval evaluation and the LLM answer step next.

## Plan

1. Download FOMC statements and minutes (2021 onwards) from federalreserve.gov
2. Chunk documents (paragraph-aware, with overlap)
3. Retrieval: BM25 vs embeddings vs hybrid, evaluated on a hand-written question set
4. Answer step with an LLM, citing the meetings it used
5. FastAPI endpoint + small web UI

## Run

```bash
pip install -r requirements.txt
python src/download.py
python src/retrieve.py "How worried were officials about AI-related debt?" --method hybrid
python src/evaluate.py      # top-5 hit rate for BM25, embeddings, hybrid on eval/questions.json
pytest -q
```
