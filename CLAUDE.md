# Fed Watcher: notes for Claude Code

RAG tool over FOMC statements and minutes (2021 onwards). Portfolio project for internship applications: I need to be able to explain every core part in an interview.

## Rules
- **Do not write these for me:** `chunk_document()` in `src/chunk.py`, the retrieval/scoring code, and the evaluation metrics. For those, give hints, point to the bug, or explain the concept. Never paste a full solution.
- You can freely write: scaffolding, downloaders, config, the FastAPI/Streamlit UI, deployment files, README.
- Run `pytest -q` after changes. Keep the tests in `tests/` as the definition of correct.
- Never commit `data/` or any API key. Keys go in `.env` (already gitignored).
- Keep commits small, with plain messages.

## Layout
- `src/download.py`: fetches FOMC docs into `data/raw/*.json` (`doc_id`, `kind`, `date`, `url`, `text`)
- `src/chunk.py`: paragraph-aware chunking with overlap (mine)
- `tests/test_chunk.py`: chunking contract

## Plan
Chunking, then retrieval (BM25 vs embeddings vs hybrid), then an eval on 40 hand-written questions (top-5 hit rate), then the LLM answer step with citations, then FastAPI + Streamlit, then deploy.
