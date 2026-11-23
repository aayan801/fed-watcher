# Fed Watcher: notes for Claude Code

RAG tool over FOMC statements and minutes (2021 onwards). Portfolio project for internship applications: I need to be able to explain every core part in an interview.

## Rules
- **Claude builds the whole project, one step at a time** (steps in Plan below), including chunking, retrieval/scoring and the evaluation metrics.
- **After each step, stop.** Explain what was built in plain words, in under 10 lines, then ask me 2 interview questions about it. Don't start the next step until I've answered.
- Run `pytest -q` after changes. Keep the tests in `tests/` as the definition of correct.
- Never commit `data/` or any API key. Keys go in `.env` (already gitignored).
- Keep commits small, with plain messages.

## Layout
- `src/download.py`: fetches FOMC docs into `data/raw/*.json` (`doc_id`, `kind`, `date`, `url`, `text`)
- `src/chunk.py`: paragraph-aware chunking with overlap
- `tests/test_chunk.py`: chunking contract

## Plan
1. Chunking
2. Retrieval: BM25 vs embeddings vs hybrid
3. Evaluation on 40 hand-written questions (top-5 hit rate)
4. LLM answer step with citations
5. Web app: FastAPI + Streamlit
6. Deploy
