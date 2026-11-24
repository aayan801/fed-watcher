import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieve import Retriever, index_text, normalize, rrf, tokenize

TEXTS = [
    "Inflation remains elevated and consumer prices rose sharply.",            # 0 inflation
    "Job gains were solid and the unemployment rate stayed low.",              # 1 jobs
    "The Committee voted to lower the federal funds rate by 25 basis points.",  # 2 rate cut
    "Stress at Silicon Valley Bank led to deposit outflows at regional banks.",  # 3 banks
    "Treasury yields moved higher over the intermeeting period.",              # 4 yields
]
CHUNKS = [{"chunk_id": f"d#{i}", "doc_id": "d", "kind": "minutes", "date": "2026-01-01",
           "url": "https://example.org", "text": t} for i, t in enumerate(TEXTS)]

# Fake "meaning" model: each topic is one dimension, and synonyms map to the same topic.
# That lets us test embedding search without downloading a real model.
TOPICS = [{"inflation", "prices", "price", "pressures"}, {"job", "jobs", "unemployment", "labor"},
          {"cut", "lower", "funds"}, {"bank", "banks", "deposit", "svb"}, {"yields", "treasury"}]


def fake_encoder(texts):
    out = np.zeros((len(texts), len(TOPICS)), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in tokenize(text):
            for dim, topic in enumerate(TOPICS):
                out[row, dim] += word in topic
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


@pytest.fixture
def retriever():
    return Retriever(CHUNKS, encoder=fake_encoder, cache_dir=None)


def test_normalize_fractions():
    assert normalize("5¼ to 5½ percent") == "5-1/4 to 5-1/2 percent"
    assert normalize("by ¼ percentage point") == "by 1/4 percentage point"


def test_normalize_dashes_and_accents():
    assert normalize("May 2–3 meeting — rates") == "May 2-3 meeting - rates"
    assert normalize("−0.5 percent") == "-0.5 percent"
    assert normalize("Café Montréal") == "Cafe Montreal"


def test_tokenize_same_for_question_and_document_forms():
    assert tokenize("Rate at 5¼ to 5½ percent") == tokenize("rate at 5-1/4 to 5-1/2 percent")
    assert tokenize("MAY 2–3") == tokenize("may 2-3") == ["may", "2", "3"]
    assert tokenize("12-month change") == ["12", "month", "change"]


def test_embedding_model_sees_normalized_text():
    seen = []

    def recording_encoder(texts):
        seen.extend(texts)
        return fake_encoder(texts)

    r = Retriever([dict(CHUNKS[0], text="Rates at 5¼ percent—Café")], encoder=recording_encoder, cache_dir=None)
    r.search("5½ percent?", method="embeddings")
    assert seen == ["Rates at 5-1/4 percent-Cafe", "5-1/2 percent?"]


def test_query_prefix_only_on_questions():
    seen = []

    def recording_encoder(texts):
        seen.extend(texts)
        return fake_encoder(texts)

    r = Retriever(CHUNKS[:1], encoder=recording_encoder, cache_dir=None, query_prefix="query: ")
    r.search("inflation", method="embeddings")
    assert seen == [TEXTS[0], "query: inflation"]  # chunk without prefix, question with it


def test_known_model_gets_its_query_prefix():
    bge = Retriever(CHUNKS, cache_dir=None, model_name="BAAI/bge-small-en-v1.5")
    minilm = Retriever(CHUNKS, cache_dir=None, model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert bge.query_prefix.startswith("Represent this sentence") and minilm.query_prefix == ""


def test_embeddings_k_larger_than_corpus(retriever):
    hits = retriever.search("inflation", method="embeddings", k=50)
    assert len(hits) == len(CHUNKS) and all(h["chunk_id"].startswith("d#") for h in hits)


def test_bm25_matches_keywords(retriever):
    hits = retriever.search("Silicon Valley Bank deposits", method="bm25", k=3)
    assert hits[0]["chunk_id"] == "d#3"
    assert [h["rank"] for h in hits] == list(range(1, len(hits) + 1))


def test_bm25_returns_nothing_without_shared_words(retriever):
    assert retriever.search("xyzzy", method="bm25") == []


def test_embeddings_match_meaning_not_words(retriever):
    # "labor market" shares no word with chunk 1, but the same topic (jobs)
    question = "labor market"
    assert retriever.search(question, method="bm25") == []  # no keyword overlap
    assert retriever.search(question, method="embeddings", k=1)[0]["chunk_id"] == "d#1"


def test_hybrid_returns_k_unique_chunks_with_metadata(retriever):
    hits = retriever.search("Did the Committee cut the funds rate?", method="hybrid", k=3)
    assert hits[0]["chunk_id"] == "d#2"
    assert len(hits) == 3 and len({h["chunk_id"] for h in hits}) == 3
    assert {"chunk_id", "doc_id", "date", "url", "text", "rank"} <= set(hits[0])


def test_rrf_rewards_agreement():
    # chunk 7 is 2nd in both lists; chunks 1 and 2 are each 1st in one list but missing from the other
    assert rrf([[1, 7, 3], [2, 7, 4]])[0] == 7


def test_rrf_weights_break_disagreements():
    # BM25 says 1 then 2, embeddings say 2 then 1: equal weights tie (lower index wins), a weight decides
    assert rrf([[1, 2], [2, 1]])[0] == 1
    assert rrf([[1, 2], [2, 1]], weights=[1.0, 2.0])[0] == 2
    assert rrf([[1, 2], [2, 1]], weights=[2.0, 1.0])[0] == 1
    with pytest.raises(ValueError):
        rrf([[1], [2]], weights=[1.0])


def test_hybrid_uses_bm25_weight():
    # BM25 only matches "Treasury" (chunk 4); the fake meaning model ranks the jobs chunk (1) first
    q = "Treasury labor market jobs"
    bm25_only = Retriever(CHUNKS, encoder=fake_encoder, cache_dir=None, bm25_weight=100.0)
    emb_only = Retriever(CHUNKS, encoder=fake_encoder, cache_dir=None, bm25_weight=0.0)
    assert bm25_only.search(q, "bm25", k=1)[0]["chunk_id"] == "d#4"
    assert emb_only.search(q, "embeddings", k=1)[0]["chunk_id"] == "d#1"
    assert bm25_only.search(q, "hybrid", k=1)[0]["chunk_id"] == "d#4"
    assert emb_only.search(q, "hybrid", k=1)[0]["chunk_id"] == "d#1"


def test_date_header_makes_date_searchable_but_keeps_text():
    chunks = [dict(c, date=f"202{i}-01-01") for i, c in enumerate(CHUNKS)]
    plain = Retriever(chunks, encoder=fake_encoder, cache_dir=None, date_header=False)
    header = Retriever(chunks, encoder=fake_encoder, cache_dir=None, date_header=True)
    assert plain.search("2023", "bm25") == []
    hits = header.search("2023", "bm25")
    assert hits[0]["chunk_id"] == "d#3"
    assert hits[0]["text"] == TEXTS[3]  # returned text has no header
    assert index_text(chunks[3], True) == "FOMC minutes, 2023-01-01: " + TEXTS[3]


def test_embedding_model_sees_header():
    seen = []

    def recording_encoder(texts):
        seen.extend(texts)
        return fake_encoder(texts)

    Retriever(CHUNKS[:1], encoder=recording_encoder, cache_dir=None, date_header=True).search("x", "embeddings")
    assert seen[0] == "FOMC minutes, 2026-01-01: " + TEXTS[0]


def test_unknown_method(retriever):
    with pytest.raises(ValueError):
        retriever.search("rates", method="tfidf")
