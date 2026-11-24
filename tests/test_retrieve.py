import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieve import Retriever, in_month, index_text, months_named, normalize, rrf, tokenize

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


MIXED = [dict(c, kind="statement" if i in (3, 4) else "minutes") for i, c in enumerate(CHUNKS)]


def test_statement_slots_reserve_places_for_statements():
    r = Retriever(MIXED, encoder=fake_encoder, cache_dir=None, statement_slots=2)
    # minutes chunk 0 matches best; statements 3 and 4 match only weakly (one word each)
    q = "inflation prices rose sharply, bank, yields"
    for method in ("bm25", "embeddings", "hybrid"):
        hits = r.search(q, method=method, k=3)
        kinds = [h["kind"] for h in hits]
        assert kinds[:2] == ["statement", "statement"], method   # 2 statements first, even if weaker
        assert kinds[2:] == ["minutes"] * len(kinds[2:]), method
        assert [h["rank"] for h in hits] == list(range(1, len(hits) + 1))
    assert r.search(q, method="hybrid", k=1)[0]["kind"] == "statement"
    # BM25 never returns chunks with no shared words, so a statement slot can stay empty
    assert [h["kind"] for h in r.search("inflation prices", method="bm25", k=3)] == ["minutes"]


def test_statement_slots_fill_from_minutes_when_statements_run_out():
    r = Retriever(MIXED, encoder=fake_encoder, cache_dir=None, statement_slots=4)
    hits = r.search("inflation", method="embeddings", k=5)
    assert [h["kind"] for h in hits].count("statement") == 2 and len(hits) == 5


def test_no_split_by_default_matches_plain_ranking():
    plain = Retriever(MIXED, encoder=fake_encoder, cache_dir=None, statement_slots=0)
    top = plain.search("inflation prices", method="embeddings", k=1)[0]
    assert top["chunk_id"] == "d#0" and top["kind"] == "minutes"


@pytest.mark.parametrize("question, expected", [
    ("Why did the Fed cut in September 2024?", {(2024, 9)}),
    ("What was decided at the July 28-29, 2026 meeting?", {(2026, 7)}),
    ("the May 2–3, 2023 meeting", {(2023, 5)}),
    ("Sept. 2024 and Dec 2024", {(2024, 9), (2024, 12)}),
    ("What happened in early 2025?", set()),        # no month named
    ("rates in 2023", set()),                       # bare year: no filter
    ("Which meeting approved a 10-2 vote?", set()),
])
def test_months_named(question, expected):
    assert months_named(question) == expected


def test_in_month_handles_meetings_that_span_two_months():
    assert in_month("2024-09-18", 2024, 9)
    assert in_month("2023-02-01", 2023, 1) and in_month("2023-02-01", 2023, 2)   # Jan 31-Feb 1 meeting
    assert in_month("2022-01-01", 2021, 12)                                       # year boundary
    assert not in_month("2024-09-18", 2024, 8) and not in_month("2024-10-02", 2024, 9)


DATED = [dict(c, date=d) for c, d in zip(CHUNKS, ["2024-09-18", "2024-07-31", "2024-09-18", "2023-03-22", "2024-11-07"])]


def test_date_filter_searches_only_the_named_meeting():
    r = Retriever(DATED, encoder=fake_encoder, cache_dir=None, date_filter=True)
    for method in ("bm25", "embeddings", "hybrid"):
        hits = r.search("jobs and the funds rate in September 2024", method=method, k=5)
        assert hits and all(h["date"] == "2024-09-18" for h in hits), method
    # mostly about jobs, a bit about inflation: the July jobs chunk wins without the filter,
    # the September inflation chunk wins with it
    q = "job gains, unemployment and inflation in September 2024"
    assert r.search(q, "embeddings", k=1)[0]["chunk_id"] == "d#0"
    off = Retriever(DATED, encoder=fake_encoder, cache_dir=None, date_filter=False)
    assert off.search(q, "embeddings", k=1)[0]["chunk_id"] == "d#1"


def test_date_filter_falls_back_when_no_meeting_that_month():
    r = Retriever(DATED, encoder=fake_encoder, cache_dir=None, date_filter=True)
    assert r.date_scope("rates in June 2019") is None
    assert len(r.search("bank deposits in June 2019", "embeddings", k=5)) == 5


def test_date_filter_combines_with_statement_slots():
    mixed = [dict(c, kind="statement" if i == 2 else "minutes") for i, c in enumerate(DATED)]
    r = Retriever(mixed, encoder=fake_encoder, cache_dir=None, date_filter=True, statement_slots=1)
    hits = r.search("funds rate in September 2024", "hybrid", k=5)
    assert [h["chunk_id"] for h in hits] == ["d#2", "d#0"]  # Sept statement, then Sept minutes; nothing else


def test_unknown_method(retriever):
    with pytest.raises(ValueError):
        retriever.search("rates", method="tfidf")
