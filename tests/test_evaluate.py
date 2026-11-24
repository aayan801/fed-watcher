import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import evaluate, format_tables, hit_rate, is_hit

ROOT = Path(__file__).resolve().parent.parent


def test_is_hit_any_relevant_doc_counts():
    assert is_hit(["a", "b", "c"], ["c", "z"])
    assert not is_hit(["a", "b"], ["z"])
    assert not is_hit([], ["z"])


def test_hit_rate():
    assert hit_rate([True, False, True, True]) == 0.75
    with pytest.raises(ValueError):
        hit_rate([])


class FakeRetriever:
    """Returns fixed doc_ids per (question, method), so the scoring can be checked by hand."""
    RESULTS = {
        ("q1", "bm25"): ["d1"], ("q1", "embeddings"): ["d9"], ("q1", "hybrid"): ["d9", "d1"],
        ("q2", "bm25"): ["d9"], ("q2", "embeddings"): ["d2"], ("q2", "hybrid"): ["d2"],
        ("q3", "bm25"): [], ("q3", "embeddings"): ["d9"], ("q3", "hybrid"): ["d9"],
    }

    def search(self, question, method, k):
        return [{"doc_id": d} for d in self.RESULTS[(question, method)][:k]]


def test_evaluate_overall_and_by_type():
    qs = [{"id": "1", "type": "keyword", "question": "q1", "relevant": ["d1"]},
          {"id": "2", "type": "paraphrase", "question": "q2", "relevant": ["d2"]},
          {"id": "3", "type": "paraphrase", "question": "q3", "relevant": ["d3"]}]
    res = evaluate(FakeRetriever(), qs, k=5)
    assert res["overall"] == {"bm25": 1 / 3, "embeddings": 1 / 3, "hybrid": 2 / 3}
    assert res["by_type"]["keyword"] == {"n": 1, "bm25": 1.0, "embeddings": 0.0, "hybrid": 1.0}
    assert res["by_type"]["paraphrase"] == {"n": 2, "bm25": 0.0, "embeddings": 0.5, "hybrid": 0.5}
    assert "paraphrase" in format_tables(res)


def test_k_limits_what_counts():
    qs = [{"id": "1", "type": "keyword", "question": "q1", "relevant": ["d1"]}]
    assert evaluate(FakeRetriever(), qs, k=1)["overall"]["hybrid"] == 0.0  # d1 is 2nd
    assert evaluate(FakeRetriever(), qs, k=2)["overall"]["hybrid"] == 1.0


def test_question_file_is_well_formed():
    qs = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    assert len(qs) == 40
    assert len({q["id"] for q in qs}) == 40
    types = {"keyword", "paraphrase", "fraction", "dash", "accent", "date"}
    for q in qs:
        assert q["type"] in types and q["question"].strip() and q["relevant"], q["id"]
