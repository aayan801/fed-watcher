"""Answer-step tests. No real API calls: every test uses a fake LLM or no LLM at all."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.answer import NOT_FOUND, SYSTEM_PROMPT, answer, build_user_prompt, check_citations, cost, load_api_key

CHUNKS = [
    {"chunk_id": "minutes_20240918#3", "doc_id": "minutes_20240918", "kind": "minutes", "date": "2024-09-18",
     "url": "https://example.org/m", "text": "Members agreed to lower the target range by 1/2 percentage point.", "rank": 1},
    {"chunk_id": "statement_20240918#1", "doc_id": "statement_20240918", "kind": "statement", "date": "2024-09-18",
     "url": "https://example.org/s", "text": "Voting against this action was Michelle W. Bowman.", "rank": 2},
]


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks, self.calls = chunks, []

    def search(self, question, method, k):
        self.calls.append((question, method, k))
        return self.chunks[:k]


class FakeLLM:
    """Records what it was sent and returns a canned reply."""

    def __init__(self, reply, usage=None):
        self.reply, self.usage, self.sent = reply, usage or {"input_tokens": 1000, "output_tokens": 100}, []

    def __call__(self, system, user):
        self.sent.append((system, user))
        return self.reply, self.usage


def test_prompt_contains_only_the_retrieved_chunks_with_labels():
    p = build_user_prompt("Why the cut?", CHUNKS)
    assert '<source id="1" kind="minutes" date="2024-09-18">' in p
    assert '<source id="2" kind="statement" date="2024-09-18">' in p
    assert CHUNKS[0]["text"] in p and CHUNKS[1]["text"] in p
    assert p.rstrip().endswith("Question: Why the cut?")


def test_system_prompt_states_the_three_rules():
    assert "only the sources" in SYSTEM_PROMPT
    assert "[minutes 2024-09-18]" in SYSTEM_PROMPT       # citation format
    assert NOT_FOUND in SYSTEM_PROMPT                    # exact refusal phrase


def test_answer_uses_top5_hybrid_and_checks_citations():
    r = FakeRetriever(CHUNKS)
    llm = FakeLLM("The Fed cut by half a point [minutes 2024-09-18]. Bowman dissented [statement 2024-09-18].")
    a = answer("Why did the Fed cut in September 2024?", r, llm)
    assert r.calls == [("Why did the Fed cut in September 2024?", "hybrid", 5)]
    assert not a.not_found
    assert a.citations == [("minutes", "2024-09-18"), ("statement", "2024-09-18")]
    assert a.bad_citations == []
    assert len(llm.sent) == 1 and llm.sent[0][0] == SYSTEM_PROMPT


def test_citation_to_a_document_not_retrieved_is_flagged():
    cited, bad = check_citations("Rates rose [minutes 2022-03-16]. Cut [minutes 2024-09-18].", CHUNKS)
    assert cited == [("minutes", "2022-03-16"), ("minutes", "2024-09-18")]
    assert bad == [("minutes", "2022-03-16")]


@pytest.mark.parametrize("reply", ["not found in the documents", "Not found in the documents.", "  not found in the documents\n"])
def test_not_found_reply_is_detected(reply):
    a = answer("What is the Fed's view on bitcoin?", FakeRetriever(CHUNKS), FakeLLM(reply))
    assert a.not_found and a.citations == []


def test_no_chunks_means_not_found_without_calling_the_llm():
    llm = FakeLLM("should not be used")
    a = answer("xyzzy", FakeRetriever([]), llm)
    assert a.not_found and a.text == NOT_FOUND and llm.sent == []


def test_cost_uses_sonnet_prices():
    assert cost({"input_tokens": 1_000_000, "output_tokens": 0}) == pytest.approx(2.00)
    assert cost({"input_tokens": 0, "output_tokens": 1_000_000}) == pytest.approx(10.00)
    assert cost({"input_tokens": 2000, "output_tokens": 300}) == pytest.approx(0.007)


def test_api_key_comes_from_env_file_only(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-shell")   # must be ignored
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-from-dotenv\n")
    assert load_api_key(env) == "sk-from-dotenv"


@pytest.mark.parametrize("content", [None, "", "OTHER=1\n", "ANTHROPIC_API_KEY=\n"])
def test_missing_key_fails_with_instructions(tmp_path, monkeypatch, content):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-shell")
    env = tmp_path / ".env"
    if content is not None:
        env.write_text(content)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        load_api_key(env)

