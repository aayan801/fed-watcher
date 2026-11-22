import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.chunk import chunk_document


def doc(text):
    return {"doc_id": "statement_20260128", "kind": "statement", "date": "2026-01-28",
            "url": "https://example.org", "text": text}


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_empty():
    assert chunk_document(doc("")) == []
    assert chunk_document(doc("   \n\n  ")) == []


def test_short_doc_is_one_chunk():
    out = chunk_document(doc("Rates held steady.\n\nInflation remains elevated."), max_words=50)
    assert len(out) == 1
    assert out[0]["text"] == "Rates held steady. Inflation remains elevated."
    assert out[0]["chunk_id"] == "statement_20260128#0"
    assert out[0]["date"] == "2026-01-28" and out[0]["kind"] == "statement"


def test_paragraphs_kept_whole():
    p1, p2, p3 = words(60, "a"), words(60, "b"), words(60, "c")
    out = chunk_document(doc(f"{p1}\n\n{p2}\n\n{p3}"), max_words=130, overlap=10)
    # p1 + p2 fit (120 words); p3 would overflow -> new chunk
    assert len(out) == 2
    assert out[0]["text"] == f"{p1} {p2}"
    assert out[1]["text"].endswith(p3)


def test_overlap():
    p1, p2 = words(60, "a"), words(60, "b")
    out = chunk_document(doc(f"{p1}\n\n{p2}"), max_words=70, overlap=10)
    assert len(out) == 2
    first_words = out[0]["text"].split()
    assert out[1]["text"].split()[:10] == first_words[-10:]


def test_long_paragraph_split():
    out = chunk_document(doc(words(450)), max_words=200, overlap=0)
    assert [len(c["text"].split()) for c in out] == [200, 200, 50]


def test_chunk_size_limit():
    text = "\n\n".join(words(n, f"p{n}_") for n in [30, 250, 10, 180, 90])
    out = chunk_document(doc(text), max_words=200, overlap=40)
    assert all(len(c["text"].split()) <= 240 for c in out)
    assert [c["chunk_id"] for c in out] == [f"statement_20260128#{i}" for i in range(len(out))]
    # nothing lost: every word of the input appears somewhere
    all_out = set(w for c in out for w in c["text"].split())
    assert set(text.split()) <= all_out
