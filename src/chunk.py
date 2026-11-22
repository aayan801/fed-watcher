"""Split documents into overlapping chunks for retrieval.

YOU write chunk_document(). The tests in tests/test_chunk.py define what "correct" means:
    pytest -q tests/test_chunk.py

Why chunking matters: the retriever compares a question with each chunk. Chunks that are too big
mix several topics (blurry matches); chunks that are too small lose context (the answer gets cut
in half). Overlap keeps a sentence that sits on a boundary findable from both sides.
"""


def chunk_document(doc: dict, max_words: int = 200, overlap: int = 40) -> list[dict]:
    """Split doc["text"] into chunks of at most `max_words` words.

    Rules (the tests check each one):
      1. Paragraphs are separated by a blank line ("\\n\\n"). Keep whole paragraphs together:
         keep adding paragraphs to the current chunk while it stays <= max_words.
      2. If ONE paragraph is longer than max_words, split it into pieces of max_words words.
      3. Every chunk after the first starts with the last `overlap` words of the previous chunk
         (so a chunk can be up to max_words + overlap words long).
      4. Return a list of dicts, in order:
             {"chunk_id": f"{doc['doc_id']}#{i}",   # i = 0, 1, 2, ...
              "doc_id": doc["doc_id"], "kind": doc["kind"], "date": doc["date"], "url": doc["url"],
              "text": <chunk text, words joined by single spaces>}
      5. An empty or whitespace-only text returns [].
    """
    raise NotImplementedError("your turn")


def chunk_all(docs: list[dict], **kw) -> list[dict]:
    return [c for d in docs for c in chunk_document(d, **kw)]
