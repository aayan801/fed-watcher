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
    paragraphs = [p.split() for p in doc["text"].split("\n\n") if p.strip()]   # NEW
    if not paragraphs:                                                          # NEW
        return []

    # A paragraph longer than max_words is cut into max_words-sized pieces;
    # after that, every piece fits in a chunk on its own.
    pieces = []
    for words in paragraphs:
        for start in range(0, len(words), max_words):
            pieces.append(words[start:start + max_words])

    # Greedy packing: add pieces to the current chunk until the next one would not fit.
    chunks, current = [], []
    for piece in pieces:
        if current and len(current) + len(piece) > max_words:
            chunks.append(current)
            current = []
        current = current + piece
    if current:
        chunks.append(current)  # don't lose the last chunk

    # Overlap: each chunk after the first starts with the last `overlap` words of the previous one.
    # (Guard overlap > 0: words[-0:] would return the WHOLE list, not an empty one.)
    texts = [chunks[0]]
    for prev, chunk in zip(chunks, chunks[1:]):
        carried = prev[-overlap:] if overlap > 0 else []
        texts.append(carried + chunk)

    return [
        {"chunk_id": f"{doc['doc_id']}#{i}", "doc_id": doc["doc_id"], "kind": doc["kind"],
         "date": doc["date"], "url": doc["url"], "text": " ".join(words)}
        for i, words in enumerate(texts)
    ]


def chunk_all(docs: list[dict], **kw) -> list[dict]:
    return [c for d in docs for c in chunk_document(d, **kw)]
