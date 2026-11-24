"""Find the chunks most relevant to a question: BM25, embeddings, or a hybrid of both.

Usage:
    python src/retrieve.py "Why did the Fed pause in September 2026?" --method hybrid

BM25        keyword match: scores chunks that share the question's words, rare words count more.
embeddings  meaning match: a small neural model turns text into vectors; similar meaning -> close vectors.
            The vectors live in a faiss index, which finds the nearest ones to the question's vector.
hybrid      merge the two rankings with reciprocal rank fusion (RRF).

Questions and chunks go through the same normalize() before BM25 and before the embedding model.
"""
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Callable

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.chunk import chunk_all

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
METHODS = ("bm25", "embeddings", "hybrid")
RRF_K = 60  # standard constant from the original RRF paper; damps the gap between rank 1 and rank 2

Encoder = Callable[[list[str]], np.ndarray]

FRACTIONS = {"¼": "1/4", "½": "1/2", "¾": "3/4"}
# en dash, em dash, minus sign, figure dash, hyphen: people type all of them as "-"
DASHES = str.maketrans({c: "-" for c in "–—−‒‐"})


def normalize(text: str) -> str:
    """Make different spellings of the same thing identical. Used on chunks AND questions."""
    # "5¼" -> "5-1/4" (how the Fed usually writes it); a lone "¼" -> "1/4"
    text = re.sub(r"(\d?)([¼½¾])", lambda m: (m[1] + "-" if m[1] else "") + FRACTIONS[m[2]], text)
    text = text.translate(DASHES)  # "May 2–3" -> "May 2-3"
    # accents: split "é" into "e" + accent mark, then drop the mark
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    """Lowercase words for BM25. \\w+ splits on dashes and slashes: "12-month" -> "12", "month"."""
    return re.findall(r"\w+", normalize(text).lower())


def load_chunks(raw_dir: Path = RAW, **chunk_kw) -> list[dict]:
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(raw_dir.glob("*.json"))]
    if not docs:
        raise SystemExit(f"no documents in {raw_dir}; run python src/download.py first")
    return chunk_all(docs, **chunk_kw)


def sentence_transformer_encoder(model_name: str = MODEL_NAME) -> Encoder:
    from sentence_transformers import SentenceTransformer  # slow import, so only when needed

    model = SentenceTransformer(model_name)
    # normalized vectors: their dot product IS the cosine similarity
    return lambda texts: model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal rank fusion: each list gives a chunk 1 / (k + rank). Highest total wins."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))


class Retriever:
    def __init__(self, chunks: list[dict], encoder: Encoder | None = None, cache_dir: Path | None = CACHE):
        if not chunks:
            raise ValueError("no chunks to search")
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        self._encoder = encoder
        self._cache_dir = cache_dir
        self._index: faiss.Index | None = None  # built on first embeddings search

    # ---- embeddings ---------------------------------------------------------------------------
    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._encoder is None:
            self._encoder = sentence_transformer_encoder()
        vectors = np.asarray(self._encoder([normalize(t) for t in texts]), dtype=np.float32)
        return np.ascontiguousarray(vectors)  # faiss needs float32, contiguous

    def _faiss_index(self) -> faiss.Index:
        if self._index is not None:
            return self._index
        # encoding 2,000+ chunks takes minutes on CPU, so cache the vectors;
        # the key changes if the model, normalize() output or any chunk changes
        texts = [normalize(c["text"]) for c in self.chunks]
        material = MODEL_NAME + "\n" + "\n".join(c["chunk_id"] + t for c, t in zip(self.chunks, texts))
        key = hashlib.sha256(material.encode()).hexdigest()[:16]
        path = self._cache_dir / f"embeddings_{key}.npy" if self._cache_dir else None
        if path and path.exists():
            vectors = np.load(path)
        else:
            vectors = self._encode([c["text"] for c in self.chunks])
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, vectors)
        # IndexFlatIP = exact search by inner product; on normalized vectors that's cosine similarity
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        return self._index

    # ---- rankings (lists of chunk indexes, best first) ------------------------------------------
    def rank_bm25(self, question: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(question))
        order = np.argsort(-scores, kind="stable")
        return [int(i) for i in order[:n] if scores[i] > 0]  # no shared words -> not a match

    def rank_embeddings(self, question: str, n: int) -> list[int]:
        index = self._faiss_index()
        _, ids = index.search(self._encode([question]), min(n, index.ntotal))
        return [int(i) for i in ids[0] if i >= 0]  # faiss pads with -1 when it has fewer results

    def search(self, question: str, method: str = "hybrid", k: int = 5) -> list[dict]:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if method == "bm25":
            order = self.rank_bm25(question, k)
        elif method == "embeddings":
            order = self.rank_embeddings(question, k)
        else:
            # fuse deeper lists than k, so a chunk ranked 8th by both methods can still make the top 5
            depth = max(50, k)
            order = rrf([self.rank_bm25(question, depth), self.rank_embeddings(question, depth)])[:k]
        return [dict(self.chunks[i], rank=r) for r, i in enumerate(order, start=1)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("question")
    ap.add_argument("--method", choices=METHODS, default="hybrid")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    for hit in Retriever(load_chunks()).search(args.question, args.method, args.k):
        print(f"\n#{hit['rank']} {hit['chunk_id']}  {hit['date']}  {hit['url']}")
        print("   " + hit["text"][:300] + ("..." if len(hit["text"]) > 300 else ""))


if __name__ == "__main__":
    main()
