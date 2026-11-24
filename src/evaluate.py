"""Measure retrieval quality: top-k hit rate for BM25, embeddings and hybrid on eval/questions.json.

Usage:
    python src/evaluate.py            # prints the tables and writes eval/results.json

Document-level hit: at least one of the top-k chunks comes from a document listed as relevant.
Chunk-level hit: at least one of the top-k chunks actually CONTAINS the answer: it comes from a relevant
document and its text matches the question's `answer_pattern`. Stricter, and closer to what the answer
step needs: the model only sees the chunks, so the right document with the wrong chunk is still a miss.
Hit rate: the share of questions that are hits.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieve import BM25_WEIGHT, DATE_HEADER, METHODS, MODEL_NAME, Retriever, load_chunks

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.json"
RESULTS = ROOT / "eval" / "results.json"


def is_hit(retrieved_doc_ids: list[str], relevant: list[str]) -> bool:
    """True if any retrieved chunk belongs to a relevant document."""
    return any(d in relevant for d in retrieved_doc_ids)


def is_chunk_hit(retrieved: list[dict], relevant: list[str], answer_pattern: str | None) -> bool:
    """True if any retrieved chunk is from a relevant document AND its text contains the answer."""
    if not answer_pattern:
        return False
    pat = re.compile(answer_pattern)
    return any(h["doc_id"] in relevant and pat.search(h.get("text", "")) for h in retrieved)


def hit_rate(hits: list[bool]) -> float:
    if not hits:
        raise ValueError("no questions to score")
    return sum(hits) / len(hits)


def evaluate(retriever: Retriever, questions: list[dict], methods=METHODS, k: int = 5) -> dict:
    """Run every question through every method. Returns per-question rows plus overall and per-type rates."""
    rows = []
    for q in questions:
        row = {"id": q["id"], "type": q["type"], "question": q["question"], "relevant": q["relevant"]}
        for m in methods:
            hits = retriever.search(q["question"], method=m, k=k)
            docs = [h["doc_id"] for h in hits]
            row[m] = {"hit": is_hit(docs, q["relevant"]),
                      "chunk_hit": is_chunk_hit(hits, q["relevant"], q.get("answer_pattern")),
                      "retrieved": [h.get("chunk_id", h["doc_id"]) for h in hits]}
        rows.append(row)

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    types = list(dict.fromkeys(q["type"] for q in questions))  # keep first-seen order

    def rates(level: str) -> tuple[dict, dict]:
        overall = {m: hit_rate([r[m][level] for r in rows]) for m in methods}
        per_type = {t: {"n": len(by_type[t]), **{m: hit_rate([r[m][level] for r in by_type[t]]) for m in methods}}
                    for t in types}
        return overall, per_type

    overall, per_type = rates("hit")
    overall_chunk, per_type_chunk = rates("chunk_hit")
    return {
        "k": k,
        "n_questions": len(rows),
        "overall": overall, "by_type": per_type,                       # document level
        "overall_chunk": overall_chunk, "by_type_chunk": per_type_chunk,  # chunk level
        "questions": rows,
    }


def format_tables(res: dict, methods=METHODS) -> str:
    head = f"{'':12s}{'n':>4s}" + "".join(f"{m:>12s}" for m in methods)
    lines = [f"Top-{res['k']} hit rate, {res['n_questions']} questions"]
    for title, overall, per_type in [("Document level (a chunk from a right document)", res["overall"], res["by_type"]),
                                     ("Chunk level (a chunk containing the answer)", res.get("overall_chunk"),
                                      res.get("by_type_chunk"))]:
        if overall is None:
            continue
        lines += ["", title, head, "-" * len(head)]
        lines.append(f"{'overall':12s}{res['n_questions']:4d}" + "".join(f"{overall[m]:12.0%}" for m in methods))
        for t, v in per_type.items():
            lines.append(f"{t:12s}{v['n']:4d}" + "".join(f"{v[m]:12.0%}" for m in methods))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--model", default=MODEL_NAME, help="embedding model (sentence-transformers name)")
    ap.add_argument("--bm25-weight", type=float, default=BM25_WEIGHT, help="hybrid: BM25 weight vs embeddings (1.0)")
    ap.add_argument("--date-header", action=argparse.BooleanOptionalAction, default=DATE_HEADER,
                    help='search "FOMC <kind>, <date>: <text>" instead of the bare chunk text')
    args = ap.parse_args()
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    retriever = Retriever(load_chunks(), model_name=args.model, bm25_weight=args.bm25_weight,
                          date_header=args.date_header)
    settings = {"model": retriever.model_name, "bm25_weight": retriever.bm25_weight,
                "date_header": retriever.date_header}
    res = {"settings": settings, **evaluate(retriever, questions, k=args.k)}
    print("settings:", res["settings"])
    RESULTS.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(format_tables(res))
    misses = [r for r in res["questions"] if not r["hybrid"]["chunk_hit"]]
    print(f"\nHybrid chunk-level misses ({len(misses)}; * = right document found, wrong chunk):")
    for r in misses:
        mark = "*" if r["hybrid"]["hit"] else " "
        print(f"  {mark} {r['id']} [{r['type']}] {r['question']}")


if __name__ == "__main__":
    main()
