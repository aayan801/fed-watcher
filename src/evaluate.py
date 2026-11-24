"""Measure retrieval quality: top-k hit rate for BM25, embeddings and hybrid on eval/questions.json.

Usage:
    python src/evaluate.py            # prints the tables and writes eval/results.json

Hit: at least one of the top-k chunks comes from a document listed as relevant for that question.
Hit rate: the share of questions that are hits (1.0 = every question found a relevant document).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieve import METHODS, Retriever, load_chunks

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.json"
RESULTS = ROOT / "eval" / "results.json"


def is_hit(retrieved_doc_ids: list[str], relevant: list[str]) -> bool:
    """True if any retrieved chunk belongs to a relevant document."""
    return any(d in relevant for d in retrieved_doc_ids)


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
            docs = [h["doc_id"] for h in retriever.search(q["question"], method=m, k=k)]
            row[m] = {"hit": is_hit(docs, q["relevant"]), "retrieved": docs}
        rows.append(row)

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    types = list(dict.fromkeys(q["type"] for q in questions))  # keep first-seen order
    return {
        "k": k,
        "n_questions": len(rows),
        "overall": {m: hit_rate([r[m]["hit"] for r in rows]) for m in methods},
        "by_type": {t: {"n": len(by_type[t]), **{m: hit_rate([r[m]["hit"] for r in by_type[t]]) for m in methods}}
                    for t in types},
        "questions": rows,
    }


def format_tables(res: dict, methods=METHODS) -> str:
    head = f"{'':12s}{'n':>4s}" + "".join(f"{m:>12s}" for m in methods)
    lines = [f"Top-{res['k']} hit rate, {res['n_questions']} questions", "", head, "-" * len(head)]
    lines.append(f"{'overall':12s}{res['n_questions']:4d}" + "".join(f"{res['overall'][m]:12.0%}" for m in methods))
    lines += ["", "By question type:", head, "-" * len(head)]
    for t, v in res["by_type"].items():
        lines.append(f"{t:12s}{v['n']:4d}" + "".join(f"{v[m]:12.0%}" for m in methods))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    res = evaluate(Retriever(load_chunks()), questions, k=args.k)
    RESULTS.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(format_tables(res))
    misses = [r for r in res["questions"] if not r["hybrid"]["hit"]]
    print(f"\nHybrid misses ({len(misses)}):")
    for r in misses:
        print(f"  {r['id']} [{r['type']}] {r['question']}")


if __name__ == "__main__":
    main()
