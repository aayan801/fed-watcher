"""Answer a question from the top-5 retrieved chunks, with a citation for every claim.

Usage:
    python src/answer.py "Why did the Fed cut rates in September 2024?"

The key is read from the project's .env file only (ANTHROPIC_API_KEY=...), never from the shell or chat.
The model sees nothing but the numbered sources, and must say NOT_FOUND when they don't answer.
"""
import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieve import Retriever, load_chunks

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
MODEL = "claude-sonnet-5-5"
PRICE_PER_MTOK = {"input": 2.00, "output": 10.00}  # USD, Claude Sonnet 5.5 (thinking tokens bill as output)
NOT_FOUND = "not found in the documents"

SYSTEM_PROMPT = f"""You answer questions about Federal Reserve (FOMC) policy statements and minutes.

You are given numbered sources. Each one is an excerpt from one FOMC document, labeled with its kind
(statement or minutes) and meeting date.

Rules:
1. Use only the sources. Do not use anything you know about the Fed from elsewhere, even if you are sure
   it is true. If a fact is not in the sources, leave it out.
2. After every sentence that states a fact, cite the source(s) it comes from in exactly this form:
   [minutes 2024-09-18] or [statement 2024-09-18]. Copy the kind and date from the source label.
   Cite several sources by repeating the form: [statement 2024-09-18][minutes 2024-09-18].
3. If the sources do not contain the answer, reply with exactly: {NOT_FOUND}
   Do not add anything else in that case. If they answer only part of the question, answer that part
   with citations and say which part the documents don't cover.
4. Be concise: a few sentences, no headings."""

CITATION = re.compile(r"\[(statement|minutes) (\d{4}-\d{2}-\d{2})\]")

# An LLM is any function (system prompt, user prompt) -> (reply text, usage). Tests pass a fake one.
LLM = Callable[[str, str], tuple[str, dict]]


@dataclass
class Answer:
    question: str
    text: str
    sources: list[dict]
    not_found: bool
    citations: list[tuple[str, str]] = field(default_factory=list)       # (kind, date) pairs used
    bad_citations: list[tuple[str, str]] = field(default_factory=list)   # cited but not among the sources
    usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    blocks = [
        f'<source id="{i}" kind="{c["kind"]}" date="{c["date"]}">\n{c["text"]}\n</source>'
        for i, c in enumerate(chunks, start=1)
    ]
    return "<sources>\n" + "\n".join(blocks) + "\n</sources>\n\nQuestion: " + question


def check_citations(text: str, chunks: list[dict]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (all cited (kind, date) pairs, the ones that match no retrieved chunk)."""
    cited = list(dict.fromkeys(CITATION.findall(text)))
    available = {(c["kind"], c["date"]) for c in chunks}
    return cited, [c for c in cited if c not in available]


def cost(usage: dict) -> float:
    return (usage.get("input_tokens", 0) * PRICE_PER_MTOK["input"]
            + usage.get("output_tokens", 0) * PRICE_PER_MTOK["output"]) / 1_000_000


def answer(question: str, retriever: Retriever, llm: LLM, k: int = 5) -> Answer:
    chunks = retriever.search(question, method="hybrid", k=k)
    if not chunks:
        return Answer(question, NOT_FOUND, [], not_found=True)
    text, usage = llm(SYSTEM_PROMPT, build_user_prompt(question, chunks))
    text = text.strip()
    not_found = text.lower().rstrip(".") == NOT_FOUND
    cited, bad = check_citations(text, chunks)
    return Answer(question, text, chunks, not_found, cited, bad, usage, cost(usage))


# ---- the real model --------------------------------------------------------------------------------

def load_api_key(env_file: Path = ENV_FILE) -> str:
    """Read ANTHROPIC_API_KEY from the .env file only (shell variables and other credentials are ignored)."""
    key = (dotenv_values(env_file).get("ANTHROPIC_API_KEY") or "").strip() if env_file.exists() else ""
    if not key:
        raise SystemExit(
            f"No ANTHROPIC_API_KEY in {env_file}.\n"
            "Create that file with one line:  ANTHROPIC_API_KEY=sk-ant-...  (it is gitignored)."
        )
    return key


def claude_llm(model: str = MODEL, env_file: Path = ENV_FILE) -> LLM:
    import anthropic  # only needed for real calls

    client = anthropic.Anthropic(api_key=load_api_key(env_file))

    def call(system: str, user: str) -> tuple[str, dict]:
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "medium"},
            # if the model declines for policy reasons, the API retries on a default fallback model
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
                 "model": response.model, "stop_reason": response.stop_reason}
        if response.stop_reason == "refusal":
            return "The model declined to answer this question.", usage
        text = "".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason == "max_tokens":
            text += "\n[answer cut off: hit max_tokens]"
        return text, usage

    return call


def print_answer(a: Answer) -> None:
    print(f"Q: {a.question}\n")
    print(a.text)
    print("\nSources:")
    for c in a.sources:
        print(f"  {c['rank']}. {c['kind']} {c['date']}  {c['url']}")
    if a.bad_citations:
        print("WARNING: cited documents that were not among the sources:", a.bad_citations)
    if a.usage:
        print(f"\n{a.usage.get('input_tokens')} input + {a.usage.get('output_tokens')} output tokens, "
              f"${a.cost_usd:.4f}, model {a.usage.get('model')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("question")
    args = ap.parse_args()
    llm = claude_llm()  # fails fast, before loading the index, if .env has no key
    print_answer(answer(args.question, Retriever(load_chunks()), llm))


if __name__ == "__main__":
    main()
