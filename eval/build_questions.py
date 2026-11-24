"""Build eval/questions.json: 40 questions, each with the doc_ids that answer it.

    python eval/build_questions.py      # rewrites eval/questions.json (hand edits there are lost)

Relevant docs are not picked by hand: a question's relevant docs are the docs whose text matches its
`check` pattern (optionally limited to some meeting dates). Date-only questions use the meeting's own
statement and minutes. `evidence` in the output shows the matched passage, so a person can verify it.
"""
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
D = {}
ROOT = Path(__file__).resolve().parent.parent
for p in sorted(glob.glob(str(ROOT / "data" / "raw" / "*.json"))):
    d = json.load(open(p, encoding="utf-8"))
    D[d["doc_id"]] = d["text"]

# (type, question, check regex or None, meeting dates or None)
Q = [
    # ---- keyword: the question uses the document's own words
    ("keyword", "What did the Fed say about Silicon Valley Bank?", r"Silicon Valley Bank", None),
    ("keyword", "When did the Committee begin reducing net asset purchases by $10 billion for Treasury securities and $5 billion for agency mortgage-backed securities?", r"\$10 billion for Treasury securities and \$5 billion for agency mortgage-backed", None),
    ("keyword", "When did the Committee decide to conclude the reduction of its aggregate securities holdings?", r"conclude the reduction of its aggregate securities holdings", None),
    ("keyword", "When was the monthly redemption cap on Treasury securities reduced from $25 billion to $5 billion?", r"from \$25 billion to \$5 billion", None),
    ("keyword", "What did the March 2022 statement say about the invasion of Ukraine by Russia?", r"invasion of Ukraine", ["20220316"]),
    ("keyword", "How did the Omicron variant affect the outlook in December 2021?", r"Omicron", ["20211215"]),
    ("keyword", "What did the staff say about debt associated with the artificial intelligence buildout?", r"artificial intelligence \(AI\) buildout", None),
    ("keyword", "When did the Committee decide to begin reducing its holdings of Treasury securities and agency mortgage-backed securities on June 1?", r"begin reducing its holdings of Treasury securities and agency debt and agency mortgage-backed securities on June 1", None),
    ("keyword", "Why did James Bullard dissent in March 2022?", r"Voting against this action ?:? (was )?James Bullard", None),
    ("keyword", "When did the Committee decide to initiate purchases of shorter-term Treasury securities to maintain ample reserves?", r"initiate purchases of shorter-term Treasury securities", None),
    ("keyword", "How did the resolution of the debt limit affect Treasury bill supply in 2023?", r"resolution of the debt limit", ["2023"]),
    ("keyword", "How did the March 2026 statement describe the implications of developments in the Middle East?", r"implications of developments in the Middle East", ["20260318"]),
    # ---- paraphrase: same facts, different words
    ("paraphrase", "When did the Fed start slowing its bond buying in 2021?", r"begin reducing the monthly pace of its net asset purchases", None),
    ("paraphrase", "When did the Fed decide to stop shrinking its balance sheet?", r"conclude the reduction of its aggregate securities holdings", None),
    ("paraphrase", "How did the collapse of two regional lenders in early 2023 affect credit markets?", r"bank closures", ["20230322"]),
    ("paraphrase", "Did anyone want a smaller rate cut than half a point in September 2024?", r"Voting against this action ?:? (was )?Michelle W\. Bowman", ["20240918"]),
    ("paraphrase", "Who wanted to hold rates steady instead of cutting in December 2024?", r"Voting against (the|this) action ?:? (was )?Beth M\. Hammack", ["20241218"]),
    ("paraphrase", "Did businesses plan to pass higher import taxes on to customers in early 2025?", r"pass on to consumers higher input costs arising from potential tariffs", None),
    ("paraphrase", "Did policymakers worry that a lapse in federal funding would hold up economic statistics in fall 2023?", r"government shutdown might result in the delayed release", None),
    ("paraphrase", "When did officials first discuss letting the balance sheet shrink after the first rate hike?", r"initiate balance sheet runoff at some point after the first increase", None),
    ("paraphrase", "Did officials expect the 2021 rise in inflation to be temporary?", r"transitory", ["20210317", "20210428", "20210616", "20210728", "20210922", "20211103", "20211215"]),
    ("paraphrase", "Could new technology like AI raise how much workers produce, according to participants in spring 2024?", r"technologies such as artificial intelligence into existing business operations", None),
    ("paraphrase", "When did the Fed set up permanent backstop facilities for short-term secured lending markets?", r"Establishment of Standing Repurchase Agreement Facilities", None),
    ("paraphrase", "Did all participants support the Fed's long-run strategy document as written at the start of 2021?", r"all participants supported the statement as written", ["20210127"]),
    # ---- fraction: the question writes the rate differently from the document
    ("fraction", "Why did members keep the target range at 5-1/4 to 5-1/2 percent in November 2023?", r"5¼ to 5½|5-1/4 to 5-1/2", ["20231101"]),
    ("fraction", "Why did the Committee lower rates by 25 basis points to 4-1/4 to 4-1/2 percent in December 2024?", r"4¼ to 4½|4-1/4 to 4-1/2", ["20241218"]),
    ("fraction", "When did the Committee raise the target range by ¼ percentage point to 3¾ to 4 percent?", r"raise the target range for the federal funds rate by 1/4 percentage point to 3-3/4 to 4 percent", None),
    ("fraction", "Which meeting kept the target range at 5 to 5¼ percent after a long series of hikes?", r"maintain the target range for the federal funds rate at 5 to 5-1/4 percent", None),
    ("fraction", "When did the Fed raise rates to 1½ to 1¾ percent?", r"raise the target range for the federal funds rate to 1-1/2 to 1-3/4 percent", None),
    # ---- dash: hyphen in the question, en dash in the document
    ("dash", "Who voted against the 9-3 decision in July 2026?", r"9 ?– ?3 vote", None),
    ("dash", "Which meeting approved its statement by a 10-2 vote?", r"10 ?– ?2 vote", None),
    ("dash", "What did the Committee decide at its July 28-29, 2026 meeting?", None, ["20260729"]),
    ("dash", "Summarize the September 15-16, 2026 FOMC meeting.", None, ["20260916"]),
    ("dash", "What was decided at the May 2-3, 2023 meeting?", None, ["20230503"]),
    # ---- accent: names typed without accents
    ("accent", "Which meeting did Enrique Martinez Garcia of the Dallas Fed attend?", r"Martínez García", None),
    ("accent", "Which FOMC meetings did Egon Zakrajsek attend?", r"Zakrajšek", None),
    ("accent", "Which meetings list Isabel Cairo of the Division of Monetary Affairs?", r"Cairó", None),
    ("accent", "Which meetings did Giovanni Nicolo attend?", r"Nicolò", None),
    # ---- date: the answer depends on the meeting date, which chunk text often lacks
    ("date", "Who dissented at the October 2026 meeting and why?", None, ["20261028"]),
    ("date", "What did the FOMC decide at its January 2021 meeting?", None, ["20210127"]),
]

out, problems = [], []
for i, (kind, q, check, dates) in enumerate(Q, 1):
    ids = [k for k in D if dates is None or any(k[-8:].startswith(d) for d in dates)]
    rel = [k for k in ids if check is None or re.search(check, D[k])]
    if check and rel:
        m = re.search(check, D[rel[0]])
        ev = re.sub(r"\s+", " ", D[rel[0]][max(0, m.start() - 60):m.end() + 60]).strip()
    else:
        ev = "the meeting's own statement and minutes" if not check else ""
    if not rel:
        problems.append(f"q{i}: no relevant docs")
    if len(rel) > 12:
        problems.append(f"q{i}: {len(rel)} relevant docs (too easy?)")
    out.append({"id": f"q{i:02d}", "type": kind, "question": q, "relevant": rel, "evidence": ev})

assert len(out) == 40, len(out)
(ROOT / "eval" / "questions.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("problems:", problems or "none")
print(Counter(q["type"] for q in out))
for q in out:
    print(f'{q["id"]} [{q["type"]}] {q["question"]}\n     -> {", ".join(q["relevant"])}\n     ev: {q["evidence"][:150]}')
