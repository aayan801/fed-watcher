"""Download FOMC policy statements and minutes (2021 onwards) from federalreserve.gov.

Usage:
    python src/download.py            # saves data/raw/<kind>_<YYYYMMDD>.json

Each file holds {"doc_id", "kind" ("statement" | "minutes"), "date" (YYYY-MM-DD), "url", "text"}.
"""
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://www.federalreserve.gov"
CALENDAR = BASE + "/monetarypolicy/fomccalendars.htm"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw"
HEADERS = {"User-Agent": "fed-watcher student project (contact: ae2417@nyu.edu)"}

# lines that are website boilerplate, not policy content
BOILERPLATE = re.compile(
    r"^(For release at|For media inquiries|Implementation Note issued|Last Update|Back to Top|Share$)",
    re.IGNORECASE,
)

# link back from each footnote in the minutes
RETURN_LINK = re.compile(r"\s*Return to text$")

PATTERNS = {
    # policy statements are linked as "HTML"; other releases (e.g. the 2025-08-22 framework review) by title
    "statement": re.compile(r'/newsevents/pressreleases/monetary(\d{8})a\.htm(?=">HTML<)'),
    "minutes": re.compile(r"/monetarypolicy/fomcminutes(\d{8})\.htm"),
}


def find_links(html: str) -> list[tuple[str, str, str]]:
    """Return sorted unique (kind, yyyymmdd, absolute_url) found in the calendar page."""
    found = set()
    for kind, pat in PATTERNS.items():
        for m in pat.finditer(html):
            found.add((kind, m.group(1), BASE + m.group(0)))
    return sorted(found, key=lambda x: (x[1], x[0]))


def extract_text(html: str) -> str:
    """Pull the body text of a Fed press release / minutes page, one paragraph per line."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find(id="article") or soup.find("main") or soup.body
    for tag in body.find_all(["script", "style", "nav", "footer"]):
        tag.decompose()
    paras = [p.get_text(" ", strip=True) for p in body.find_all(["p", "li"])]
    paras = [re.sub(r"\s+", " ", p) for p in paras]
    paras = [p for p in paras if len(p.split()) >= 5 and not BOILERPLATE.match(p)]
    # clean after the word-count filter, otherwise short footnotes would be dropped
    paras = [RETURN_LINK.sub("", p.replace("\N{SOFT HYPHEN}", "")) for p in paras]
    return "\n\n".join(paras)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cal = requests.get(CALENDAR, headers=HEADERS, timeout=30)
    cal.raise_for_status()
    cal.encoding = "utf-8"
    links = find_links(cal.text)
    print(f"found {len(links)} documents")
    for kind, d, url in links:
        path = OUT / f"{kind}_{d}.json"
        if path.exists():
            continue
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.encoding = "utf-8"  # the Fed serves UTF-8; without this, dashes turn into garbage
        if r.status_code != 200:
            print(f"skip {url} ({r.status_code})")
            continue
        text = extract_text(r.text)
        doc = {"doc_id": f"{kind}_{d}", "kind": kind, "date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "url": url, "text": text}
        path.write_text(json.dumps(doc, indent=1))
        print(f"saved {path.name}: {len(text.split())} words")
        time.sleep(1)  # be polite to the Fed's server


if __name__ == "__main__":
    main()
