"""BetNHL free daily tips: save the raw page every day, extract picks best-effort.

The raw HTML is the source of truth; the extractor can be improved and re-run later.
"""
import json
import re
from datetime import date

from bs4 import BeautifulSoup

from maxine.config import BETNHL_BASE
from maxine.http import get, raw_path


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def tip_url(day: date) -> str:
    month = day.strftime("%B").lower()
    return f"{BETNHL_BASE}/tonights-tips/{month}-{_ordinal(day.day)}-{day.year}-nhl-betting-tips.php"


PICK_RE = re.compile(r"\b(?:our\s+)?pick\s*[:\-]\s*(?P<pick>[^\n]+)", re.IGNORECASE)
DECIMAL_RE = re.compile(r"\b([1-9]\.\d{2})\b")
AMERICAN_RE = re.compile(r"(?<![\d.])([+-]\d{3,4})\b")


def extract_picks(html: str) -> list[dict]:
    """Find 'Pick: ...' lines and the first odds quoted after each. Version 0: verify against raw pages."""
    text = BeautifulSoup(html, "html.parser").get_text("\n")
    picks = []
    matches = list(PICK_RE.finditer(text))
    for i, m in enumerate(matches):
        window_end = matches[i + 1].start() if i + 1 < len(matches) else m.end() + 600
        window = text[m.end():window_end]
        dec = DECIMAL_RE.search(window)
        amer = AMERICAN_RE.search(window)
        picks.append({
            "pick_text": m.group("pick").strip(),
            "first_decimal_odds": float(dec.group(1)) if dec else None,
            "first_american_odds": int(amer.group(1)) if amer else None,
        })
    return picks


def snapshot(day: date) -> dict:
    url = tip_url(day)
    resp = get(url, min_interval=2.0)
    html_path = raw_path("betnhl", f"{day.isoformat()}.html")
    html_path.write_text(resp.text)
    picks = extract_picks(resp.text)
    raw_path("betnhl", f"{day.isoformat()}.picks.json").write_text(
        json.dumps({"date": day.isoformat(), "url": url, "extractor": "v0", "picks": picks}, indent=2))
    return {"url": url, "picks": len(picks), "html": str(html_path)}
