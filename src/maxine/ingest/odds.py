"""The Odds API snapshots. Each snapshot is saved raw with a UTC timestamp, plus the credit headers."""
import json
from datetime import datetime, timezone

from maxine.config import ODDS_API_KEY, ODDS_BASE
from maxine.http import get, raw_path

SPORTS = {"nhl": "icehockey_nhl", "ahl": "icehockey_ahl"}
MARKETS = "h2h,spreads,totals"
REGIONS = "us"


def _require_key() -> str:
    if not ODDS_API_KEY:
        raise SystemExit("ODDS_API_KEY is not set. Copy .env.example to .env and add your key.")
    return ODDS_API_KEY


def snapshot(league: str) -> dict:
    """Current odds for one league. Returns {path, events, credits_used_this_call, credits_remaining}."""
    sport = SPORTS[league]
    resp = get(f"{ODDS_BASE}/sports/{sport}/odds", params={
        "apiKey": _require_key(), "regions": REGIONS, "markets": MARKETS,
        "oddsFormat": "decimal", "dateFormat": "iso",
    }, min_interval=1.0)
    taken_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    path = raw_path("odds", league, f"{taken_at}.json")
    payload = {
        "taken_at": taken_at,
        "sport": sport,
        "credits": {
            "last": resp.headers.get("x-requests-last"),
            "used": resp.headers.get("x-requests-used"),
            "remaining": resp.headers.get("x-requests-remaining"),
        },
        "data": resp.json(),
    }
    path.write_text(json.dumps(payload))
    return {"path": str(path), "events": len(payload["data"]),
            "credits_used_this_call": payload["credits"]["last"],
            "credits_remaining": payload["credits"]["remaining"]}


def historical_snapshot(league: str, at_iso: str) -> dict:
    """Historical odds at a UTC timestamp (paid plans; 10 credits per market per region)."""
    sport = SPORTS[league]
    resp = get(f"{ODDS_BASE}/historical/sports/{sport}/odds", params={
        "apiKey": _require_key(), "regions": REGIONS, "markets": MARKETS,
        "oddsFormat": "decimal", "dateFormat": "iso", "date": at_iso,
    }, min_interval=1.0)
    safe = at_iso.replace(":", "-")
    path = raw_path("odds_hist", league, f"{safe}.json")
    body = resp.json()
    path.write_text(json.dumps({"requested_at": at_iso, "credits": {
        "last": resp.headers.get("x-requests-last"),
        "remaining": resp.headers.get("x-requests-remaining")}, **body}))
    return {"path": str(path), "credits_used_this_call": resp.headers.get("x-requests-last"),
            "credits_remaining": resp.headers.get("x-requests-remaining")}
