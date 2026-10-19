"""Historical odds backfill from The Odds API, driven by NHL game start times in the warehouse.

Per game day: one morning snapshot (8:00 PT = 15:00 UTC) plus one snapshot 10 minutes before
each distinct puck-drop time. Newest season first. Stops before credits fall below a reserve.
"""
from datetime import datetime, timedelta, timezone

from maxine.http import raw_path
from maxine.ingest import odds

CREDITS_PER_SNAPSHOT = 30          # 3 markets x 1 region x 10 credits (historical pricing)
MORNING_UTC_HOUR = 15              # 8:00 PT during the season (PDT); 7:00 PT in PST, close enough
BEFORE_PUCK = timedelta(minutes=10)


def planned_snapshots(con, seasons: list[int]) -> list[str]:
    """ISO UTC timestamps to request, newest first."""
    rows = con.execute("""
        SELECT DISTINCT game_date, start_utc FROM games
        WHERE league = 'NHL' AND game_type = 2 AND season IN (SELECT UNNEST(?)) AND start_utc IS NOT NULL
    """, [seasons]).fetchall()
    stamps: set[datetime] = set()
    for game_date, start_utc in rows:
        morning = datetime(game_date.year, game_date.month, game_date.day, MORNING_UTC_HOUR, tzinfo=timezone.utc)
        stamps.add(morning)
        start = start_utc.replace(tzinfo=timezone.utc) if start_utc.tzinfo is None else start_utc
        stamps.add(start - BEFORE_PUCK)
    return [s.strftime("%Y-%m-%dT%H:%M:%SZ") for s in sorted(stamps, reverse=True)]


def already_saved(at_iso: str, league: str = "nhl") -> bool:  # league = feed name
    return raw_path("odds_hist", league, f"{at_iso.replace(':', '-')}.json").exists()


def run(con, seasons: list[int], dry_run: bool, reserve: int, feed: str = "nhl") -> dict:
    per = odds.credits_per_historical(feed)
    plan = planned_snapshots(con, seasons)
    todo = [s for s in plan if not already_saved(s, feed)]
    summary = {"feed": feed, "seasons": seasons, "snapshots_planned": len(plan), "already_saved": len(plan) - len(todo),
               "to_fetch": len(todo), "credits_per_snapshot": per, "estimated_credits": len(todo) * per}
    if dry_run:
        return summary
    fetched, remaining = 0, None
    for at_iso in todo:
        if remaining is not None and remaining - per < reserve:
            summary["stopped"] = f"credits remaining {remaining} would drop below reserve {reserve}"
            break
        result = odds.historical_snapshot(feed, at_iso)
        fetched += 1
        remaining = int(result["credits_remaining"]) if result["credits_remaining"] else None
        if fetched == 1 or fetched % 50 == 0:
            print(f"  {fetched}/{len(todo)} {at_iso} cost={result['credits_used_this_call']} remaining={remaining}")
    summary.update({"fetched": fetched, "credits_remaining": remaining})
    return summary
