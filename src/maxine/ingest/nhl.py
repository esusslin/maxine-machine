"""NHL web API ingest (api-web.nhle.com/v1). Raw JSON lands under data/raw/nhl/."""
from datetime import date, timedelta

from maxine.config import NHL_BASE
from maxine.http import fetch_json, raw_path

REGULAR_SEASON = 2
PLAYOFFS = 3
FINAL_STATES = {"OFF", "FINAL"}


def season_meta() -> list[dict]:
    """All seasons with their regular-season start/end dates (standings-season endpoint)."""
    data = fetch_json(f"{NHL_BASE}/standings-season", raw_path("nhl", "standings-season.json"), refresh=True)
    return data["seasons"]


def season_window(season_id: int) -> tuple[date, date]:
    for s in season_meta():
        if s["id"] == season_id:
            return date.fromisoformat(s["standingsStart"]), date.fromisoformat(s["standingsEnd"])
    raise ValueError(f"unknown season {season_id}")


def schedule_week(day: date, refresh: bool = False) -> dict:
    """schedule/{date} returns the 7-day gameWeek starting at that date."""
    return fetch_json(f"{NHL_BASE}/schedule/{day.isoformat()}",
                      raw_path("nhl", "schedule", f"{day.isoformat()}.json"), refresh=refresh)


def games_between(start: date, end: date, refresh: bool = False) -> list[dict]:
    games, day = [], start
    while day <= end:
        week = schedule_week(day, refresh=refresh)
        for gd in week.get("gameWeek", []):
            for g in gd.get("games", []):
                if start.isoformat() <= gd["date"] <= end.isoformat():
                    games.append({**g, "_local_date": gd["date"]})
        day += timedelta(days=7)
    return games


def boxscore(game_id: int, refresh: bool = False) -> dict:
    return fetch_json(f"{NHL_BASE}/gamecenter/{game_id}/boxscore",
                      raw_path("nhl", "boxscore", f"{game_id}.json"), refresh=refresh)


def play_by_play(game_id: int, refresh: bool = False) -> dict:
    return fetch_json(f"{NHL_BASE}/gamecenter/{game_id}/play-by-play",
                      raw_path("nhl", "pbp", f"{game_id}.json"), refresh=refresh)


def standings_on(day: date) -> dict:
    return fetch_json(f"{NHL_BASE}/standings/{day.isoformat()}",
                      raw_path("nhl", "standings", f"{day.isoformat()}.json"))


def ingest_range(start: date, end: date, with_pbp: bool = True, refresh_schedule: bool = False) -> int:
    """Fetch schedule, then boxscore (+ play-by-play) for every finished regular-season/playoff game."""
    count = 0
    for g in games_between(start, end, refresh=refresh_schedule):
        if g.get("gameType") not in (REGULAR_SEASON, PLAYOFFS) or g.get("gameState") not in FINAL_STATES:
            continue
        boxscore(g["id"])
        if with_pbp:
            play_by_play(g["id"])
        count += 1
    return count


def ingest_season(season_id: int, with_pbp: bool = True) -> int:
    start, end = season_window(season_id)
    n = ingest_range(start, end, with_pbp=with_pbp)
    standings_on(end)
    return n
