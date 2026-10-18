"""Parse raw JSON/HTML into the warehouse.

Idempotent: rows for a key are replaced, never duplicated.
Incremental: raw files already loaded (same size and mtime) are skipped; `maxine load --full` reloads all.
Bulk: rows are inserted in batches through Arrow, not one at a time.
"""
import json
from pathlib import Path

import pyarrow as pa

from maxine.config import RAW_DIR

SHOT_EVENTS = {"shot-on-goal", "goal", "missed-shot", "blocked-shot"}


def _toi_seconds(toi: str | None) -> int | None:
    if not toi or ":" not in toi:
        return None
    m, s = toi.split(":")
    return int(m) * 60 + int(s)


def _split_shots(value: str | None) -> tuple[int | None, int | None]:
    """'23/25' -> (saves=23, shots_against=25)."""
    if not value or "/" not in value:
        return None, None
    saves, sa = value.split("/")
    return int(saves), int(sa)


def parse_boxscore(box: dict) -> tuple[tuple, list[tuple], list[tuple]]:
    home, away = box["homeTeam"], box["awayTeam"]
    game = (
        box["id"], "NHL", box["season"], box["gameType"], box["gameDate"], box.get("startTimeUTC"),
        home["abbrev"], away["abbrev"], home.get("id"), away.get("id"),
        home.get("score"), away.get("score"),
        (box.get("gameOutcome") or {}).get("lastPeriodType"),
        (box.get("venue") or {}).get("default"),
    )
    teams = [
        (box["id"], home["abbrev"], True, home.get("score"), home.get("sog")),
        (box["id"], away["abbrev"], False, away.get("score"), away.get("sog")),
    ]
    goalies = []
    pbg = box.get("playerByGameStats") or {}
    for side, team in (("homeTeam", home), ("awayTeam", away)):
        for g in (pbg.get(side) or {}).get("goalies", []):
            saves, sa = _split_shots(g.get("saveShotsAgainst"))
            goalies.append((
                box["id"], team["abbrev"], g["playerId"], (g.get("name") or {}).get("default"),
                g.get("starter"), _toi_seconds(g.get("toi")), sa, saves, g.get("goalsAgainst"), g.get("decision"),
            ))
    return game, teams, goalies


def parse_pbp(pbp: dict) -> list[tuple]:
    rows = []
    for p in pbp.get("plays", []):
        if p.get("typeDescKey") not in SHOT_EVENTS:
            continue
        d = p.get("details") or {}
        rows.append((
            pbp["id"], p["eventId"], (p.get("periodDescriptor") or {}).get("number"), p.get("timeInPeriod"),
            p.get("situationCode"), p["typeDescKey"], d.get("eventOwnerTeamId"),
            d.get("shootingPlayerId") or d.get("scoringPlayerId"), d.get("goalieInNetId"),
            d.get("shotType"), d.get("xCoord"), d.get("yCoord"),
        ))
    return rows


def ot_empty_net_loss(pbp: dict) -> bool:
    """True when the deciding goal came in overtime into an empty net.

    NHL rule: a team that pulls its goalie in OT and loses gets no point (a regulation-style loss).
    """
    goals = [p for p in pbp.get("plays", []) if p.get("typeDescKey") == "goal"]
    if not goals:
        return False
    last = goals[-1]
    in_ot = (last.get("periodDescriptor") or {}).get("periodType") == "OT"
    return in_ot and (last.get("details") or {}).get("goalieInNetId") is None


GAME_COLS = ["game_id", "league", "season", "game_type", "game_date", "start_utc", "home", "away",
             "home_team_id", "away_team_id", "home_score", "away_score", "last_period_type", "venue",
             "ot_empty_net_loss"]
TEAM_COLS = ["game_id", "team", "is_home", "goals", "sog"]
GOALIE_COLS = ["game_id", "team", "player_id", "name", "starter", "toi_seconds", "shots_against", "saves",
               "goals_against", "decision"]
SHOT_COLS = ["game_id", "event_id", "period", "time_in_period", "situation_code", "event_type", "owner_team_id",
             "shooter_id", "goalie_id", "shot_type", "x", "y"]
ODDS_COLS = ["taken_at", "league", "event_id", "commence_time", "home", "away", "book", "market", "outcome",
             "price", "point"]
ODDS_KEY = "taken_at, league, event_id, book, market, outcome"
BATCH_FILES = 500


# ---------- helpers ----------

def _signature(*paths: Path) -> str:
    parts = []
    for path in paths:
        if path.exists():
            st = path.stat()
            parts.append(f"{st.st_size}:{st.st_mtime_ns}")
        else:
            parts.append("-")
    return "|".join(parts)


def _pending(con, kind: str, items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """items = [(key, signature)]; return those not loaded with the same signature."""
    done = dict(con.execute("SELECT key, signature FROM loaded_files WHERE kind = ?", [kind]).fetchall())
    return [(k, sig) for k, sig in items if done.get(k) != sig]


def _mark_loaded(con, kind: str, items: list[tuple[str, str]]) -> None:
    if not items:
        return
    _bulk(con, "loaded_files", ["kind", "key", "signature"], [(kind, k, sig) for k, sig in items],
          replace_where="kind = ? AND key IN (SELECT key FROM _bulk)", replace_params=[kind])


def _bulk(con, table: str, cols: list[str], rows: list[tuple], replace_where: str | None = None,
          replace_params: list | None = None, distinct_on: str | None = None) -> int:
    """Insert rows via an Arrow table. Optionally delete matching rows first (replace_where may use _bulk)."""
    if not rows:
        return 0
    arrow = pa.table({c: pa.array(list(v)) for c, v in zip(cols, zip(*rows))})
    con.register("_bulk", arrow)
    try:
        if replace_where:
            con.execute(f"DELETE FROM {table} WHERE {replace_where}", replace_params or [])
        col_list = ", ".join(cols)
        select = f"SELECT DISTINCT ON ({distinct_on}) {col_list} FROM _bulk" if distinct_on else f"SELECT {col_list} FROM _bulk"
        con.execute(f"INSERT INTO {table} ({col_list}) {select}")
    finally:
        con.unregister("_bulk")
    return len(rows)


def _chunks(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


# ---------- NHL ----------

def load_nhl(con) -> dict:
    box_dir, pbp_dir = RAW_DIR / "nhl" / "boxscore", RAW_DIR / "nhl" / "pbp"
    items = [(path.name, _signature(path, pbp_dir / path.name)) for path in sorted(box_dir.glob("*.json"))]
    todo = _pending(con, "nhl_game", items)
    n_games = n_shots = 0
    for chunk in _chunks(todo, BATCH_FILES):
        games, teams, goalies, shots, ids = [], [], [], [], []
        for name, _sig in chunk:
            box = json.loads((box_dir / name).read_text())
            if box.get("gameState") not in ("OFF", "FINAL"):
                continue
            game, t, g = parse_boxscore(box)
            pbp_path = pbp_dir / name
            pbp = json.loads(pbp_path.read_text()) if pbp_path.exists() else None
            games.append((*game, ot_empty_net_loss(pbp) if pbp else None))
            teams += t
            goalies += g
            if pbp:
                shots += parse_pbp(pbp)
            ids.append(game[0])
        if ids:
            id_list = ", ".join(str(i) for i in ids)  # integer game ids only
            for table in ("games", "team_games", "goalie_games", "shots"):
                con.execute(f"DELETE FROM {table} WHERE game_id IN ({id_list})")
            _bulk(con, "games", GAME_COLS, games)
            _bulk(con, "team_games", TEAM_COLS, teams)
            _bulk(con, "goalie_games", GOALIE_COLS, goalies)
            _bulk(con, "shots", SHOT_COLS, shots, distinct_on="game_id, event_id")
        _mark_loaded(con, "nhl_game", chunk)
        n_games += len(ids)
        n_shots += len(shots)
    return {"games": n_games, "shots": n_shots, "skipped_unchanged": len(items) - len(todo)}


# ---------- odds ----------

def parse_odds_snapshot(payload: dict, league: str) -> list[tuple]:
    rows = []
    for ev in payload["data"]:
        for book in ev.get("bookmakers", []):
            for market in book.get("markets", []):
                for o in market.get("outcomes", []):
                    rows.append((
                        payload["taken_at"], league, ev["id"], ev.get("commence_time"),
                        ev.get("home_team"), ev.get("away_team"), book["key"], market["key"],
                        o["name"], o.get("price"), o.get("point"),
                    ))
    return rows


def _load_odds_dir(con, subdir: str, kind: str, to_payload) -> dict:
    """Shared loader for live (odds/) and historical (odds_hist/) snapshot files."""
    loaded = skipped = 0
    for league_dir in sorted((RAW_DIR / subdir).glob("*")):
        league = league_dir.name
        items = [(f"{league}/{p.name}", _signature(p)) for p in sorted(league_dir.glob("*.json"))]
        todo = _pending(con, kind, items)
        skipped += len(items) - len(todo)
        for chunk in _chunks(todo, BATCH_FILES):
            rows = []
            for key, _sig in chunk:
                payload = to_payload(json.loads((RAW_DIR / subdir / key).read_text()))
                rows += parse_odds_snapshot(payload, league)
            _bulk(con, "odds_snapshots", ODDS_COLS, rows, distinct_on=ODDS_KEY,
                  replace_where="league = ? AND taken_at IN (SELECT DISTINCT taken_at FROM _bulk)",
                  replace_params=[league])
            _mark_loaded(con, kind, chunk)
            loaded += len(rows)
    return {"rows": loaded, "skipped_unchanged": skipped}


def load_odds(con) -> dict:
    return _load_odds_dir(con, "odds", "odds_live", lambda d: d)


def load_odds_hist(con) -> dict:
    """taken_at is the snapshot time The Odds API returns, not the time requested."""
    return _load_odds_dir(con, "odds_hist", "odds_hist",
                          lambda d: {"taken_at": d["timestamp"], "data": d.get("data", [])})


# ---------- BetNHL ----------

def load_betnhl(con) -> int:
    n = 0
    for path in sorted((RAW_DIR / "betnhl").glob("*.picks.json")):
        doc = json.loads(path.read_text())
        con.execute("DELETE FROM betnhl_picks WHERE pick_date = ?", [doc["date"]])
        for i, p in enumerate(doc["picks"]):
            con.execute("INSERT INTO betnhl_picks VALUES (?,?,?,?,?,?)", [
                doc["date"], i, p["pick_text"], p["first_decimal_odds"], p["first_american_odds"], doc["extractor"]])
            n += 1
    return n


def reset_loaded(con) -> None:
    """For `maxine load --full`: forget what was loaded so every raw file is parsed again."""
    con.execute("DELETE FROM loaded_files")
