"""Parse raw JSON/HTML into the warehouse. Idempotent: rows for a key are replaced, never duplicated."""
import json

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


def load_nhl(con) -> dict:
    n_games = n_shots = 0
    for path in sorted((RAW_DIR / "nhl" / "boxscore").glob("*.json")):
        box = json.loads(path.read_text())
        if box.get("gameState") not in ("OFF", "FINAL"):
            continue
        game, teams, goalies = parse_boxscore(box)
        gid = game[0]
        for table in ("games", "team_games", "goalie_games"):
            con.execute(f"DELETE FROM {table} WHERE game_id = ?", [gid])
        con.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", game)
        con.executemany("INSERT INTO team_games VALUES (?,?,?,?,?)", teams)
        if goalies:
            con.executemany("INSERT INTO goalie_games VALUES (?,?,?,?,?,?,?,?,?,?)", goalies)
        n_games += 1
        pbp_path = RAW_DIR / "nhl" / "pbp" / path.name
        if pbp_path.exists():
            shots = parse_pbp(json.loads(pbp_path.read_text()))
            con.execute("DELETE FROM shots WHERE game_id = ?", [gid])
            if shots:
                con.executemany("INSERT INTO shots VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", shots)
            n_shots += len(shots)
    return {"games": n_games, "shots": n_shots}


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


def load_odds(con) -> int:
    n = 0
    for league_dir in sorted((RAW_DIR / "odds").glob("*")):
        for path in sorted(league_dir.glob("*.json")):
            payload = json.loads(path.read_text())
            con.execute("DELETE FROM odds_snapshots WHERE taken_at = ? AND league = ?",
                        [payload["taken_at"], league_dir.name])
            rows = parse_odds_snapshot(payload, league_dir.name)
            if rows:
                con.executemany("INSERT INTO odds_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
            n += len(rows)
    return n


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
