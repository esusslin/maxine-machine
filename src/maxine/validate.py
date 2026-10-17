"""Phase 0 gate: standings rebuilt from stored games must match the NHL's official standings."""
from maxine.ingest import nhl


def rebuilt_standings(con, season: int) -> dict[str, dict]:
    rows = con.execute("""
        SELECT home, away, home_score, away_score, last_period_type, COALESCE(ot_empty_net_loss, FALSE)
        FROM games WHERE league = 'NHL' AND season = ? AND game_type = 2
    """, [season]).fetchall()
    table: dict[str, dict] = {}
    for home, away, hs, as_, period, en_loss in rows:
        for team in (home, away):
            table.setdefault(team, {"gp": 0, "w": 0, "l": 0, "otl": 0})
        winner, loser = (home, away) if hs > as_ else (away, home)
        table[winner]["w"] += 1
        # OT/SO loss earns a point, except when the loser pulled its goalie in OT and gave up the winner
        table[loser]["otl" if period in ("OT", "SO") and not en_loss else "l"] += 1
        table[home]["gp"] += 1
        table[away]["gp"] += 1
    for t in table.values():
        t["pts"] = 2 * t["w"] + t["otl"]
    return table


def official_standings(season: int) -> dict[str, dict]:
    _, end = nhl.season_window(season)
    data = nhl.standings_on(end)
    return {
        s["teamAbbrev"]["default"]: {"gp": s["gamesPlayed"], "w": s["wins"], "l": s["losses"],
                                     "otl": s["otLosses"], "pts": s["points"]}
        for s in data["standings"]
    }


def validate_season(con, season: int) -> list[str]:
    """Return a list of mismatches; empty list means the season passes."""
    ours, theirs = rebuilt_standings(con, season), official_standings(season)
    problems = []
    for team in sorted(set(ours) | set(theirs)):
        a, b = ours.get(team), theirs.get(team)
        if a != b:
            problems.append(f"{team}: rebuilt={a} official={b}")
    return problems
