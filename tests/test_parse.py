"""Parser tests on hand-built fixtures shaped like the NHL web API and BetNHL pages.

These check our own parsing logic; the first real backfill + `maxine validate` checks the shapes.
"""
from datetime import date

import duckdb

from maxine.ingest.betnhl import extract_picks, tip_url
from maxine.validate import rebuilt_standings
from maxine.warehouse.load import ot_empty_net_loss, parse_boxscore, parse_pbp

BOX = {
    "id": 2024020001, "season": 20242025, "gameType": 2, "gameDate": "2024-10-04",
    "startTimeUTC": "2024-10-04T17:00:00Z", "gameState": "OFF",
    "venue": {"default": "O2 Czech Republic"},
    "gameOutcome": {"lastPeriodType": "REG"},
    "homeTeam": {"id": 1, "abbrev": "NJD", "score": 4, "sog": 31},
    "awayTeam": {"id": 7, "abbrev": "BUF", "score": 1, "sog": 22},
    "playerByGameStats": {
        "homeTeam": {"goalies": [{"playerId": 11, "name": {"default": "J. Markstrom"}, "toi": "60:00",
                                  "starter": True, "saveShotsAgainst": "21/22", "goalsAgainst": 1, "decision": "W"}]},
        "awayTeam": {"goalies": [{"playerId": 22, "name": {"default": "U. Luukkonen"}, "toi": "58:30",
                                  "starter": True, "saveShotsAgainst": "27/30", "goalsAgainst": 3, "decision": "L"}]},
    },
}

PBP = {"id": 2024020001, "plays": [
    {"eventId": 5, "typeDescKey": "faceoff", "details": {}},
    {"eventId": 8, "typeDescKey": "shot-on-goal", "timeInPeriod": "01:10", "situationCode": "1551",
     "periodDescriptor": {"number": 1},
     "details": {"eventOwnerTeamId": 1, "shootingPlayerId": 99, "goalieInNetId": 22,
                 "shotType": "wrist", "xCoord": 70, "yCoord": -5}},
    {"eventId": 9, "typeDescKey": "goal", "timeInPeriod": "02:00", "situationCode": "1551",
     "periodDescriptor": {"number": 1},
     "details": {"eventOwnerTeamId": 1, "scoringPlayerId": 98, "goalieInNetId": 22,
                 "shotType": "snap", "xCoord": 80, "yCoord": 3}},
]}


def test_parse_boxscore():
    game, teams, goalies = parse_boxscore(BOX)
    assert game[0] == 2024020001 and game[6] == "NJD" and game[12] == "REG"
    assert teams[0] == (2024020001, "NJD", True, 4, 31)
    assert goalies[1][5] == 58 * 60 + 30          # toi seconds
    assert goalies[1][6:9] == (30, 27, 3)          # shots against, saves, goals against


def test_parse_pbp_keeps_only_shot_events():
    rows = parse_pbp(PBP)
    assert [r[1] for r in rows] == [8, 9]
    assert rows[1][7] == 98                         # scorer used as shooter on goals


def test_rebuilt_standings_points():
    con = duckdb.connect()
    con.execute("CREATE TABLE games (league VARCHAR, season INT, game_type INT, home VARCHAR, away VARCHAR,"
                " home_score INT, away_score INT, last_period_type VARCHAR, ot_empty_net_loss BOOLEAN)")
    con.executemany("INSERT INTO games VALUES ('NHL', 20242025, 2, ?, ?, ?, ?, ?, ?)", [
        ("NJD", "BUF", 4, 1, "REG", False),   # NJD W, BUF L
        ("BUF", "NJD", 3, 2, "SO", False),    # BUF W, NJD OTL
        ("MIN", "VGK", 1, 2, "OT", True),     # VGK W, MIN pulled goalie in OT: L, no point
    ])
    t = rebuilt_standings(con, 20242025)
    assert t["NJD"] == {"gp": 2, "w": 1, "l": 0, "otl": 1, "pts": 3}
    assert t["BUF"] == {"gp": 2, "w": 1, "l": 1, "otl": 0, "pts": 2}
    assert t["MIN"] == {"gp": 1, "w": 0, "l": 1, "otl": 0, "pts": 0}


def test_ot_empty_net_loss():
    def pbp(period_type, goalie):
        return {"plays": [{"typeDescKey": "goal", "periodDescriptor": {"periodType": period_type},
                           "details": {"goalieInNetId": goalie}}]}
    assert ot_empty_net_loss(pbp("OT", None)) is True
    assert ot_empty_net_loss(pbp("OT", 8470594)) is False
    assert ot_empty_net_loss(pbp("REG", None)) is False      # regulation empty-netter is a normal loss anyway


def test_betnhl_url():
    assert tip_url(date(2026, 10, 16)).endswith("/tonights-tips/october-16th-2026-nhl-betting-tips.php")
    assert tip_url(date(2026, 10, 1)).endswith("/october-1st-2026-nhl-betting-tips.php")
    assert tip_url(date(2026, 10, 12)).endswith("/october-12th-2026-nhl-betting-tips.php")


def test_betnhl_extract():
    html = "<p><b>Pick:</b> Avalanche win in regulation</p><ul><li>Ontario: 1.78</li><li>USA: -128</li></ul>"
    picks = extract_picks(html)
    assert picks == [{"pick_text": "Avalanche win in regulation",
                      "first_decimal_odds": 1.78, "first_american_odds": -128}]
