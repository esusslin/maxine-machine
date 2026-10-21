"""Serving plane: auth flow copied from the-algo, paper-pick storage and grading."""
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.update({"APP_DB_PATH": os.path.join(_tmp, "app.db"), "ENABLE_JOBS": "false",
                   "ADMIN_USERNAME": "admin", "ADMIN_PASSWORD": "adminpass1", "JWT_SECRET_KEY": "test-secret"})

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from maxine import paper  # noqa: E402


def test_login_invite_and_admin_guard():
    with TestClient(server.app) as c:
        assert c.get("/api/board").status_code == 401
        tok = c.post("/api/auth/login", json={"username": "admin", "password": "adminpass1"}).json()["token"]
        h = {"Authorization": f"Bearer {tok}"}
        assert c.post("/api/auth/login", json={"username": "admin", "password": "wrong"}).status_code == 401
        inv = c.post("/api/admin/invites", headers=h, json={"note": "t"}).json()
        r = c.post("/api/auth/register", json={"invite": inv["code"], "username": "pal", "password": "palpass12"})
        assert r.status_code == 200
        h2 = {"Authorization": f"Bearer {r.json()['token']}"}
        assert c.get("/api/board", headers=h2).status_code == 200
        assert c.get("/api/admin/status", headers=h2).status_code == 403
        # an invite works once
        again = c.post("/api/auth/register", json={"invite": inv["code"], "username": "pal2", "password": "palpass12"})
        assert again.status_code == 400


def _bet(game_id, market, side, selection, line, price=2.0, line_type="early"):
    return {"game_id": game_id, "game_date": "2026-10-20", "start_utc": pd.Timestamp("2026-10-21T00:00Z"),
            "home": "NYR", "away": "ANA", "market": market, "side": side, "selection": selection, "line": line,
            "line_type": line_type, "validated": market in ("h2h", "totals"), "p_model": 0.55, "p_market": 0.5,
            "edge": 0.05, "best_price": price, "best_book": "fanduel", "med_price": price - 0.05, "books": 10}


def test_paper_store_dedupe_and_grade_all_markets():
    server.run_migrations()
    bets = pd.DataFrame([
        {**_bet(99, "h2h", "b", "ANA", 0.0, 2.25), "stake": 1.5},   # away wins -> W, 1.5 units
        _bet(99, "totals", "a", "Over 5.5", 5.5),          # 3+2 = 5 -> L
        _bet(97, "spreads", "b", "ANA +1.5", -1.5),        # home line -1.5, away covers -> W
        _bet(99, "reg3", "draw", "Tie after regulation", 0.0, 4.2),   # reg 2-2 -> W
        _bet(98, "totals", "b", "Under 6", 6.0),           # 4+2 = 6 -> push
    ])
    assert paper.store_picks(bets) == 5
    assert paper.store_picks(bets) == 0              # first price seen is the bet; never overwritten
    # same game, same team as the open ANA moneyline bet: covered by it, not stored
    assert paper.store_picks(pd.DataFrame([_bet(99, "reg3", "away", "ANA in regulation", 0.0)])) == 0
    con = duckdb.connect()
    con.execute("CREATE TABLE games (game_id BIGINT, home_score INT, away_score INT, home_team_id INT, away_team_id INT)")
    con.execute("CREATE TABLE shots (game_id BIGINT, event_type VARCHAR, period INT, owner_team_id INT)")
    con.execute("INSERT INTO games VALUES (99, 2, 3, 1, 2), (98, 4, 2, 1, 2), (97, 2, 3, 1, 2)")
    con.executemany("INSERT INTO shots VALUES (?,?,?,?)", [(99, "goal", 1, 1), (99, "goal", 2, 1), (99, "goal", 2, 2),
                                                          (99, "goal", 3, 2), (99, "goal", 4, 2)])
    assert paper.grade(con) == 5
    res = {(r["game_id"], r["market"]): (r["result"], r["profit"]) for r in paper.summary()["picks"]}
    assert res[(99, "h2h")] == ("W", 1.875)
    assert res[(99, "totals")] == ("L", -1.0)
    assert res[(97, "spreads")][0] == "W"
    assert res[(99, "reg3")] == ("W", 3.2)
    assert res[(98, "totals")] == ("P", 0.0)
    by = paper.summary()["by_market"]
    assert by["totals|early"]["pushes"] == 1
