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


def test_paper_store_dedupe_and_grade():
    server.run_migrations()
    pick = pd.DataFrame([{"game_id": 99, "game_date": "2026-10-20", "start_utc": pd.Timestamp("2026-10-21T00:00Z"),
                          "home": "NYR", "away": "ANA", "side": "away", "pick_team": "ANA", "p_model": 0.50,
                          "p_market": 0.56, "edge": 0.06, "best_price": 2.25, "best_book": "fanduel",
                          "med_price": 2.15, "books": 10}])
    assert paper.store_picks(pick) == 1
    assert paper.store_picks(pick) == 0          # first price seen is the bet; never overwritten
    con = duckdb.connect()
    con.execute("CREATE TABLE games (game_id BIGINT, home_score INT, away_score INT)")
    con.execute("INSERT INTO games VALUES (99, 2, 3)")   # away team won
    assert paper.grade(con) == 1
    rec = paper.summary()["record"]
    assert rec["bets"] == 1 and rec["wins"] == 1 and rec["units"] == 1.25
