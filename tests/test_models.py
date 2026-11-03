from maxine.models import elo
from maxine.teams import ODDS_NAME_TO_NHL


def test_win_prob_symmetry():
    assert elo.win_prob(0) == 0.5
    assert abs(elo.win_prob(50) + elo.win_prob(-50) - 1) < 1e-12


def test_elo_is_pregame_and_updates():
    games = [(1, 20242025, "AAA", "BBB", 4, 1), (2, 20242025, "AAA", "BBB", 2, 3)]
    preds = elo.run(games, elo.EloParams(k=10, home_adv=0))
    assert preds[1] == 0.5                      # no information before the first game
    assert preds[2] > 0.5                       # AAA won game 1, so favored in game 2


def test_franchise_inherits_rating():
    games = [(1, 20232024, "ARI", "BBB", 5, 0), (2, 20242025, "UTA", "BBB", 1, 2)]
    preds = elo.run(games, elo.EloParams(k=10, home_adv=0, carryover=1.0))
    assert preds[2] > 0.5                       # Utah starts from Arizona's rating


def test_team_map_covers_utah_and_montreal():
    assert ODDS_NAME_TO_NHL["Utah Mammoth"] == ODDS_NAME_TO_NHL["Utah Hockey Club"] == "UTA"
    assert ODDS_NAME_TO_NHL["Montréal Canadiens"] == "MTL"


def test_goalie_candidates_back_to_back():
    import pandas as pd
    from maxine import goalies
    d = pd.Timestamp("2026-01-10")
    starters = [1, 1, 2, 1, 1]
    dates = [d - pd.Timedelta(days=k) for k in (9, 7, 5, 3, 1)]
    last = {1: dates[-1], 2: dates[2]}
    rows = {r["player_id"]: r for r in goalies._candidate_rows(starters, dates, last, d, {})}
    assert set(rows) == {1, 2}
    assert rows[1]["b2b"] == 1 and rows[1]["b2b_started_last"] == 1 and rows[2]["b2b_started_last"] == 0
    assert rows[1]["share5"] == 0.8 and rows[2]["days_since"] == 5


def test_exposure():
    from maxine.picks import exposure
    assert exposure("h2h", "a") == exposure("spreads", "a") == exposure("reg3", "home") == "home"
    assert exposure("totals", "a") == exposure("totals", "b") == "total"
    assert exposure("reg3", "draw") == "draw"


def test_select_bets_one_per_exposure():
    import pandas as pd
    from maxine.picks import select_bets
    b = pd.DataFrame([  # the TOR @ CBJ board: CBJ moneyline, CBJ -1.5, CBJ in regulation, under (no pick)
        {"game_id": 1, "market": "h2h", "side": "a", "edge": 0.089, "min_edge": 0.05, "is_pick": True},
        {"game_id": 1, "market": "spreads", "side": "a", "edge": 0.074, "min_edge": 0.07, "is_pick": True},
        {"game_id": 1, "market": "reg3", "side": "home", "edge": 0.090, "min_edge": 0.05, "is_pick": True},
        {"game_id": 1, "market": "totals", "side": "b", "edge": 0.001, "min_edge": 0.05, "is_pick": False},
    ])
    out = select_bets(b).set_index("market")
    assert out["is_pick"].tolist() == [False, False, True, False]
    assert out.loc["h2h", "covered_by"] == "reg3"


def test_goalie_candidates_missing_starter():
    """A completed game with no starter in the box score (NULL -> pd.NA) must not break the frame."""
    import duckdb
    from maxine import goalies
    con = duckdb.connect()
    con.execute("""CREATE TABLE games AS SELECT * FROM (VALUES
        (1, 20262027, TIMESTAMP '2026-10-10 23:00', DATE '2026-10-10', 'NYR', 'BOS', 3, 2, 'NHL', 2),
        (2, 20262027, TIMESTAMP '2026-10-12 23:00', DATE '2026-10-12', 'NYR', 'BOS', 1, 2, 'NHL', 2),
        (3, 20262027, TIMESTAMP '2026-10-14 23:00', DATE '2026-10-14', 'NYR', 'BOS', 4, 1, 'NHL', 2))
        t(game_id, season, start_utc, game_date, home, away, home_score, away_score, league, game_type)""")
    con.execute("""CREATE TABLE goalie_games AS SELECT * FROM (VALUES
        (1, 'NYR', 30, true), (1, 'BOS', 40, true), (3, 'NYR', 30, true), (3, 'BOS', 41, true))
        t(game_id, team, player_id, starter)""")      # game 2 has no goalie rows at all
    c = goalies.candidate_frame(con)
    # game 2: candidates from game 1, nobody recorded as starting; game 3: NYR's 30 starts again,
    # BOS starts a new goalie (41) who was not a candidate
    assert sorted(c.loc[c["game_id"] == 2, "y"]) == [0, 0]
    g3 = c[c["game_id"] == 3].set_index("player_id")["y"].to_dict()
    assert g3 == {30: 1, 40: 0}


def test_parlays_one_leg_per_game_and_book_pricing():
    import pandas as pd
    from maxine import parlays
    t = pd.Timestamp("2026-11-04T00:00Z")
    games = [("DAL", "WSH", 1), ("STL", "NYR", 2), ("COL", "CBJ", 3), ("EDM", "NJD", 4)]
    inputs, books = [], []
    for away, home, gid in games:
        # model likes the away +1.5 more than the market does; moneyline has no edge either way
        inputs.append({"game_id": gid, "home": home, "away": away, "start_utc": t, "market": "spreads",
                       "line": -1.5, "p_a": 0.30, "p_mkt_a": 0.33})
        inputs.append({"game_id": gid, "home": home, "away": away, "start_utc": t, "market": "h2h",
                       "line": 0.0, "p_a": 0.55, "p_mkt_a": 0.55})
        for book, b in (("dk", 1.40), ("fd", 1.45)):
            books.append({"home": home, "away": away, "commence": t, "market": "spreads", "line": -1.5,
                          "book": book, "a": 2.9, "b": b})
    books[-1]["book"] = "other"            # fd misses the last game: only dk has all four legs
    out = parlays.build(inputs, pd.DataFrame(books))
    assert [p["legs_n"] for p in out] == [3, 4]
    four = out[1]
    assert all(l["selection"].endswith("+1.5") for l in four["legs"])
    assert four["book"] == "dk" and abs(four["price"] - 1.40 ** 4) < 1e-9
    assert abs(four["p_model"] - 0.70 ** 4) < 1e-9
