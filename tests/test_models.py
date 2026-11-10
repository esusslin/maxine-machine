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
    from maxine.blend import blend
    assert abs(four["p_model_only"] - 0.70 ** 4) < 1e-9
    assert abs(four["p_model"] - blend(0.70, 0.67, "spreads", 1.5) ** 4) < 1e-9     # away +1.5: underdog weight


def _ladder_payload():
    def book(key, alt, extra_total=False):
        totals = [{"name": "Over", "price": 1.25, "point": 4.5}, {"name": "Under", "price": 3.8, "point": 4.5},
                  {"name": "Over", "price": 1.9, "point": 6.0}]                     # whole number: skipped
        if extra_total:
            totals += [{"name": "Under", "price": 1.08, "point": 9.5}]               # one book only: not shared
        return lambda home, away: {"key": key, "markets": [
            {"key": "h2h", "outcomes": [{"name": home, "price": 1.70}, {"name": away, "price": 2.20}]},
            {"key": "alternate_spreads", "outcomes": [
                {"name": away, "price": alt, "point": 2.5}, {"name": home, "price": 3.9, "point": -2.5},
                {"name": away, "price": 1.55, "point": 1.5}, {"name": home, "price": 2.45, "point": -1.5}]},
            {"key": "alternate_totals", "outcomes": totals}]}
    books = [book("bovada", 1.22, extra_total=True), book("draftkings", 1.24), book("fanduel", 1.23)]
    games = [("New York Rangers", "Boston Bruins"), ("Dallas Stars", "St Louis Blues"),
             ("Tampa Bay Lightning", "Detroit Red Wings")]
    return {"events": [{"id": f"e{i}", "home_team": h, "away_team": a, "bookmakers": [b(h, a) for b in books]}
                       for i, (h, a) in enumerate(games)]}


def test_ladder_shared_lines_and_search():
    import pandas as pd
    from maxine import ladder
    from maxine.models.goals import GoalModel, price
    t = pd.Timestamp("2026-11-10T00:00Z")
    fx = pd.DataFrame([{"game_id": 1, "home": "NYR", "away": "BOS", "start_utc": t},
                       {"game_id": 2, "home": "DAL", "away": "STL", "start_utc": t},
                       {"game_id": 3, "home": "TBL", "away": "DET", "start_utc": t}])
    gm = GoalModel(None, None, 1.37, 0.36, 0.0, 0.9)
    prices = {g: price(gm, 3.1, 2.9) for g in (1, 2, 3)}
    cals = {("total", 5.5): (0.0, 1.0), ("spread", 1.5): (0.0, 1.0)}
    lg = ladder.legs(_ladder_payload(), fx, {1: 0.55, 2: 0.55, 3: 0.55}, prices, cals)
    assert not (lg["line"] % 1 == 0).loc[lg["kind"] != "moneyline"].any()     # Over 6.0 skipped
    assert not lg[lg["selection"] == "Under 9.5"]["shared"].any()              # only one book has it
    assert lg[lg["selection"] == "BOS +2.5"]["shared"].all()
    pb = lg[(lg["book"] == "bovada") & (lg["game_id"] == 1) & (lg["selection"] == "BOS +2.5")]["p_book"].iloc[0]
    assert abs(pb - (1 / 1.22) / (1 / 1.22 + 1 / 3.9)) < 1e-9                  # paired with NYR -2.5, same book
    opts = ladder.search(lg)
    assert 1 <= len(opts) <= 3
    assert [o["rank"] for o in opts] == list(range(1, len(opts) + 1))
    assert all(-145 <= o["american"] <= -120 and o["book"] in ("bovada", "draftkings") for o in opts)
    assert len({tuple(sorted(l["game_id"] for l in o["legs"])) for o in opts}) == len(opts)
    assert opts == sorted(opts, key=lambda o: -o["p_model"])
    assert all("Under 9.5" not in [l["selection"] for l in o["legs"]] for o in opts)
    for o in opts:
        assert len({l["game_id"] for l in o["legs"]}) == o["legs_n"]           # one leg per game


def test_picks_breakdown_sides():
    from maxine.evaluate import picks_breakdown
    rows = [{"market": "totals", "side": "b", "line": 6.5, "result": "W", "profit": 0.9, "clv": 0.01, "p_market": 0.5},
            {"market": "totals", "side": "a", "line": 6.5, "result": "L", "profit": -1.0, "clv": -0.01, "p_market": 0.5},
            {"market": "spreads", "side": "b", "line": -1.5, "result": "W", "profit": 0.6, "clv": None, "p_market": 0.6},
            {"market": "h2h", "side": "a", "line": 0, "result": None, "profit": None, "clv": None, "p_market": 0.6}]
    b = picks_breakdown(rows)
    assert b["totals|under"]["wins"] == 1 and b["totals|over"]["losses"] == 1
    assert "spreads|underdog +1.5" in b                       # away +1.5 when the home line is -1.5
    assert "h2h|favourite" not in b                            # ungraded picks are left out


def test_blend_weights_by_leg_type():
    from maxine.blend import blend, leg_type
    assert leg_type("puck line", 2.5) == leg_type("spreads", 1.5) == "puck line underdog"
    assert leg_type("puck line", -1.5) == "puck line favourite"
    assert leg_type("h2h") == "moneyline" and leg_type("totals") == "total"
    # equal inputs stay put; otherwise the result sits between market and model, nearer the heavier weight
    assert abs(blend(0.6, 0.6, "total") - 0.6) < 1e-12
    dog, tot = blend(0.80, 0.70, "puck line", 2.5), blend(0.80, 0.70, "total")
    assert 0.70 < tot < dog < 0.80 and dog > 0.75 > tot
    assert blend(0.8, None, "total") == 0.8
