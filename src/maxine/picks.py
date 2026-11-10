"""Picks for every market: moneyline, totals, puck line (spreads), regulation 3-way.

Prices every NHL regular-season game starting in the next `hours` hours, against current odds.
Moneyline: the shot-based win model (models/winprob). Totals, puck line, regulation 3-way: the goal
model (models/goals), totals and spreads calibrated per market on out-of-sample results.
A game is "clean" when neither team plays before it (all model inputs known at bet time).
Live: one run a day at 2 p.m. Pacific, today's games only (today_only=True, game_day=True).
Backtest note (docs/experiments.md): on the morning and ~75-minute lines, picks did not beat the fair
closing price; the edge measured was on the day-before line.
Goalies: the expected starter (goalies.py), since the starter is usually unknown this early. Picks are logged for paper trading.
"""
import json
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from maxine import features, goalies, ladder, live_markets, parlays
from maxine.config import DATA_DIR
from maxine.ingest import nhl, odds
from maxine.models import goals, winprob, xg
from maxine.teams import ODDS_NAME_TO_NHL
from maxine.warehouse import load

PICK_FEATURES = [f for f in features.FEATURES if f != features.ACTUAL_GOALIE]
CURRENT_SEASON = 20262027
PACIFIC = ZoneInfo("America/Los_Angeles")
MARKETS = ("h2h", "totals", "spreads", "reg3")
def exposure(market: str, side: str) -> str:
    """Bets on the same exposure win or lose together: moneyline, puck line and regulation on the same team.
    Totals and the regulation draw are their own groups."""
    if market == "totals":
        return "total"
    if market == "reg3":
        return side                         # home / away / draw
    return "home" if side == "a" else "away"


def select_bets(board: pd.DataFrame) -> pd.DataFrame:
    """One pick per game per exposure: keep the largest edge, mark the others as covered by it."""
    board = board.copy()
    board["exposure"] = [exposure(m, s) if isinstance(s, str) else None for m, s in zip(board["market"], board["side"])]
    board["covered_by"] = None
    for _, grp in board[board["is_pick"]].groupby(["game_id", "exposure"]):
        best = grp["edge"].idxmax()
        for i in grp.index.drop(best):
            board.loc[i, "is_pick"] = False
            board.loc[i, "covered_by"] = board.loc[best, "market"]
    return board


DEFAULT_MIN_EDGE = {"h2h": 0.05, "totals": 0.05, "spreads": 0.07, "reg3": 0.05}   # docs/experiments.md: strict CLV
VALIDATED = {("h2h", "early"), ("totals", "early"), ("reg3", "early")}
MARKET_LABEL = {"h2h": "Moneyline", "totals": "Total", "spreads": "Puck line", "reg3": "Regulation 3-way"}


def catch_up(con) -> dict:
    """Fetch this season's finished games through today, load, re-score xG."""
    start, _ = nhl.season_window(CURRENT_SEASON)
    n = nhl.ingest_range(start, date.today(), refresh_schedule=True)
    return {"fetched_final_games": n, "load": load.load_nhl(con), "xg": xg.score_shots(con)["shots_scored"]}


def fixtures(hours: int) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    games = nhl.games_between(date.today(), date.today() + timedelta(days=3), refresh=True)
    rows = []
    for g in games:
        start = datetime.fromisoformat(g["startTimeUTC"].replace("Z", "+00:00"))
        if g.get("gameType") == 2 and g.get("gameState") in ("FUT", "PRE") and now < start <= now + timedelta(hours=hours):
            rows.append({"game_id": g["id"], "start_utc": start, "game_date": g["_local_date"],
                         "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"]})
    df = pd.DataFrame(rows).drop_duplicates("game_id")
    if df.empty:
        return df
    # clean = neither team has another game between now and this game's start
    def clean(r):
        others = df[(df["game_id"] != r.game_id) & (df["start_utc"] < r.start_utc)]
        return not (others["home"].isin([r.home, r.away]) | others["away"].isin([r.home, r.away])).any()
    df["clean"] = [clean(r) for r in df.itertuples()]
    return df.sort_values("start_utc").reset_index(drop=True)




def market_now(snap: dict) -> pd.DataFrame:
    """Moneyline table with the original column names (used by paper.capture_closes for h2h)."""
    tw = live_markets.two_way(snap["path"])
    if tw.empty:
        return tw
    h = tw[tw["market"] == "h2h"].rename(columns={
        "p_a": "p_market", "med_a": "med_home_price", "med_b": "med_away_price",
        "best_a": "best_home_price", "best_a_book": "best_home_book",
        "best_b": "best_away_price", "best_b_book": "best_away_book"})
    return h.reset_index(drop=True)


def _match(df: pd.DataFrame, f) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    m = df[(df["home"] == f.home) & (df["away"] == f.away)]
    return m[(m["commence"] - f.start_utc).abs() < pd.Timedelta(hours=12)]


def _two_way_row(base: dict, market: str, mk, p_model_a: float) -> dict:
    a = p_model_a >= mk.p_a
    line = float(mk.line)
    if market == "h2h":
        sel = base["home"] if a else base["away"]
    elif market == "spreads":
        team, tl = (base["home"], line) if a else (base["away"], -line)
        sel = f"{team} {tl:+.1f}"
    else:
        sel = f"{'Over' if a else 'Under'} {line:g}"
    return {**base, "market": market, "line": line, "side": "a" if a else "b", "selection": sel,
            "p_model": p_model_a if a else 1 - p_model_a, "p_market": mk.p_a if a else 1 - mk.p_a,
            "edge": abs(p_model_a - mk.p_a), "best_price": mk.best_a if a else mk.best_b,
            "best_book": mk.best_a_book if a else mk.best_b_book, "med_price": mk.med_a if a else mk.med_b,
            "books": int(mk.books)}


def run(con, hours: int = 36, min_edge: dict | float | None = None, game_day: bool = False,
        reg3_feed: bool = True, today_only: bool = False, ladder_feed: bool = True) -> dict:
    """today_only: price only today's games (Pacific date); the 2 p.m. game-day run uses this."""
    thresholds = dict(DEFAULT_MIN_EDGE)
    if isinstance(min_edge, (int, float)):
        thresholds = {k: float(min_edge) for k in thresholds}
    elif isinstance(min_edge, dict):
        thresholds.update(min_edge)

    status = catch_up(con)
    fx = fixtures(hours)
    if today_only and not fx.empty:
        fx = fx[fx["game_date"].astype(str) == datetime.now(PACIFIC).date().isoformat()]
    empty = {"status": status, "board": pd.DataFrame(), "picks": pd.DataFrame(), "credits_remaining": None,
             "parlays": []}
    if fx.empty:
        return empty
    fx["start_utc"] = pd.to_datetime(fx["start_utc"], utc=True)

    # models
    ml_predict = winprob.fit_final(features.game_features(con), PICK_FEATURES)
    gx = goalies.upcoming(con, fx)
    ml_feats = features.upcoming_features(con, fx, gx)
    p_ml = dict(zip(ml_feats["game_id"], ml_predict(ml_feats))) if not ml_feats.empty else {}
    gm = goals.fit_final(con)
    cal = goals.calibrators(con)
    lam = goals.game_lambdas(gm, goals.fixture_sides(features.latest_form(con), fx, goalie_exp=gx))
    prices = {r.game_id: goals.price(gm, r.lam_h, r.lam_a) for r in lam.itertuples()}

    # odds
    snap = odds.snapshot("nhl")
    tw = live_markets.main_lines(live_markets.two_way(snap["path"]))
    credits = snap["credits_remaining"]
    r3 = pd.DataFrame()
    if reg3_feed:
        try:
            snap_eu = odds.snapshot("nhl_eu")
            r3 = live_markets.reg3(snap_eu["path"])
            credits = snap_eu["credits_remaining"]
        except Exception:  # noqa: BLE001 — regulation market is optional
            r3 = pd.DataFrame()

    likely = {(r.game_id, r.team): f"{r.likely_goalie} ({r.likely_p:.0%})" for r in gx.itertuples()}
    today_pt = datetime.now(PACIFIC).date()
    rows, leg_in = [], []
    for f in fx.itertuples():
        early = pd.to_datetime(f.game_date).date() > today_pt
        base = {"game_id": f.game_id, "game_date": f.game_date, "start_utc": f.start_utc, "home": f.home,
                "away": f.away, "clean": bool(f.clean), "early": early,
                "line_type": "early" if early else "game-day",
                "home_goalie": likely.get((f.game_id, f.home)), "away_goalie": likely.get((f.game_id, f.away))}
        mk_all = _match(tw, f)
        pr = prices.get(f.game_id)
        # moneyline
        mk = mk_all[mk_all["market"] == "h2h"] if not mk_all.empty else mk_all
        if f.game_id in p_ml and len(mk):
            rows.append(_two_way_row(base, "h2h", mk.iloc[0], float(p_ml[f.game_id])))
            leg_in.append({**base, "market": "h2h", "line": 0.0, "p_a": float(p_ml[f.game_id]),
                           "p_mkt_a": float(mk.iloc[0].p_a)})
        elif f.game_id in p_ml:
            rows.append({**base, "market": "h2h", "selection": None, "p_model": float(p_ml[f.game_id]),
                         "p_market": None, "edge": None})
        if pr is None or mk_all.empty:
            continue
        for market in ("totals", "spreads"):
            mk = mk_all[mk_all["market"] == market]
            if not len(mk):
                continue
            m0 = mk.iloc[0]
            a, b, _ = (goals.prob_total(pr, m0.line) if market == "totals" else goals.prob_spread(pr, m0.line))
            p_a = goals.apply_cal(a / (a + b), cal.get(market)) if (a + b) > 0 else None
            if p_a is not None:
                rows.append(_two_way_row(base, market, m0, p_a))
                leg_in.append({**base, "market": market, "line": float(m0.line), "p_a": p_a, "p_mkt_a": float(m0.p_a)})
        r = _match(r3, f)
        if len(r):
            r = r.iloc[0]
            model = {"home": pr["reg_home"], "draw": pr["reg_tie"], "away": pr["reg_away"]}
            o = max(model, key=lambda k: model[k] - r[f"p_{k}"])
            label = {"home": f"{f.home} in regulation", "draw": "Tie after regulation", "away": f"{f.away} in regulation"}[o]
            rows.append({**base, "market": "reg3", "line": 0.0, "side": o, "selection": label,
                         "p_model": model[o], "p_market": float(r[f"p_{o}"]), "edge": model[o] - float(r[f"p_{o}"]),
                         "best_price": float(r[f"best_{o}"]), "best_book": r[f"best_{o}_book"],
                         "med_price": float(r[f"med_{o}"]), "books": int(r["books"])})

    board = pd.DataFrame(rows)
    board["min_edge"] = board["market"].map(thresholds)
    board["validated"] = [(m, lt) in VALIDATED for m, lt in zip(board["market"], board["line_type"])]
    board["is_pick"] = (board["clean"] & (board["early"] | game_day) & board["p_market"].notna()
                        & (board["edge"].fillna(-1) >= board["min_edge"]))
    board = select_bets(board)
    picks = board[board["is_pick"]].copy()
    picks["logged_at"] = snap["path"].rsplit("/", 1)[-1].removesuffix(".json")
    if not picks.empty:
        log_dir = DATA_DIR / "picks"
        log_dir.mkdir(parents=True, exist_ok=True)
        out = picks.copy()
        out["start_utc"] = out["start_utc"].astype(str)
        with open(log_dir / f"{date.today().isoformat()}.jsonl", "a") as fh:
            for rec in out.to_dict("records"):
                fh.write(json.dumps(rec, default=str) + "\n")
    parlay_list = parlays.build(leg_in, live_markets.book_rows(snap["path"]))
    lad, lad_state = {"options": [], "legs": 0, "error": None}, None
    if ladder_feed:
        try:
            lad, lad_state, credits = _ladder(con, snap["path"], fx, p_ml, gm, lam, credits)
        except Exception as e:  # noqa: BLE001 — the Ladder tab must never break the picks run
            lad["error"] = f"{type(e).__name__}: {e}"
    return {"status": status, "credits_remaining": credits, "board": board, "picks": picks, "parlays": parlay_list,
            "ladder": lad, "ladder_state": lad_state}


def _ladder(con, us_snapshot_path: str, fx: pd.DataFrame, p_ml: dict, gm, lam: pd.DataFrame, credits):
    """Main + alternate lines from several books for today's games -> the three best parlays in the window.
    Also returns the model state needed to re-price the Ladder later without refitting (ladder_refresh)."""
    data = json.loads(open(us_snapshot_path).read())["data"]
    by_teams = {(f.home, f.away): f for f in fx.itertuples()}
    games = []
    for ev in data:
        f = by_teams.get((ODDS_NAME_TO_NHL.get(ev["home_team"]), ODDS_NAME_TO_NHL.get(ev["away_team"])))
        if f is not None:
            games.append({"game_id": int(f.game_id), "home": f.home, "away": f.away,
                          "start_utc": pd.Timestamp(f.start_utc).isoformat(), "event_id": ev["id"]})
    cals = goals.line_calibrators(con)
    state = {"games": games,
             "p_ml": {str(k): float(v) for k, v in p_ml.items()},
             "lam": {str(r.game_id): [float(r.lam_h), float(r.lam_a)] for r in lam.itertuples()},
             "gm": [gm.tie_factor, gm.en_shift, gm.ot_a, gm.ot_b],
             "line_cals": [[k[0], k[1], v[0], v[1]] for k, v in cals.items()]}
    out = ladder_refresh(state)
    state["alt_path"] = out.pop("alt_path", None)
    return out, state, out.pop("credits_remaining", None) or credits


def ladder_refresh(state: dict) -> dict:
    """Re-pull main + alternate lines for the games in `state` that have not started and re-rank. No model
    refit: uses the win chances, goal rates and line calibrations saved by the last full run."""
    now = pd.Timestamp.now(tz="UTC")
    games = [g for g in state.get("games", []) if pd.Timestamp(g["start_utc"]) > now]
    base = {"options": [], "lines": 0, "shared_lines": 0, "books": [], "games": 0, "bet_books": list(ladder.BET_BOOKS),
            "target": [ladder.TARGET_SHORT, ladder.TARGET_LONG], "priced_at": None, "error": None,
            "shared_rule": f"offered by {ladder.SHARED_MIN_BOOKS}+ books and at least half of those quoting the game"}
    if not games:
        return base
    fx = pd.DataFrame(games)
    fx["start_utc"] = pd.to_datetime(fx["start_utc"], utc=True)
    snap = odds.alt_lines([g["event_id"] for g in games], ladder.LINE_BOOKS)
    payload = json.loads(open(snap["path"]).read())
    lg = _ladder_legs(state, payload, fx)
    shared = lg[lg["shared"]] if len(lg) else lg
    return {**base, "options": ladder.search(lg), "lines": int(len(lg)), "shared_lines": int(len(shared)),
            "books": sorted(lg["book"].unique().tolist()) if len(lg) else [],
            "games": int(lg["game_id"].nunique()) if len(lg) else 0,
            "priced_at": payload.get("taken_at"), "credits_remaining": snap["credits_remaining"],
            "alt_path": snap["path"]}


def _ladder_legs(state: dict, payload: dict, fx: pd.DataFrame) -> pd.DataFrame:
    gm = goals.GoalModel(None, None, *state["gm"])
    prices = {int(k): goals.price(gm, lh, la) for k, (lh, la) in state["lam"].items()}
    p_ml = {int(k): v for k, v in state["p_ml"].items()}
    cals = {(kind, float(line)): (a, b) for kind, line, a, b in state["line_cals"]}
    return ladder.legs(payload, fx, p_ml, prices, cals)


def ladder_my_teams(state: dict, teams: list[dict]) -> dict:
    """teams: [{game_id, team}] -> the best way to pad exactly those teams (one leg each: moneyline or any puck
    line) into the window at both books, from the last pulled lines (no credits). If nothing fits, the
    combination closest to the window is returned and marked outside."""
    path = state.get("alt_path")
    if not path:
        raise RuntimeError("no lines yet: run the full formula or refresh the Ladder first")
    payload = json.loads(open(path).read())
    fx = pd.DataFrame(state["games"])
    fx["start_utc"] = pd.to_datetime(fx["start_utc"], utc=True)
    lg = _ladder_legs(state, payload, fx)
    want = {(int(t["game_id"]), t["team"]) for t in teams}
    sel = lg[lg["kind"].isin(["moneyline", "puck line"])].copy()
    sel = sel[[(g, s.split()[0]) in want for g, s in zip(sel["game_id"], sel["selection"])]]
    out = {"priced_at": payload.get("taken_at"), "teams": len(want), "inside_window": True,
           "target": [ladder.TARGET_SHORT, ladder.TARGET_LONG]}
    if sel["game_id"].nunique() < len(want):
        missing = sorted({t for g, t in want} - {s.split()[0] for s in sel["selection"]})
        return {**out, "options": [], "error": f"no shared lines at both books for: {', '.join(missing)}"}
    n = len(want)
    found = ladder.search(sel, sizes=(n,), options=1)
    if not found:                       # nothing lands in the window: the nearest option on each side
        out["inside_window"] = False
        shorter = (ladder.search(sel, short=ladder.TARGET_SHORT - 60, long=ladder.TARGET_SHORT - 1, sizes=(n,), options=1)
                   or ladder.search(sel, short=ladder.TARGET_SHORT - 260, long=ladder.TARGET_SHORT - 61,
                                    sizes=(n,), options=1))                  # -141..-200, else -201..-400
        longer = ladder.search(sel, short=ladder.TARGET_LONG + 1, long=150, sizes=(n,), options=1)   # -119..+150
        found = shorter + longer
    for i, o in enumerate(found, 1):
        o["rank"] = i
    return {**out, "options": found}


def print_report(r: dict) -> None:
    print("catch-up:", r["status"])
    board = r["board"]
    if board.empty:
        print("No upcoming regular-season games in the window.")
        return
    print(f"odds credits remaining: {r.get('credits_remaining')}")
    for gid, g in board.groupby("game_id", sort=False):
        f = g.iloc[0]
        start = f.start_utc.tz_convert("America/Los_Angeles").strftime("%a %m-%d %H:%M")
        print(f"\n{start}  {f.away}@{f.home}  [{f.line_type}{'' if f.clean else ', plays before this'}]")
        for b in g.itertuples():
            if pd.isna(b.p_market):
                print(f"   {MARKET_LABEL[b.market]:<17} model {b.p_model:.3f} (home)  no line")
                continue
            tag = ("PICK" if b.validated else "pick (unproven)") if b.is_pick else ""
            print(f"   {MARKET_LABEL[b.market]:<17} {b.selection:<24} model {b.p_model:.3f} mkt {b.p_market:.3f} "
                  f"edge {b.edge:+.3f}  {b.best_price:.2f} {b.best_book:<12} {tag}")
    n = len(r["picks"])
    print(f"\n{n} pick(s) logged." if n else "\nNo picks.")
