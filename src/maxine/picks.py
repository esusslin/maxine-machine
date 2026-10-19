"""Evening picks at the early line.

Prices every NHL regular-season game starting in the next `hours` hours, against current odds.
A game is "clean" when neither team plays before it (all model inputs known at bet time).
Two kinds of pick, tracked separately:
  early     clean games on a later date than today (Pacific). The backtest found an edge here.
  game-day  clean games today, when game_day=True. The backtest found NO edge here (2023-26, edge >= 5%:
            1,081 bets, line moved our way 51%, ROI -0.8%); they are logged so the live record can say
            whether that holds.
The starting goalie is not used (usually unknown this early). Picks are logged for paper trading.
"""
import json
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from maxine import features
from maxine.config import DATA_DIR
from maxine.ingest import nhl, odds
from maxine.models import winprob, xg
from maxine.teams import ODDS_NAME_TO_NHL
from maxine.warehouse import load

PICK_FEATURES = [f for f in features.FEATURES if f != "gsax30_diff"]
CURRENT_SEASON = 20262027
PACIFIC = ZoneInfo("America/Los_Angeles")


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
    """Median no-vig home probability, median and best prices per event, from a live snapshot file."""
    payload = json.loads(open(snap["path"]).read())
    rows = []
    for ev in payload["data"]:
        home, away = ODDS_NAME_TO_NHL.get(ev["home_team"]), ODDS_NAME_TO_NHL.get(ev["away_team"])
        for b in ev.get("bookmakers", []):
            for m in b.get("markets", []):
                if m["key"] != "h2h":
                    continue
                prices = {o["name"]: o["price"] for o in m["outcomes"]}
                hp, ap = prices.get(ev["home_team"]), prices.get(ev["away_team"])
                if hp and ap and hp > 1 and ap > 1:
                    rows.append({"home": home, "away": away, "commence": pd.to_datetime(ev["commence_time"]),
                                 "book": b["key"], "hp": hp, "ap": ap,
                                 "p_book": (1 / hp) / (1 / hp + 1 / ap)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = df.groupby(["home", "away", "commence"]).agg(
        p_market=("p_book", "median"), books=("book", "nunique"),
        med_home_price=("hp", "median"), med_away_price=("ap", "median")).reset_index()
    best_h = df.loc[df.groupby(["home", "away", "commence"])["hp"].idxmax(), ["home", "away", "commence", "hp", "book"]]
    best_a = df.loc[df.groupby(["home", "away", "commence"])["ap"].idxmax(), ["home", "away", "commence", "ap", "book"]]
    agg = agg.merge(best_h.rename(columns={"hp": "best_home_price", "book": "best_home_book"}), on=["home", "away", "commence"])
    return agg.merge(best_a.rename(columns={"ap": "best_away_price", "book": "best_away_book"}), on=["home", "away", "commence"])


def run(con, hours: int = 36, min_edge: float = 0.05, game_day: bool = False) -> dict:
    status = catch_up(con)
    fx = fixtures(hours)
    if fx.empty:
        return {"status": status, "board": pd.DataFrame(), "picks": pd.DataFrame()}
    predict = winprob.fit_final(features.game_features(con), PICK_FEATURES)
    feats = features.upcoming_features(con, fx)
    fx = fx.merge(feats, on="game_id", how="inner")
    fx["p_model"] = predict(fx)

    snap = odds.snapshot("nhl")
    mk = market_now(snap)
    if mk.empty:
        mk = pd.DataFrame(columns=["home", "away", "commence", "p_market", "books", "med_home_price",
                                   "med_away_price", "best_home_price", "best_home_book",
                                   "best_away_price", "best_away_book"])
    fx["start_utc"] = pd.to_datetime(fx["start_utc"], utc=True)
    mk["commence"] = pd.to_datetime(mk["commence"], utc=True)
    board = fx.merge(mk, on=["home", "away"], how="left")
    board = board[(board["commence"].isna()) |
                  ((board["commence"] - board["start_utc"]).abs() < pd.Timedelta(hours=12))]
    home = board["p_model"] >= board["p_market"]
    board["side"] = home.map({True: "home", False: "away"})
    board["pick_team"] = board["home"].where(home, board["away"])
    board["edge"] = (board["p_model"] - board["p_market"]).where(home, board["p_market"] - board["p_model"])
    board["best_price"] = board["best_home_price"].where(home, board["best_away_price"])
    board["best_book"] = board["best_home_book"].where(home, board["best_away_book"])
    board["med_price"] = board["med_home_price"].where(home, board["med_away_price"])
    today_pt = datetime.now(PACIFIC).date()
    board["early"] = pd.to_datetime(board["game_date"]).dt.date > today_pt
    board["line_type"] = board["early"].map({True: "early", False: "game-day"})
    board["is_pick"] = (board["clean"] & (board["early"] | game_day) & board["p_market"].notna()
                        & (board["edge"] >= min_edge))
    picks = board[board["is_pick"]].copy()
    picks["logged_at"] = snap["path"].rsplit("/", 1)[-1].removesuffix(".json")
    log_dir = DATA_DIR / "picks"
    log_dir.mkdir(parents=True, exist_ok=True)
    cols = ["logged_at", "line_type", "game_id", "start_utc", "home", "away", "side", "pick_team", "p_model", "p_market",
            "edge", "best_price", "best_book", "med_price", "books"]
    if not picks.empty:
        out = picks[cols].copy()
        out["start_utc"] = out["start_utc"].astype(str)
        path = log_dir / f"{date.today().isoformat()}.jsonl"
        with open(path, "a") as fh:
            for rec in out.to_dict("records"):
                fh.write(json.dumps(rec) + "\n")
    return {"status": status, "credits_remaining": snap["credits_remaining"], "board": board, "picks": picks}


def print_report(r: dict) -> None:
    print("catch-up:", r["status"])
    board = r["board"]
    if board.empty:
        print("No upcoming regular-season games in the window.")
        return
    print(f"odds credits remaining: {r.get('credits_remaining')}\n")
    print(f"{'start (PT)':<17}{'game':<12}{'model':>7}{'market':>8}{'edge':>7}  {'lean':<5}{'best price':>11}  {'book':<14}{'clean':<6}{'line':<9}{'PICK'}")
    for b in board.sort_values("start_utc").itertuples():
        start = b.start_utc.tz_convert("America/Los_Angeles").strftime("%a %m-%d %H:%M")
        game = f"{b.away}@{b.home}"
        if pd.isna(b.p_market):
            print(f"{start:<17}{game:<12}{b.p_model:>7.3f}{'no line':>8}")
            continue
        print(f"{start:<17}{game:<12}{b.p_model:>7.3f}{b.p_market:>8.3f}{b.edge:>+7.3f}  {b.pick_team:<5}"
              f"{b.best_price:>11.2f}  {b.best_book:<14}{'yes' if b.clean else 'no':<6}"
              f"{'early' if b.early else 'game-day':<9}{'<== PICK' if b.is_pick else ''}")
    n = len(r["picks"])
    print(f"\n{n} pick(s) logged to data/picks/" if n else
          "\nNo picks (needs a clean game on a later date with a posted line and edge >= threshold).")
    print("Game-day lines are shown for reference only: the backtest found no edge on them.")
