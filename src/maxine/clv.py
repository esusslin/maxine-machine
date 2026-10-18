"""Edge vs the market: does the line move toward the model, and would betting it have paid?

Two bet times, both free of look-ahead:
  early   = the earliest line we have (median ~33 h before puck), only for games where neither team
            plays between that snapshot and puck drop, so the model's inputs were all knowable then.
  morning = first snapshot after 14:00 UTC on game day, after the previous night's games are final.
Edge = model p - market no-vig p (bet time) for the side the model prefers. Bets: 1 unit flat at the
median book price at bet time when edge >= threshold. CLV = no-vig p of our side at close minus at
bet time (positive = market moved our way). Seasons are out of sample for both the model and its
calibration. The goalie input is left out by default: at bet time the starter is usually unknown.
"""
import pandas as pd

from maxine import features
from maxine.models import winprob

BET_TIMES = {
    "early": ("open_ts", "p_home_open", "med_home_price_open", "med_away_price_open"),
    "morning": ("morning_ts", "p_home_morning", "med_home_price_morning", "med_away_price_morning"),
}


def _clean_early_games(con) -> set[int]:
    return set(con.execute("""
        SELECT m.game_id FROM market_lines m JOIN games g USING (game_id)
        WHERE m.open_ts < m.morning_ts AND NOT EXISTS (
            SELECT 1 FROM games p WHERE p.league = 'NHL' AND p.game_type = 2
              AND p.start_utc > m.open_ts - INTERVAL 3 HOUR AND p.start_utc < g.start_utc
              AND (p.home IN (g.home, g.away) OR p.away IN (g.home, g.away)))
    """).df()["game_id"])


def bets_table(con, preds: dict, seasons: list[int], bet_time: str) -> pd.DataFrame:
    ts, p_col, hp, ap = BET_TIMES[bet_time]
    m = con.execute(f"""
        SELECT m.game_id, g.season, g.home_score > g.away_score AS home_win,
               m.{p_col} AS p_bet, m.p_home_close, m.{hp} AS home_price_bet, m.{ap} AS away_price_bet,
               m.med_home_price_close, m.med_away_price_close
        FROM market_lines m JOIN games g USING (game_id)
        WHERE m.{p_col} IS NOT NULL AND m.{ts} < m.close_ts
    """).df()
    m = m[m["season"].isin(seasons) & m["game_id"].isin(preds)].copy()
    if bet_time == "early":
        m = m[m["game_id"].isin(_clean_early_games(con))]
    m["p_model"] = m["game_id"].map(preds)
    home = m["p_model"] >= m["p_bet"]
    m["side"] = home.map({True: "home", False: "away"})
    m["edge"] = (m["p_model"] - m["p_bet"]).where(home, m["p_bet"] - m["p_model"])
    m["clv"] = (m["p_home_close"] - m["p_bet"]).where(home, m["p_bet"] - m["p_home_close"])
    m["price_bet"] = m["home_price_bet"].where(home, m["away_price_bet"])
    m["price_close"] = m["med_home_price_close"].where(home, m["med_away_price_close"])
    won = m["home_win"].where(home, ~m["home_win"])
    m["profit_bet"] = (m["price_bet"] - 1).where(won, -1.0)
    m["profit_close"] = (m["price_close"] - 1).where(won, -1.0)
    return m


def summarize(m: pd.DataFrame, thresholds=(0.0, 0.02, 0.03, 0.05)) -> list[dict]:
    rows = []
    for t in thresholds:
        b = m[m["edge"] >= t]
        if b.empty:
            continue
        rows.append({
            "min_edge": t, "bets": len(b),
            "avg_edge": round(float(b["edge"].mean()), 4),
            "avg_clv": round(float(b["clv"].mean()), 4),
            "clv_pos_pct": round(float((b["clv"] > 0).mean()), 3),
            "roi_bet_price": round(float(b["profit_bet"].mean()), 4),
            "roi_close_price": round(float(b["profit_close"].mean()), 4),
            "units": round(float(b["profit_bet"].sum()), 1),
        })
    return rows


def run(con, seasons=(20232024, 20242025, 20252026), use_goalie: bool = False) -> dict:
    feats = features.FEATURES if use_goalie else [f for f in features.FEATURES if f != "gsax30_diff"]
    preds, _ = winprob.walk_forward(features.game_features(con), first_test_season=20222023, features=feats)
    out = {}
    for bet_time in BET_TIMES:
        m = bets_table(con, preds, list(seasons), bet_time)
        out[bet_time] = {"all": summarize(m),
                         "by_season_edge_5pct": {int(s): summarize(g, (0.05,)) for s, g in m.groupby("season")}}
    return out


def print_report(r: dict) -> None:
    cols = ["min_edge", "bets", "avg_edge", "avg_clv", "clv_pos_pct", "roi_bet_price", "roi_close_price", "units"]
    for bet_time, res in r.items():
        print(f"\n{bet_time} line (seasons 2023-24 to 2025-26, 1u flat at median price)")
        print("  " + "".join(f"{c:>16}" for c in cols))
        for row in res["all"]:
            print("  " + "".join(f"{row[c]:>16}" for c in cols))
        print("  edge >= 5% by season:")
        for s, rows in res["by_season_edge_5pct"].items():
            if rows:
                x = rows[0]
                print(f"    {s}: bets {x['bets']}, CLV {x['avg_clv']:+.4f}, moved our way {x['clv_pos_pct']:.0%}, "
                      f"ROI {x['roi_bet_price']:+.1%}")
