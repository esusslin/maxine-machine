"""Before/after yardstick for model changes. Every change must improve these on 2023-26, out of sample.

  ml_log_loss          moneyline shot model (no goalie), walk-forward + calibration, 2023-26 games
  ml_early_5           early-line clean games, edge >= 5%: bets, avg CLV, share where the line moved our way
  totals_/spreads_ll   goal model log loss on the morning main line (pushes excluded)
  totals_early_3       totals early line, edge >= 3%: bets, CLV, moved our way
  spreads_early_5      puck line early line, edge >= 5%
  reg3_early_5         regulation 3-way early line, edge >= 5% (only if the EU feed is loaded)
"""
from __future__ import annotations

import math

from maxine import clv, features, markets_bt
from maxine.models import winprob

SEASONS = (20232024, 20242025, 20252026)


def evaluate(con) -> dict:
    out = {}
    df = features.game_features(con)
    feats = [f for f in features.FEATURES if f != "gsax30_diff"]
    preds, _ = winprob.walk_forward(df, 20222023, feats)
    test = df[df["season"].isin(SEASONS) & df["game_id"].isin(preds)]
    ll = -sum(y * math.log(preds[g]) + (1 - y) * math.log(1 - preds[g]) for g, y in zip(test["game_id"], test["home_win"]))
    out["ml_log_loss"] = round(ll / len(test), 4)
    m = clv.bets_table(con, preds, list(SEASONS), "early")
    b = m[m["edge"] >= 0.05]
    out["ml_early_5"] = (len(b), round(float(b["clv"].mean()), 4), round(float((b["clv"] > 0).mean()), 3))

    rows = markets_bt.calibrate(markets_bt.rows(con))
    for x in markets_bt.log_loss_vs_close(rows):
        if x["market"] in ("totals", "spreads"):
            out[f"{x['market']}_ll"] = x["model"]
    bets = markets_bt.bets(rows)
    for market, t in (("totals", 0.03), ("spreads", 0.05), ("reg3", 0.05)):
        s = bets[(bets["market"] == market) & (bets["bet_time"] == "early") & (bets["edge"] >= t)].dropna(subset=["clv"])
        if len(s):
            out[f"{market}_early_{int(t * 100)}"] = (len(s), round(float(s["clv"].mean()), 4), round(float((s["clv"] > 0).mean()), 3))
    return out
