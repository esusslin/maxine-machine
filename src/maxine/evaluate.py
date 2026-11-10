"""Season evaluation: how the model does on every finished game this season, market by market, and how the
logged picks did, split by side (favourite / underdog, over / under).

Model numbers are honest out-of-sample: the moneyline and goal models for this season are fit only on
earlier seasons (walk-forward), team form uses only games before each game, and the totals / puck line
calibration is fit on earlier seasons only. The starting goalie input is the expected starter, as live.

"Market" = the median no-margin price in the last odds snapshot we saved before puck drop (usually the
2 PM run, or the closing-price capture for picked games). Main line = the line most books quoted.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from maxine import features, market
from maxine.models import goals, winprob
from maxine.warehouse import load


def _ll(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _rate(x) -> float | None:
    x = list(x)
    return round(sum(x) / len(x), 4) if x else None


def _summary(model_p, market_p, won_a) -> dict:
    """model_p / market_p: chance of side A; won_a: 1 if side A won. Rows with a market only for market stats."""
    d = pd.DataFrame({"m": model_p, "k": market_p, "y": won_a}).dropna(subset=["m", "y"])
    k = d.dropna(subset=["k"])
    out = {"games": int(len(d)),
           "model_right": _rate(((d.m >= 0.5) == (d.y == 1))),
           "model_log_loss": round(_ll(d.m, d.y), 4) if len(d) else None,
           "games_with_market": int(len(k)),
           "model_right_same_games": _rate(((k.m >= 0.5) == (k.y == 1))),
           "market_right": _rate(((k.k >= 0.5) == (k.y == 1))),
           "model_log_loss_same_games": round(_ll(k.m, k.y), 4) if len(k) else None,
           "market_log_loss": round(_ll(k.k, k.y), 4) if len(k) else None}
    return out


def _lean(model_p, market_p, won_a, a_name: str, b_name: str) -> dict:
    """Where the model disagreed with the market: how often the side it leaned to won, by side."""
    d = pd.DataFrame({"m": model_p, "k": market_p, "y": won_a}).dropna()
    out = {}
    for name, sel, hit, kside in ((a_name, d.m > d.k, d.y == 1, d.k), (b_name, d.m < d.k, d.y == 0, 1 - d.k)):
        big = sel & ((d.m - d.k).abs() >= 0.03)
        out[name] = {"leans": int(sel.sum()), "won": _rate(hit[sel]),
                     "market_expected": round(float(kside[sel].mean()), 4) if sel.any() else None,
                     "leans_3pts": int(big.sum()), "won_3pts": _rate(hit[big]),
                     "market_expected_3pts": round(float(kside[big].mean()), 4) if big.any() else None}
    return out


def model_season(con, season: int) -> dict:
    load.load_odds(con)
    market.build_closing_lines(con)
    market.build_line_markets(con)
    out = {"season": season}

    # moneyline: walk-forward win model (fit on earlier seasons), calibrated on earlier seasons
    df = features.game_features(con)
    feats = [f for f in features.FEATURES if f != features.ACTUAL_GOALIE]
    preds, _ = winprob.walk_forward(df, 20222023, feats)
    g = df[(df["season"] == season) & df["game_id"].isin(preds)].copy()
    g["p"] = g["game_id"].map(preds)
    ml = con.execute("SELECT game_id, p_home_close FROM market_lines").df().set_index("game_id")["p_home_close"]
    g["k"] = g["game_id"].map(ml)
    out["moneyline"] = _summary(g["p"], g["k"], g["home_win"])
    out["moneyline"]["leans"] = _lean(g["p"], g["k"], g["home_win"], "home", "away")
    fav = np.where(g["k"] >= 0.5, g["p"], 1 - g["p"]), np.where(g["k"] >= 0.5, g["k"], 1 - g["k"])
    fav_won = np.where(g["k"] >= 0.5, g["home_win"], 1 - g["home_win"])
    out["moneyline"]["leans_fav_dog"] = _lean(pd.Series(fav[0]), pd.Series(fav[1]), pd.Series(fav_won),
                                               "favourite", "underdog")

    # puck line and totals: walk-forward goal model, calibration fit on earlier seasons only
    lam, models = goals.walk_forward(con)
    cal = goals.calibrators(con, before_season=season)
    res = goals.game_outcomes(con)
    res = res[res["season"] == season].merge(lam, on=["game_id", "season"])
    pts = con.execute("""
        SELECT p.* FROM market_points p JOIN market_lines m ON m.game_id = p.game_id AND p.ts = m.close_ts
    """).df()
    main = pts.sort_values("n_books", ascending=False).drop_duplicates(["game_id", "market"])
    rows = []
    for r in res.itertuples():
        pr = goals.price(models[int(r.season)], r.lam_h, r.lam_a)
        for mk in ("spreads", "totals"):
            m = main[(main["game_id"] == r.game_id) & (main["market"] == mk)]
            if m.empty:
                continue
            m = m.iloc[0]
            if mk == "totals":
                total = r.home_score + r.away_score
                if total == m.line:
                    continue                                           # push
                a, b, _ = goals.prob_total(pr, m.line)
                won_a = int(total > m.line)
            else:
                margin = r.home_score - r.away_score
                if margin + m.line == 0:
                    continue
                a, b, _ = goals.prob_spread(pr, m.line)
                won_a = int(margin + m.line > 0)
            p_a = goals.apply_cal(a / (a + b), cal.get(mk))
            rows.append({"market": mk, "line": float(m.line), "p": p_a, "k": float(m.p_a), "y": won_a})
    pm = pd.DataFrame(rows, columns=["market", "line", "p", "k", "y"])
    t = pm[pm["market"] == "totals"]
    out["totals"] = _summary(t["p"], t["k"], t["y"])
    out["totals"]["leans"] = _lean(t["p"], t["k"], t["y"], "over", "under")
    s = pm[pm["market"] == "spreads"].copy()
    # side A = home covers; turn into favourite (-1.5) / underdog (+1.5)
    home_fav = s["line"] < 0
    s["pf"], s["kf"] = np.where(home_fav, s["p"], 1 - s["p"]), np.where(home_fav, s["k"], 1 - s["k"])
    s["yf"] = np.where(home_fav, s["y"], 1 - s["y"])
    out["puck_line"] = _summary(s["pf"], s["kf"], s["yf"])
    out["puck_line"]["leans"] = _lean(s["pf"], s["kf"], s["yf"], "favourite -1.5", "underdog +1.5")
    return out


def picks_breakdown(rows: list[dict]) -> dict:
    """Graded paper picks by market and side (favourite / underdog, over / under, home / draw / away)."""
    out = {}
    for r in rows:
        if not r.get("result"):
            continue
        m = r["market"]
        if m == "totals":
            side = "over" if r["side"] == "a" else "under"
        elif m == "spreads":
            team_line = r["line"] if r["side"] == "a" else -r["line"]
            side = "underdog +1.5" if team_line > 0 else "favourite -1.5"
        elif m == "h2h":
            side = "favourite" if (r.get("p_market") or 0) >= 0.5 else "underdog"
        else:
            side = r["side"]
        k = f"{m}|{side}"
        o = out.setdefault(k, {"market": m, "side": side, "bets": 0, "wins": 0, "losses": 0, "pushes": 0,
                               "units": 0.0, "clv_n": 0, "beat_close": 0})
        o["bets"] += 1
        o["wins"] += r["result"] == "W"
        o["losses"] += r["result"] == "L"
        o["pushes"] += r["result"] == "P"
        o["units"] = round(o["units"] + (r.get("profit") or 0.0), 2)
        if r.get("clv") is not None:
            o["clv_n"] += 1
            o["beat_close"] += r["clv"] > 0
    for o in out.values():
        dec = o["wins"] + o["losses"]
        o["win_pct"] = round(o["wins"] / dec, 3) if dec else None
        o["roi"] = round(o["units"] / o["bets"], 3) if o["bets"] else None
        o["beat_close_pct"] = round(o["beat_close"] / o["clv_n"], 3) if o["clv_n"] else None
    return dict(sorted(out.items()))
