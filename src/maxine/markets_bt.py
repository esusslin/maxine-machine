"""Per-market backtest of the goal model against each market's own lines (2023-26, out of sample).

Markets: moneyline (h2h, incl. OT/SO), totals, spreads (puck line), reg3 (regulation 3-way, EU feed:
one row per outcome; we may only bet an outcome, so a bet is the outcome with the largest positive edge). Bet times as in clv.py:
early (earliest snapshot, clean games only) and morning (first snapshot after 14:00 UTC on game day).
At each bet time the main line = the line most books quote. Side A = home (moneyline), home covers
(spreads, keyed by the home line), over (totals). Model probability for two-way markets is conditional
on no push, like the de-vigged market price. CLV only when the close quotes the same line.
Calibration: per market, logistic on logit(p) fit on the model's out-of-sample predictions from
EARLIER test seasons only (first test season uncalibrated), the same scheme as the moneyline model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from maxine.clv import _clean_early_games
from maxine.models import goals

TEST_SEASONS = (20232024, 20242025, 20252026)
EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _model_p(price: dict, market: str, line: float) -> float:
    if market == "h2h":
        return price["ml_home"]
    if market == "totals":
        o, u, _ = goals.prob_total(price, line)
    else:
        o, u, _ = goals.prob_spread(price, line)
    return o / (o + u) if (o + u) > 0 else np.nan


def _won(market, line, hs, as_):
    """+1 side A wins, -1 side B wins, 0 push."""
    if market == "h2h":
        return 1 if hs > as_ else -1
    v = (hs + as_ - line) if market == "totals" else (hs - as_ + line)
    return 0 if v == 0 else (1 if v > 0 else -1)


def rows(con) -> pd.DataFrame:
    lam, models = goals.walk_forward(con)
    games = goals.game_outcomes(con).merge(lam, on=["game_id", "season"])
    games = games[games["season"].isin(TEST_SEASONS)]
    prices = {r.game_id: goals.price(models[int(r.season)], r.lam_h, r.lam_a) for r in games.itertuples()}
    ml = con.execute("""SELECT game_id, open_ts, morning_ts, close_ts,
            p_home_open, med_home_price_open, med_away_price_open,
            p_home_morning, med_home_price_morning, med_away_price_morning,
            p_home_close, med_home_price_close, med_away_price_close FROM market_lines""").df().set_index("game_id")
    pts = con.execute("SELECT * FROM market_points").df()
    pts_by = {k: v for k, v in pts.groupby(["game_id", "ts", "market"])}
    try:
        r3 = con.execute("SELECT * FROM market_reg3 ORDER BY game_id, ts").df()
    except Exception:  # noqa: BLE001 — table absent until the EU feed is backfilled
        r3 = pd.DataFrame()
    r3_by = {k: v for k, v in r3.groupby("game_id")} if not r3.empty else {}
    clean = _clean_early_games(con)
    out = []
    for g in games.itertuples():
        if g.game_id not in ml.index:
            continue
        m = ml.loc[g.game_id]
        price = prices[g.game_id]
        for bet_time, ts in (("early", m.open_ts), ("morning", m.morning_ts)):
            if pd.isna(ts) or ts >= m.close_ts or (bet_time == "early" and g.game_id not in clean):
                continue
            # moneyline
            sfx = "open" if bet_time == "early" else "morning"
            out.append(dict(game_id=g.game_id, season=g.season, market="h2h", bet_time=bet_time, line=0.0,
                            p_model=price["ml_home"], p_bet=m[f"p_home_{sfx}"], p_close=m.p_home_close,
                            a_price=m[f"med_home_price_{sfx}"], b_price=m[f"med_away_price_{sfx}"],
                            result=_won("h2h", 0, g.home_score, g.away_score)))
            for market in ("totals", "spreads"):
                snap = pts_by.get((g.game_id, ts, market))
                if snap is None:
                    continue
                main = snap.sort_values("n_books", ascending=False).iloc[0]
                close = pts_by.get((g.game_id, m.close_ts, market))
                same = close[close["line"] == main.line] if close is not None else None
                out.append(dict(game_id=g.game_id, season=g.season, market=market, bet_time=bet_time,
                                line=float(main.line), p_model=_model_p(price, market, main.line),
                                p_bet=main.p_a, p_close=(same.iloc[0].p_a if same is not None and len(same) else np.nan),
                                a_price=main.med_a, b_price=main.med_b,
                                result=_won(market, main.line, g.home_score, g.away_score)))
            snaps = r3_by.get(g.game_id)
            if snaps is not None:
                before = snaps[(snaps["ts"] <= ts) & (snaps["ts"] >= ts - pd.Timedelta(minutes=30))]
                closing = snaps[snaps["ts"] <= m.close_ts]
                if len(before) and len(closing):
                    b, c = before.iloc[-1], closing.iloc[-1]
                    happened = {"home": g.home_reg > g.away_reg, "draw": g.home_reg == g.away_reg,
                                "away": g.home_reg < g.away_reg}
                    model = {"home": price["reg_home"], "draw": price["reg_tie"], "away": price["reg_away"]}
                    for o in ("home", "draw", "away"):
                        out.append(dict(game_id=g.game_id, season=g.season, market="reg3", bet_time=bet_time,
                                        line=0.0, outcome=o, p_model=model[o], p_bet=b[f"p_{o}"], p_close=c[f"p_{o}"],
                                        a_price=b[f"med_{o}"], b_price=np.nan,
                                        result=1 if happened[o] else -1))
    return pd.DataFrame(out)


def calibrate(df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward per-market Platt calibration of p_model -> p_cal."""
    from sklearn.linear_model import LogisticRegression
    df = df.copy()
    df["p_cal"] = df["p_model"]
    for market, part in df[df["market"] != "reg3"].groupby("market"):
        fit_on = part[(part["bet_time"] == "morning") & (part["result"] != 0)]
        for season in TEST_SEASONS:
            prior = fit_on[fit_on["season"] < season]
            if len(prior) < 300:
                continue
            lr = LogisticRegression(C=1e6).fit(_logit(prior["p_model"]).reshape(-1, 1), (prior["result"] == 1).astype(int))
            idx = part.index[part["season"] == season]
            df.loc[idx, "p_cal"] = lr.predict_proba(_logit(df.loc[idx, "p_model"]).reshape(-1, 1))[:, 1]
    return df


def bets(df: pd.DataFrame, p_col: str = "p_cal") -> pd.DataFrame:
    d = df.dropna(subset=[p_col, "p_bet"]).copy()
    if "outcome" in d.columns:
        two = d[d["market"] != "reg3"]
        r3 = d[d["market"] == "reg3"].copy()
        r3["edge"] = r3[p_col] - r3["p_bet"]
        r3 = r3[r3["edge"] > 0].sort_values("edge").groupby(["game_id", "bet_time"]).tail(1)
        r3["clv"] = r3["p_close"] - r3["p_bet"]
        r3["profit"] = np.where(r3["result"] == 1, r3["a_price"] - 1, -1.0)
        d = two
    else:
        r3 = None
    a = d[p_col] >= d["p_bet"]
    d["edge"] = (d[p_col] - d["p_bet"]).where(a, d["p_bet"] - d[p_col])
    d["clv"] = (d["p_close"] - d["p_bet"]).where(a, d["p_bet"] - d["p_close"])
    price = d["a_price"].where(a, d["b_price"])
    win = (d["result"] == 1).where(a, d["result"] == -1)
    d["profit"] = np.where(d["result"] == 0, 0.0, np.where(win, price - 1, -1.0))
    return pd.concat([d, r3], ignore_index=True) if r3 is not None and len(r3) else d


def summarize(d: pd.DataFrame, thresholds=(0.03, 0.05)) -> list[dict]:
    out = []
    for (market, bt), part in d.groupby(["market", "bet_time"]):
        for t in thresholds:
            b = part[part["edge"] >= t]
            c = b.dropna(subset=["clv"])
            if b.empty:
                continue
            out.append({"market": market, "bet_time": bt, "min_edge": t, "bets": len(b),
                        "avg_clv": round(float(c["clv"].mean()), 4) if len(c) else None,
                        "clv_pos_pct": round(float((c["clv"] > 0).mean()), 3) if len(c) else None,
                        "roi": round(float(b["profit"].mean()), 4),
                        "seasons_roi": {int(s): round(float(x["profit"].mean()), 3) for s, x in b.groupby("season")}})
    return out


def log_loss_vs_close(df: pd.DataFrame, p_col: str = "p_cal") -> list[dict]:
    """Model vs market on the morning main line (pushes excluded), and market close where same line."""
    out = []
    for market, part in df[(df["bet_time"] == "morning") & (df["result"] != 0) & (df["market"] != "reg3")].groupby("market"):
        y = (part["result"] == 1).astype(float)
        def ll(p):
            p = np.clip(p, EPS, 1 - EPS)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        sub = part.dropna(subset=["p_close"])
        ys = (sub["result"] == 1).astype(float)
        def ll2(p):
            p = np.clip(p, EPS, 1 - EPS)
            return float(-np.mean(ys * np.log(p) + (1 - ys) * np.log(1 - p)))
        out.append({"market": market, "n": len(part), "model": round(ll(part[p_col]), 4),
                    "market_morning": round(ll(part["p_bet"]), 4),
                    "model_same_games_as_close": round(ll2(sub[p_col]), 4), "market_close": round(ll2(sub["p_close"]), 4)})
    return out


def run(con) -> dict:
    from maxine import market
    built = {"line_markets": market.build_line_markets(con), "reg3_snapshots": market.build_reg3(con)}
    df = calibrate(rows(con))
    return {"built": built, "log_loss": log_loss_vs_close(df), "bets": summarize(bets(df), thresholds=(0.03, 0.05))}


def print_report(r: dict) -> None:
    print("built:", r["built"])
    print("\nlog loss on the morning main line (lower is better)")
    for x in r["log_loss"]:
        print(f"  {x['market']:<8} n={x['n']:<5} model {x['model']}  market morning {x['market_morning']}  "
              f"| same games: model {x['model_same_games_as_close']} vs close {x['market_close']}")
    print("\nbets by market, bet time, min edge (2023-26, 1u flat at median price)")
    for x in r["bets"]:
        print(f"  {x['market']:<8} {x['bet_time']:<8} >= {x['min_edge']:.0%}  bets {x['bets']:<5} "
              f"CLV {x['avg_clv'] if x['avg_clv'] is not None else '-':<8} moved our way "
              f"{x['clv_pos_pct'] if x['clv_pos_pct'] is not None else '-':<6} ROI {x['roi']:+.3f}  by season {x['seasons_roi']}")
