"""Bovada parlay finder: the most reliable 2- to 4-leg parlay priced near -130.

1. Legs: every Bovada line for today's games: moneyline, main and alternate puck lines, main and alternate
   totals. Whole-number lines are skipped (they can push).
2. Chance of each leg from our models: moneyline from the win model; puck lines and totals from the goal
   model's score grid, calibrated per line (goals.line_calibrators), because one calibration per market is
   wrong far from the main line.
3. One leg per game (legs from the same game move together).
4. Every 2-, 3- and 4-game combination whose combined Bovada price is between TARGET_LONG and TARGET_SHORT
   (-120 to -145) is a candidate. At a near-fixed price, the highest chance to hit is also the best value.
5. Options 1-3: the highest chance to hit, each on a different set of games.
"""
from __future__ import annotations

import itertools
import math

import pandas as pd

from maxine.models import goals
from maxine.teams import ODDS_NAME_TO_NHL

TARGET_SHORT = -145     # shortest combined price allowed (American)
TARGET_LONG = -120      # longest combined price allowed
SIZES = (2, 3, 4)
OPTIONS = 3


def american_to_decimal(a: float) -> float:
    return 1 + (100 / -a if a < 0 else a / 100)


def decimal_to_american(d: float) -> int:
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def legs(payload: dict, fixtures: pd.DataFrame, p_ml: dict, prices: dict, line_cals: dict) -> pd.DataFrame:
    """One row per Bovada line: game_id, home, away, start_utc, kind, selection, line, price, p_model, p_book."""
    fx = {(f.home, f.away): f for f in fixtures.itertuples()}
    rows = []
    for ev in payload.get("events", []):
        home, away = ODDS_NAME_TO_NHL.get(ev.get("home_team")), ODDS_NAME_TO_NHL.get(ev.get("away_team"))
        f = fx.get((home, away))
        if f is None:
            continue
        pr = prices.get(f.game_id)
        base = {"game_id": f.game_id, "home": home, "away": away, "start_utc": f.start_utc}
        for bk in ev.get("bookmakers", []):
            if bk.get("key") != "bovada":
                continue
            for m in bk.get("markets", []):
                key = m["key"]
                for o in m.get("outcomes", []):
                    price, point = o.get("price"), o.get("point")
                    if not price or price <= 1:
                        continue
                    if key == "h2h":
                        if f.game_id not in p_ml or o["name"] not in (ev["home_team"], ev["away_team"]):
                            continue
                        is_home = o["name"] == ev["home_team"]
                        p = p_ml[f.game_id] if is_home else 1 - p_ml[f.game_id]
                        rows.append({**base, "kind": "moneyline", "selection": f"{home if is_home else away} to win",
                                     "line": 0.0, "side": "home" if is_home else "away", "price": price, "p_model": p})
                    elif key in ("spreads", "alternate_spreads") and pr is not None and point is not None:
                        if float(point) % 1 == 0 or o["name"] not in (ev["home_team"], ev["away_team"]):
                            continue
                        is_home = o["name"] == ev["home_team"]
                        home_line = float(point) if is_home else -float(point)
                        c, l_, _ = goals.prob_spread(pr, home_line)
                        p_home = goals.apply_cal(c / (c + l_), goals.line_cal_for(line_cals, "spread", home_line))
                        team = home if is_home else away
                        rows.append({**base, "kind": "puck line", "selection": f"{team} {float(point):+.1f}",
                                     "line": float(point), "side": "home" if is_home else "away", "price": price,
                                     "p_model": p_home if is_home else 1 - p_home})
                    elif key in ("totals", "alternate_totals") and pr is not None and point is not None:
                        if float(point) % 1 == 0 or o["name"] not in ("Over", "Under"):
                            continue
                        ov, un, _ = goals.prob_total(pr, float(point))
                        p_over = goals.apply_cal(ov / (ov + un), goals.line_cal_for(line_cals, "total", float(point)))
                        rows.append({**base, "kind": "total", "selection": f"{o['name']} {float(point):g}",
                                     "line": float(point), "side": o["name"].lower(), "price": price,
                                     "p_model": p_over if o["name"] == "Over" else 1 - p_over})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("price", ascending=False).drop_duplicates(["game_id", "selection"])
    # Bovada's own no-margin chance, where the opposite side of the same line is listed
    opp = {"home": "away", "away": "home", "over": "under", "under": "over"}
    idx = {(r.game_id, r.kind, r.side, r.line): r.price for r in df.itertuples()}
    p_book = []
    for r in df.itertuples():
        other_line = -r.line if r.kind == "puck line" else r.line
        q = idx.get((r.game_id, r.kind, opp[r.side], other_line))
        p_book.append((1 / r.price) / (1 / r.price + 1 / q) if q else None)
    df["p_book"] = p_book
    return df.reset_index(drop=True)


def _frontier(g: pd.DataFrame, max_price: float) -> list:
    """Legs worth considering in one game: no other leg in the game pays more AND is more likely."""
    g = g[g["price"] < max_price].sort_values(["price", "p_model"], ascending=[False, False])
    keep, best_p = [], -1.0
    for r in g.itertuples():
        if r.p_model > best_p:
            keep.append(r)
            best_p = r.p_model
    return keep


def search(legs_df: pd.DataFrame, short: int = TARGET_SHORT, long: int = TARGET_LONG,
           sizes=SIZES, options: int = OPTIONS) -> list[dict]:
    if legs_df is None or legs_df.empty:
        return []
    lo, hi = american_to_decimal(short), american_to_decimal(long)      # e.g. 1.690 .. 1.833
    by_game = {gid: _frontier(g, hi) for gid, g in legs_df.groupby("game_id")}
    by_game = {k: v for k, v in by_game.items() if v}
    best = {}                                                          # game set -> (p, legs, price)
    for n in sizes:
        for games in itertools.combinations(sorted(by_game), n):
            for combo in itertools.product(*(by_game[g] for g in games)):
                price = math.prod(l.price for l in combo)
                if not lo <= price <= hi:
                    continue
                p = math.prod(l.p_model for l in combo)
                if p > best.get(games, (0,))[0]:
                    best[games] = (p, combo, price)
    ranked = sorted(best.values(), key=lambda t: -t[0])[:options]
    out = []
    for rank, (p, combo, price) in enumerate(ranked, 1):
        p_book = [None if l.p_book is None or pd.isna(l.p_book) else float(l.p_book) for l in combo]
        out.append({
            "rank": rank, "legs_n": len(combo), "price": price, "american": decimal_to_american(price),
            "p_model": p, "breakeven": 1 / price, "ev": p * price - 1,
            "p_book": math.prod(p_book) if all(x is not None for x in p_book) else None,
            "legs": [{"game_id": int(l.game_id), "home": l.home, "away": l.away, "start_utc": str(l.start_utc),
                      "kind": l.kind, "selection": l.selection, "price": float(l.price),
                      "american": decimal_to_american(l.price), "p_model": float(l.p_model),
                      "p_book": None if l.p_book is None or pd.isna(l.p_book) else float(l.p_book)}
                     for l in combo],
        })
    return out
