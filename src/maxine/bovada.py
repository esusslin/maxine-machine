"""Ladder: the most likely 2- to 5-leg Bovada parlay priced near -130.

1. Legs: every Bovada line for today's games: moneyline, main and alternate puck lines, main and alternate
   totals. Whole-number lines are skipped (they can push).
2. Chance of each leg from our models: moneyline from the win model; puck lines and totals from the goal
   model's score grid, calibrated per line (goals.line_calibrators), because one calibration per market is
   wrong far from the main line.
3. One leg per game (legs from the same game move together).
4. Every 2-, 3-, 4- and 5-game combination whose combined Bovada price is between TARGET_LONG and
   TARGET_SHORT (-120 to -145) is a candidate (exact search, with branch-and-bound pruning). At a near-fixed price, the highest chance to hit is also the best value.
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
SIZES = (2, 3, 4, 5)
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
    by_game = {gid: sorted(_frontier(g, hi), key=lambda l: -l.p_model) for gid, g in legs_df.groupby("game_id")}
    by_game = {k: v for k, v in by_game.items() if v}
    games_all = sorted(by_game, key=lambda g: -by_game[g][0].p_model)   # likeliest games first: finds good sets early
    top: list[tuple] = []                                              # best per game set, (p, combo, price), desc

    def threshold() -> float:
        return top[-1][0] if len(top) >= options else 0.0

    for n in sizes:
        for games in itertools.combinations(games_all, n):
            ls = [by_game[g] for g in games]
            max_p = [l[0].p_model for l in ls]
            if math.prod(max_p) <= threshold():
                continue                                               # this set cannot reach the top
            rest_p = [math.prod(max_p[i:]) for i in range(n + 1)]
            rest_lo = [math.prod(min(x.price for x in l) for l in ls[i:]) for i in range(n + 1)]
            rest_hi = [math.prod(max(x.price for x in l) for l in ls[i:]) for i in range(n + 1)]
            best = [0.0, None, None]

            def dfs(i, p, price, chosen):
                if i == n:
                    if lo <= price <= hi and p > best[0]:
                        best[:] = [p, tuple(chosen), price]
                    return
                for leg in ls[i]:                                     # sorted by chance, highest first
                    p2 = p * leg.p_model
                    if p2 * rest_p[i + 1] <= max(best[0], threshold()):
                        break                                          # every later leg is less likely
                    pr2 = price * leg.price
                    if pr2 * rest_lo[i + 1] > hi or pr2 * rest_hi[i + 1] < lo:
                        continue                                       # price can no longer land in the window
                    chosen.append(leg)
                    dfs(i + 1, p2, pr2, chosen)
                    chosen.pop()

            dfs(0, 1.0, 1.0, [])
            if best[1] is not None:
                top.append(tuple(best))
                top.sort(key=lambda t: -t[0])
                del top[options:]
    ranked = top
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
