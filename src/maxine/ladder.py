"""Ladder: the most likely 2- to 5-leg parlay priced near -130, at one book.

1. Lines: main and alternate moneyline, puck lines and totals for today's games from LINE_BOOKS (one request
   per game; up to 10 books cost the same as one). Whole-number lines are skipped (they can push).
2. Shared lines only: a line (game, market, side, number) is used only if at least SHARED_MIN_BOOKS books and
   at least SHARED_SHARE of the books quoting that game offer it. Exotic lines one book offers (e.g. a 9.5
   total) are dropped.
3. Chance of each leg: the model (moneyline from the win model; puck lines and totals from the goal model's
   score grid, calibrated per line) blended with the market by leg type (blend.py: puck line underdog 60%
   model, moneyline 40%, puck line favourite 25%, totals 20%). Market = median no-margin chance of that exact
   line across the books listing both sides; if none does, 1/price less a 5% margin.
4. For each book you bet at (BET_BOOKS): one leg per game, every 2- to 5-game combination whose combined price
   at that book is between TARGET_LONG and TARGET_SHORT (-130 to -145); exact search with pruning.
5. Options 1-3: the highest chance to hit across those books, each on a different set of games. Each option
   also shows the price of the same legs at the other bet books that offer them all.
At a near-fixed price, the highest chance to hit is also the best value.
"""
from __future__ import annotations

import itertools
import math

import pandas as pd

from maxine.blend import blend
from maxine.models import goals
from maxine.teams import ODDS_NAME_TO_NHL

BET_BOOKS = ("bovada", "draftkings")                                   # where parlays are placed
LINE_BOOKS = ("bovada", "draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet")   # who votes on "shared"
SHARED_MIN_BOOKS = 3
SHARED_SHARE = 0.5
TARGET_SHORT = -145     # shortest combined price allowed (American)
TARGET_LONG = -130      # longest combined price allowed
SIZES = (2, 3, 4, 5)
OPTIONS = 3


def american_to_decimal(a: float) -> float:
    return 1 + (100 / -a if a < 0 else a / 100)


def decimal_to_american(d: float) -> int:
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def legs(payload: dict, fixtures: pd.DataFrame, p_ml: dict, prices: dict, line_cals: dict) -> pd.DataFrame:
    """One row per book per line: book, game_id, home, away, start_utc, kind, selection, side, line, price,
    p_model, p_book (that book's no-margin chance, when it lists the other side), books_offering, shared."""
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
            book = bk.get("key")
            for m in bk.get("markets", []):
                key = m["key"]
                for o in m.get("outcomes", []):
                    price, point = o.get("price"), o.get("point")
                    if not price or price <= 1:
                        continue
                    row = None
                    if key == "h2h":
                        if f.game_id in p_ml and o["name"] in (ev["home_team"], ev["away_team"]):
                            is_home = o["name"] == ev["home_team"]
                            row = {"kind": "moneyline", "selection": f"{home if is_home else away} to win", "line": 0.0,
                                   "side": "home" if is_home else "away",
                                   "p_model": p_ml[f.game_id] if is_home else 1 - p_ml[f.game_id]}
                    elif key in ("spreads", "alternate_spreads") and pr is not None and point is not None:
                        if float(point) % 1 != 0 and o["name"] in (ev["home_team"], ev["away_team"]):
                            is_home = o["name"] == ev["home_team"]
                            home_line = float(point) if is_home else -float(point)
                            c, l_, _ = goals.prob_spread(pr, home_line)
                            p_home = goals.apply_cal(c / (c + l_), goals.line_cal_for(line_cals, "spread", home_line))
                            row = {"kind": "puck line", "selection": f"{home if is_home else away} {float(point):+.1f}",
                                   "line": float(point), "side": "home" if is_home else "away",
                                   "p_model": p_home if is_home else 1 - p_home}
                    elif key in ("totals", "alternate_totals") and pr is not None and point is not None:
                        if float(point) % 1 != 0 and o["name"] in ("Over", "Under"):
                            ov, un, _ = goals.prob_total(pr, float(point))
                            p_over = goals.apply_cal(ov / (ov + un), goals.line_cal_for(line_cals, "total", float(point)))
                            row = {"kind": "total", "selection": f"{o['name']} {float(point):g}", "line": float(point),
                                   "side": o["name"].lower(), "p_model": p_over if o["name"] == "Over" else 1 - p_over}
                    if row:
                        rows.append({**base, **row, "book": book, "price": float(price)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("price", ascending=False).drop_duplicates(["book", "game_id", "selection"])
    # each book's own no-margin chance, where it lists the other side of the same line
    opp = {"home": "away", "away": "home", "over": "under", "under": "over"}
    idx = {(r.book, r.game_id, r.kind, r.side, r.line): r.price for r in df.itertuples()}
    p_book = []
    for r in df.itertuples():
        q = idx.get((r.book, r.game_id, r.kind, opp[r.side], -r.line if r.kind == "puck line" else r.line))
        p_book.append((1 / r.price) / (1 / r.price + 1 / q) if q else None)
    df["p_book"] = p_book
    # shared lines: offered by enough of the books quoting that game
    offering = df.groupby(["game_id", "selection"])["book"].nunique().rename("books_offering")
    quoting = df.groupby("game_id")["book"].nunique().rename("books_quoting")
    df = df.join(offering, on=["game_id", "selection"]).join(quoting, on="game_id")
    df["shared"] = (df["books_offering"] >= SHARED_MIN_BOOKS) & (df["books_offering"] >= SHARED_SHARE * df["books_quoting"])
    # market chance of each exact line: median of the books' no-margin chances; fallback 1/price less 5% margin
    cons = df.groupby(["game_id", "selection"])["p_book"].median().rename("p_market")
    df = df.join(cons, on=["game_id", "selection"])
    df["p_market"] = df["p_market"].fillna(1 / df["price"] / 1.05)
    df["p_use"] = [blend(m, k, kind, line if kind == "puck line" else 0.0)
                   for m, k, kind, line in zip(df["p_model"], df["p_market"], df["kind"], df["line"])]
    return df.reset_index(drop=True)


def _frontier(g: pd.DataFrame, max_price: float) -> list:
    """Legs worth considering in one game at one book: no other leg pays more AND is more likely."""
    g = g[g["price"] < max_price].sort_values(["price", "p_use"], ascending=[False, False])
    keep, best_p = [], -1.0
    for r in g.itertuples():
        if r.p_use > best_p:
            keep.append(r)
            best_p = r.p_use
    return keep


def _search_book(legs_df: pd.DataFrame, lo: float, hi: float, sizes, options: int) -> list[tuple]:
    """Top `options` (p, combo, price) at one book, each on a different set of games."""
    by_game = {gid: sorted(_frontier(g, hi), key=lambda l: -l.p_use) for gid, g in legs_df.groupby("game_id")}
    by_game = {k: v for k, v in by_game.items() if v}
    games_all = sorted(by_game, key=lambda g: -by_game[g][0].p_use)   # likeliest games first: good sets early
    top: list[tuple] = []

    def threshold() -> float:
        return top[-1][0] if len(top) >= options else 0.0

    for n in sizes:
        for games in itertools.combinations(games_all, n):
            ls = [by_game[g] for g in games]
            max_p = [l[0].p_use for l in ls]
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
                    p2 = p * leg.p_use
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
    return top


def search(legs_df: pd.DataFrame, bet_books=BET_BOOKS, short: int = TARGET_SHORT, long: int = TARGET_LONG,
           sizes=SIZES, options: int = OPTIONS) -> list[dict]:
    if legs_df is None or legs_df.empty:
        return []
    lo, hi = american_to_decimal(short), american_to_decimal(long)      # e.g. 1.690 .. 1.833
    shared = legs_df[legs_df["shared"]]
    found = []
    for book in bet_books:
        for p, combo, price in _search_book(shared[shared["book"] == book], lo, hi, sizes, options):
            found.append((p, combo, price, book))
    # one option per set of games: the book where that set is most likely within the window
    best_by_set = {}
    for p, combo, price, book in found:
        key = tuple(sorted(l.game_id for l in combo))
        if p > best_by_set.get(key, (0,))[0]:
            best_by_set[key] = (p, combo, price, book)
    ranked = sorted(best_by_set.values(), key=lambda t: -t[0])[:options]
    price_at = {(r.book, r.game_id, r.selection): r.price for r in shared.itertuples()}
    out = []
    for rank, (p, combo, price, book) in enumerate(ranked, 1):
        p_book = [None if l.p_book is None or pd.isna(l.p_book) else float(l.p_book) for l in combo]
        elsewhere = {}
        for other in bet_books:
            if other == book:
                continue
            ps = [price_at.get((other, l.game_id, l.selection)) for l in combo]
            if all(ps):
                elsewhere[other] = decimal_to_american(math.prod(ps))
        out.append({
            "rank": rank, "book": book, "legs_n": len(combo), "price": price, "american": decimal_to_american(price),
            "p_model": p, "p_model_only": math.prod(l.p_model for l in combo),
            "breakeven": 1 / price, "ev": p * price - 1,
            "p_book": math.prod(p_book) if all(x is not None for x in p_book) else None,
            "elsewhere": elsewhere,
            "legs": [{"game_id": int(l.game_id), "home": l.home, "away": l.away, "start_utc": str(l.start_utc),
                      "kind": l.kind, "selection": l.selection, "price": float(l.price),
                      "american": decimal_to_american(l.price), "p_model": float(l.p_use),
                      "p_model_only": float(l.p_model), "p_market": float(l.p_market),
                      "books_offering": int(l.books_offering),
                      "p_book": None if l.p_book is None or pd.isna(l.p_book) else float(l.p_book)}
                     for l in combo],
        })
    return out
