"""Best-chance parlays (3 to 8 legs) from today's board.

1. Candidate legs: both sides of every game's moneyline, total and puck line (main line, US books).
   Regulation 3-way is left out (European books; not parlayable with US books).
2. A leg is eligible only if the model rates it at least as likely as the market does (model >= no-vig
   market probability): the model never pays a price it thinks is too short.
3. Chance of each leg = model and market blended by leg type (blend.py: puck line underdog 60% model,
   moneyline 40%, puck line favourite 25%, totals 20%).
4. One leg per game (legs from the same game are correlated), the eligible leg with the highest blended chance.
5. An N-leg parlay = the N games whose best leg has the highest blended chance.
   Chance to hit = product of the leg chances (games are independent).
6. Price: every book that offers all N legs at the same lines; the one with the highest combined price.
   If no single book has all legs, the product of the best price per leg is shown instead (marked mixed).
"""
from __future__ import annotations

import math

import pandas as pd

from maxine.blend import blend

SIZES = range(3, 9)


def _label(market: str, side: str, home: str, away: str, line: float) -> str:
    if market == "h2h":
        return f"{home if side == 'a' else away} to win"
    if market == "spreads":
        return f"{home} {line:+.1f}" if side == "a" else f"{away} {-line:+.1f}"
    return f"{'Over' if side == 'a' else 'Under'} {line:g}"


def candidate_legs(inputs: list[dict]) -> pd.DataFrame:
    """inputs: game_id, home, away, start_utc, market, line, p_a (model), p_mkt_a (no-vig market)."""
    rows = []
    for x in inputs:
        for side in ("a", "b"):
            pm = x["p_a"] if side == "a" else 1 - x["p_a"]
            pk = x["p_mkt_a"] if side == "a" else 1 - x["p_mkt_a"]
            team_line = (x["line"] if side == "a" else -x["line"]) if x["market"] == "spreads" else 0.0
            rows.append({**{k: x[k] for k in ("game_id", "home", "away", "start_utc", "market", "line")},
                         "side": side, "p_model": pm, "p_market": pk,
                         "p_blend": blend(pm, pk, x["market"], team_line),
                         "selection": _label(x["market"], side, x["home"], x["away"], x["line"])})
    return pd.DataFrame(rows)


def best_leg_per_game(legs: pd.DataFrame) -> pd.DataFrame:
    ok = legs[legs["p_model"] >= legs["p_market"]]
    if ok.empty:
        return ok
    best = ok.loc[ok.groupby("game_id")["p_blend"].idxmax()]
    return best.sort_values("p_blend", ascending=False).reset_index(drop=True)


def _book_price(books: pd.DataFrame, leg) -> pd.Series:
    """Decimal price of this leg at each book quoting the same game, market and line."""
    m = books[(books["home"] == leg.home) & (books["away"] == leg.away) & (books["market"] == leg.market)
              & ((books["line"] - leg.line).abs() < 1e-9)]
    m = m[(m["commence"] - leg.start_utc).abs() < pd.Timedelta(hours=12)]
    return m.groupby("book")["a" if leg.side == "a" else "b"].max()


def build(inputs: list[dict], books: pd.DataFrame) -> list[dict]:
    legs = best_leg_per_game(candidate_legs(inputs))
    out = []
    if legs.empty:
        return out
    prices = [_book_price(books, leg) for leg in legs.itertuples()]
    for n in SIZES:
        if n > len(legs):
            break
        top, pr = legs.head(n), prices[:n]
        common = set.intersection(*(set(p.index) for p in pr))
        if common:
            book = max(common, key=lambda b: math.prod(p[b] for p in pr))
            leg_prices, mixed = [float(p[book]) for p in pr], False
            books_used = [book] * n
        else:
            leg_prices = [float(p.max()) if len(p) else float("nan") for p in pr]
            books_used = [p.idxmax() if len(p) else None for p in pr]
            book, mixed = None, True
        price = math.prod(leg_prices)
        p_hit = math.prod(top["p_blend"])
        out.append({
            "legs_n": n, "p_model": p_hit, "p_model_only": math.prod(top["p_model"]),
            "p_market": math.prod(top["p_market"]),
            "price": price, "book": book, "mixed_books": mixed, "ev": p_hit * price - 1,
            "legs": [{"game_id": int(r.game_id), "home": r.home, "away": r.away, "start_utc": str(r.start_utc),
                      "market": r.market, "selection": r.selection, "line": float(r.line),
                      "p_model": float(r.p_blend), "p_model_only": float(r.p_model), "p_market": float(r.p_market),
                      "price": lp, "book": bk}
                     for r, lp, bk in zip(top.itertuples(), leg_prices, books_used)],
        })
    return out
