"""Parse live odds snapshots into per-game market tables (median no-vig probability, median and best prices).

US feed: h2h (moneyline incl. OT/SO), spreads (puck line, keyed by the HOME line), totals.
EU feed (nhl_eu): h2h with a Draw outcome = regulation 3-way.
"""
from __future__ import annotations

import json

import pandas as pd

from maxine.teams import ODDS_NAME_TO_NHL


def _events(path: str):
    payload = json.loads(open(path).read())
    for ev in payload["data"]:
        home, away = ODDS_NAME_TO_NHL.get(ev["home_team"]), ODDS_NAME_TO_NHL.get(ev["away_team"])
        if home and away:
            yield ev, home, away, pd.to_datetime(ev["commence_time"], utc=True)


def two_way(path: str) -> pd.DataFrame:
    """Rows: home, away, commence, market (h2h/spreads/totals), line, p_a, books, med_a, med_b,
    best_a, best_a_book, best_b, best_b_book. Side A = home / home covers / over."""
    rows = []
    for ev, home, away, commence in _events(path):
        for b in ev.get("bookmakers", []):
            for m in b.get("markets", []):
                key = m["key"]
                if key not in ("h2h", "spreads", "totals"):
                    continue
                oc = {o["name"]: o for o in m["outcomes"]}
                if key == "h2h":
                    if "Draw" in oc:
                        continue
                    a, bb, line = oc.get(ev["home_team"]), oc.get(ev["away_team"]), 0.0
                elif key == "spreads":
                    a, bb = oc.get(ev["home_team"]), oc.get(ev["away_team"])
                    line = a.get("point") if a else None
                else:
                    a, bb = oc.get("Over"), oc.get("Under")
                    line = a.get("point") if a else None
                if not a or not bb or line is None or a["price"] <= 1 or bb["price"] <= 1:
                    continue
                pa, pb = a["price"], bb["price"]
                rows.append({"home": home, "away": away, "commence": commence, "market": key, "line": float(line),
                             "book": b["key"], "a": pa, "b": pb, "p": (1 / pa) / (1 / pa + 1 / pb)})
    return _aggregate(pd.DataFrame(rows), ["a", "b"])


def reg3(path: str) -> pd.DataFrame:
    """Rows: home, away, commence, p_home, p_draw, p_away, books, med_*, best_*, best_*_book."""
    rows = []
    for ev, home, away, commence in _events(path):
        for b in ev.get("bookmakers", []):
            for m in b.get("markets", []):
                if m["key"] != "h2h":
                    continue
                oc = {o["name"]: o["price"] for o in m["outcomes"]}
                hp, dp, ap = oc.get(ev["home_team"]), oc.get("Draw"), oc.get(ev["away_team"])
                if not (hp and dp and ap) or min(hp, dp, ap) <= 1:
                    continue
                tot = 1 / hp + 1 / dp + 1 / ap
                rows.append({"home": home, "away": away, "commence": commence, "market": "reg3", "line": 0.0,
                             "book": b["key"], "home_p": hp, "draw_p": dp, "away_p": ap,
                             "p_home": (1 / hp) / tot, "p_draw": (1 / dp) / tot, "p_away": (1 / ap) / tot})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    keys = ["home", "away", "commence"]
    agg = df.groupby(keys).agg(p_home=("p_home", "median"), p_draw=("p_draw", "median"), p_away=("p_away", "median"),
                               books=("book", "nunique"), med_home=("home_p", "median"), med_draw=("draw_p", "median"),
                               med_away=("away_p", "median")).reset_index()
    for o in ("home", "draw", "away"):
        best = df.loc[df.groupby(keys)[f"{o}_p"].idxmax(), keys + [f"{o}_p", "book"]]
        agg = agg.merge(best.rename(columns={f"{o}_p": f"best_{o}", "book": f"best_{o}_book"}), on=keys)
    return agg


def _aggregate(df: pd.DataFrame, sides) -> pd.DataFrame:
    if df.empty:
        return df
    keys = ["home", "away", "commence", "market", "line"]
    agg = df.groupby(keys).agg(p_a=("p", "median"), books=("book", "nunique"),
                               med_a=("a", "median"), med_b=("b", "median")).reset_index()
    for s in sides:
        best = df.loc[df.groupby(keys)[s].idxmax(), keys + [s, "book"]]
        agg = agg.merge(best.rename(columns={s: f"best_{s}", "book": f"best_{s}_book"}), on=keys)
    return agg


def main_lines(tw: pd.DataFrame) -> pd.DataFrame:
    """Keep, per game and market, the line the most books quote."""
    if tw.empty:
        return tw
    return (tw.sort_values("books", ascending=False)
              .groupby(["home", "away", "commence", "market"]).head(1).reset_index(drop=True))
