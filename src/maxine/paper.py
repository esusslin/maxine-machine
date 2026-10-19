"""Paper trading for every market: store picks, capture the closing price, grade.

One bet per game per market; the first price seen is the bet (best price at that moment), 1 unit flat.
Close = the last live snapshot within CLOSE_WINDOW_MIN of puck drop, same market AND same line.
CLV = no-vig probability of our side at close minus when we bet.
Settlement:
  h2h       final result (OT/SO included)
  totals    final total (books credit the OT/SO winner one goal) vs line; whole-number lines can push
  spreads   final margin + home line (side a = home covers, b = away covers); can push on whole numbers
  reg3      regulation goals only (periods 1-3): home / draw / away
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from maxine import live_markets
from maxine.ingest import odds
from maxine.web.appdb import db, insert_rows, query, utcnow

CLOSE_WINDOW_MIN = 30
MARKETS = ("h2h", "totals", "spreads", "reg3")


def store_picks(picks: pd.DataFrame) -> int:
    if picks is None or picks.empty:
        return 0
    rows = []
    for p in picks.to_dict("records"):
        rows.append({
            "created_at": utcnow(), "game_id": int(p["game_id"]), "market": p.get("market", "h2h"),
            "line_type": p.get("line_type", "early"), "validated": int(bool(p.get("validated", True))),
            "game_date": str(p["game_date"]), "start_utc": pd.Timestamp(p["start_utc"]).isoformat(),
            "home": p["home"], "away": p["away"],
            "side": p.get("side") or ("a" if p.get("selection") == p["home"] else "b"),
            "selection": p.get("selection") or p.get("pick_team"), "line": float(p.get("line") or 0.0),
            "p_model": float(p["p_model"]), "p_market": float(p["p_market"]), "edge": float(p["edge"]),
            "best_price": float(p["best_price"]), "best_book": p["best_book"],
            "med_price": float(p["med_price"]), "books": int(p["books"]),
        })
    with db() as conn:
        before = conn.total_changes
        insert_rows(conn, "paper_bets", rows, on_conflict="ON CONFLICT(game_id, market) DO NOTHING")
        return conn.total_changes - before


def due_for_close(now: datetime | None = None, force: bool = False) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    if force:
        rows = query("SELECT * FROM paper_bets WHERE result IS NULL AND start_utc > ?", (now.isoformat(),))
    else:
        soon = (now + timedelta(minutes=CLOSE_WINDOW_MIN)).isoformat()
        rows = query("SELECT * FROM paper_bets WHERE close_at IS NULL AND start_utc > ? AND start_utc <= ?",
                     (now.isoformat(), soon))
    return [dict(r) for r in rows]


def _side_close(bet: dict, tw: pd.DataFrame, r3: pd.DataFrame) -> tuple[float, float] | None:
    """(no-vig probability of our side at close, median price of our side) or None if not quoted."""
    if bet["market"] == "reg3":
        m = r3[(r3["home"] == bet["home"]) & (r3["away"] == bet["away"])] if not r3.empty else r3
        if m.empty:
            return None
        m = m.iloc[0]
        return float(m[f"p_{bet['side']}"]), float(m[f"med_{bet['side']}"])
    m = tw[(tw["home"] == bet["home"]) & (tw["away"] == bet["away"]) & (tw["market"] == bet["market"])] \
        if not tw.empty else tw
    if bet["market"] != "h2h":
        m = m[(m["line"] - bet["line"]).abs() < 1e-9]
    if m.empty:
        return None
    m = m.sort_values("books", ascending=False).iloc[0]
    if bet["side"] == "a":
        return float(m["p_a"]), float(m["med_a"])
    return float(1 - m["p_a"]), float(m["med_b"])


def capture_closes(force: bool = False) -> dict:
    """Snapshot odds only if a picked game starts soon (US feed 3 credits; EU feed 1 more if a reg3 bet is due)."""
    due = due_for_close(force=force)
    if not due:
        return {"snapshots": 0, "closed": 0}
    snap = odds.snapshot("nhl")
    tw = live_markets.two_way(snap["path"])
    credits, snaps, r3 = snap["credits_remaining"], 1, pd.DataFrame()
    if any(b["market"] == "reg3" for b in due):
        snap_eu = odds.snapshot("nhl_eu")
        r3, credits, snaps = live_markets.reg3(snap_eu["path"]), snap_eu["credits_remaining"], 2
    closed = 0
    with db() as conn:
        for b in due:
            c = _side_close(b, tw, r3)
            if c is None:
                continue
            p_close, price = c
            conn.execute("UPDATE paper_bets SET close_p_side=?, close_price=?, close_at=?, clv=? WHERE id=?",
                         (p_close, price, utcnow(), p_close - b["p_market"], b["id"]))
            closed += 1
    return {"snapshots": snaps, "closed": closed, "credits_remaining": credits}


def settle(bet: dict, hs: int, as_: int, h_reg: int, a_reg: int) -> str:
    """W / L / P for one bet given final and regulation scores."""
    m, side, line = bet["market"], bet["side"], bet["line"] or 0.0
    if m == "h2h":
        return "W" if (hs > as_) == (side == "a") else "L"
    if m == "reg3":
        actual = "home" if h_reg > a_reg else ("away" if h_reg < a_reg else "draw")
        return "W" if actual == side else "L"
    v = (hs + as_ - line) if m == "totals" else (hs - as_ + line)
    if v == 0:
        return "P"
    return "W" if (v > 0) == (side == "a") else "L"


def grade(con) -> int:
    """Settle bets whose game is final in the warehouse."""
    open_ = [dict(r) for r in query("SELECT * FROM paper_bets WHERE result IS NULL")]
    if not open_:
        return 0
    ids = sorted({b["game_id"] for b in open_})
    ph = ",".join("?" * len(ids))
    finals = {r[0]: r[1:] for r in con.execute(f"""
        SELECT g.game_id, g.home_score, g.away_score,
               sum(CASE WHEN s.event_type='goal' AND s.period<=3 AND s.owner_team_id=g.home_team_id THEN 1 ELSE 0 END),
               sum(CASE WHEN s.event_type='goal' AND s.period<=3 AND s.owner_team_id=g.away_team_id THEN 1 ELSE 0 END)
        FROM games g LEFT JOIN shots s USING (game_id) WHERE g.game_id IN ({ph}) GROUP BY ALL""", ids).fetchall()}
    graded = 0
    with db() as conn:
        for b in open_:
            if b["game_id"] not in finals:
                continue
            hs, as_, h_reg, a_reg = finals[b["game_id"]]
            res = settle(b, hs, as_, int(h_reg or 0), int(a_reg or 0))
            profit = 0.0 if res == "P" else ((b["best_price"] - 1) if res == "W" else -1.0)
            conn.execute("UPDATE paper_bets SET result=?, profit=?, graded_at=? WHERE id=?",
                         (res, profit, utcnow(), b["id"]))
            graded += 1
    return graded


def _record(rows: list[dict]) -> dict:
    settled = [r for r in rows if r["result"]]
    decided = [r for r in settled if r["result"] != "P"]
    with_clv = [r for r in rows if r["clv"] is not None]
    units = sum(r["profit"] for r in settled)
    return {"bets": len(settled), "wins": sum(r["result"] == "W" for r in decided),
            "losses": sum(r["result"] == "L" for r in decided), "pushes": len(settled) - len(decided),
            "units": round(units, 2), "roi": round(units / len(settled), 4) if settled else None,
            "avg_clv": round(sum(r["clv"] for r in with_clv) / len(with_clv), 4) if with_clv else None,
            "beat_close_pct": round(sum(r["clv"] > 0 for r in with_clv) / len(with_clv), 3) if with_clv else None,
            "open": len(rows) - len(settled)}


def summary() -> dict:
    rows = [dict(r) for r in query("SELECT * FROM paper_bets ORDER BY start_utc DESC, market")]
    by = {}
    for m in MARKETS:
        for lt in ("early", "game-day"):
            sub = [r for r in rows if r["market"] == m and (r["line_type"] or "early") == lt]
            if sub:
                by[f"{m}|{lt}"] = _record(sub)
    return {"picks": rows, "record": _record(rows), "by_market": by,
            "by_line": {lt: _record([r for r in rows if (r["line_type"] or "early") == lt])
                        for lt in ("early", "game-day")}}
