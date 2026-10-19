"""Paper trading for early-line picks: store, capture the closing price, grade.

The bet is the first price seen for a game (best price at the evening snapshot). Close = the last
live snapshot taken before puck drop. CLV = no-vig probability of our side at close minus at bet.
Profit is at the best price we logged, 1 unit flat.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from maxine.ingest import odds
from maxine.picks import market_now
from maxine.web.appdb import db, insert_rows, query, utcnow

CLOSE_WINDOW_MIN = 30


def store_picks(picks: pd.DataFrame) -> int:
    if picks.empty:
        return 0
    rows = []
    for p in picks.itertuples():
        rows.append({
            "created_at": utcnow(), "game_id": int(p.game_id), "game_date": str(p.game_date),
            "start_utc": pd.Timestamp(p.start_utc).isoformat(), "home": p.home, "away": p.away,
            "side": p.side, "pick_team": p.pick_team, "p_model": float(p.p_model),
            "p_market": float(p.p_market), "edge": float(p.edge), "best_price": float(p.best_price),
            "best_book": p.best_book, "med_price": float(p.med_price), "books": int(p.books),
        })
    with db() as conn:
        before = conn.total_changes
        insert_rows(conn, "paper_picks", rows, on_conflict="ON CONFLICT(game_id) DO NOTHING")
        return conn.total_changes - before


def due_for_close(now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    soon = (now + timedelta(minutes=CLOSE_WINDOW_MIN)).isoformat()
    rows = query("SELECT * FROM paper_picks WHERE close_at IS NULL AND start_utc > ? AND start_utc <= ?",
                 (now.isoformat(), soon))
    return [dict(r) for r in rows]


def capture_closes(force: bool = False) -> dict:
    """Snapshot odds (3 credits) only if a picked game starts within the window, then record closes."""
    due = due_for_close() if not force else [dict(r) for r in query(
        "SELECT * FROM paper_picks WHERE result IS NULL AND start_utc > ?",
        (datetime.now(timezone.utc).isoformat(),))]
    if not due:
        return {"snapshots": 0, "closed": 0}
    snap = odds.snapshot("nhl")
    mk = market_now(snap)
    closed = 0
    with db() as conn:
        for p in due:
            m = mk[(mk["home"] == p["home"]) & (mk["away"] == p["away"])] if not mk.empty else mk
            if m.empty:
                continue
            m = m.iloc[0]
            home = p["side"] == "home"
            p_side = float(m["p_market"]) if home else 1 - float(m["p_market"])
            p_bet = p["p_market"] if home else 1 - p["p_market"]
            price = float(m["med_home_price"] if home else m["med_away_price"])
            conn.execute("UPDATE paper_picks SET close_p_side=?, close_price=?, close_at=?, clv=? WHERE id=?",
                         (p_side, price, utcnow(), p_side - p_bet, p["id"]))
            closed += 1
    return {"snapshots": 1, "closed": closed, "credits_remaining": snap["credits_remaining"]}


def grade(con) -> int:
    """Settle picks whose game is final in the warehouse."""
    open_ = [dict(r) for r in query("SELECT * FROM paper_picks WHERE result IS NULL")]
    if not open_:
        return 0
    ids = [p["game_id"] for p in open_]
    finals = {r[0]: (r[1], r[2]) for r in con.execute(
        f"SELECT game_id, home_score, away_score FROM games WHERE game_id IN ({','.join('?' * len(ids))})",
        ids).fetchall()}
    graded = 0
    with db() as conn:
        for p in open_:
            if p["game_id"] not in finals:
                continue
            hs, as_ = finals[p["game_id"]]
            won = (hs > as_) == (p["side"] == "home")
            profit = (p["best_price"] - 1) if won else -1.0
            conn.execute("UPDATE paper_picks SET result=?, profit=?, graded_at=? WHERE id=?",
                         ("W" if won else "L", profit, utcnow(), p["id"]))
            graded += 1
    return graded


def summary() -> dict:
    rows = [dict(r) for r in query("SELECT * FROM paper_picks ORDER BY start_utc DESC")]
    settled = [r for r in rows if r["result"]]
    with_clv = [r for r in rows if r["clv"] is not None]
    return {
        "picks": rows,
        "record": {"bets": len(settled), "wins": sum(r["result"] == "W" for r in settled),
                   "units": round(sum(r["profit"] for r in settled), 2),
                   "roi": round(sum(r["profit"] for r in settled) / len(settled), 4) if settled else None,
                   "avg_clv": round(sum(r["clv"] for r in with_clv) / len(with_clv), 4) if with_clv else None,
                   "beat_close_pct": round(sum(r["clv"] > 0 for r in with_clv) / len(with_clv), 3) if with_clv else None,
                   "open": len(rows) - len(settled)},
    }
