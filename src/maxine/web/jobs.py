"""Scheduled jobs (APScheduler, in-process, one at a time). Times are America/Los_Angeles.

    03:30  results + xG + grade paper picks
    10:00  BetNHL daily tips snapshot
    09:30  picks: today's games on the morning line (game-day picks, if GAME_DAY_PICKS) + tomorrow
    18:30  picks: tomorrow's early line (+ any of today's games not started, if GAME_DAY_PICKS)
    every 15 min, 09:00-20:45  closing-price capture, only when a picked game is about to start
    hourly heartbeat
First boot with an empty warehouse: backfill five NHL seasons from the free NHL API (about 2 hours).
"""
from __future__ import annotations

import json
import logging
import math
from datetime import date

from maxine import paper, picks
from maxine.ingest import betnhl, nhl
from maxine.models import xg
from maxine.warehouse import load
from maxine.warehouse.db import connect
from maxine.web.appdb import job_run, set_kv
from maxine.web.settings import settings

log = logging.getLogger(__name__)
BOOTSTRAP_SEASONS = [20212022, 20222023, 20232024, 20242025, 20252026]


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if hasattr(v, "item"):
        return v.item()
    return v


def heartbeat() -> None:
    with job_run("heartbeat"):
        pass


def warehouse_games() -> int:
    con = connect()
    try:
        return con.execute("SELECT count(*) FROM games").fetchone()[0]
    finally:
        con.close()


def bootstrap() -> None:
    """Fill an empty volume from the NHL API. Safe to re-run: downloaded files are skipped."""
    with job_run("bootstrap") as ctx:
        total = 0
        for season in BOOTSTRAP_SEASONS:
            total += nhl.ingest_season(season, with_pbp=True)
            log.info("bootstrap: season %s done (%s games so far)", season, total)
        con = connect()
        try:
            ctx["note"] = json.dumps({"load": load.load_nhl(con), "xg": xg.score_shots(con)["shots_scored"]})
        finally:
            con.close()
        ctx["rows_affected"] = total


def results_and_grade() -> None:
    with job_run("results_and_grade") as ctx:
        con = connect()
        try:
            status = picks.catch_up(con)
            graded = paper.grade(con)
        finally:
            con.close()
        ctx["rows_affected"] = graded
        ctx["note"] = json.dumps({"catch_up": status, "graded": graded})


def run_picks() -> dict:
    with job_run("picks") as ctx:
        con = connect()
        try:
            r = picks.run(con, hours=36, min_edge=settings.MIN_EDGES, game_day=settings.GAME_DAY_PICKS,
                          reg3_feed=settings.REG3_FEED)
        finally:
            con.close()
        board = r["board"]
        cols = ["game_id", "game_date", "start_utc", "home", "away", "market", "selection", "line", "side",
                "p_model", "p_market", "edge", "min_edge", "best_price", "best_book", "med_price", "books",
                "clean", "early", "line_type", "validated", "is_pick", "home_goalie", "away_goalie", "covered_by"]
        records = [] if board.empty else [{c: _clean(row.get(c)) for c in cols}
                                          for row in board.to_dict("records")]
        set_kv("board", json.dumps({"records": records, "min_edges": settings.MIN_EDGES,
                                    "game_day_picks": settings.GAME_DAY_PICKS}))
        if r.get("credits_remaining") is not None:
            set_kv("odds_credits_remaining", str(r["credits_remaining"]))
        stored = paper.store_picks(r["picks"]) if not r["picks"].empty else 0
        ctx["rows_affected"] = stored
        ctx["note"] = json.dumps({"board": len(records), "new_picks": stored})
        return {"board": len(records), "new_picks": stored}


def capture_closes(force: bool = False) -> dict:
    with job_run("capture_closes") as ctx:
        r = paper.capture_closes(force=force)
        if r.get("credits_remaining") is not None:
            set_kv("odds_credits_remaining", str(r["credits_remaining"]))
        ctx["rows_affected"] = r["closed"]
        ctx["note"] = json.dumps(r)
        return r


def betnhl_snapshot() -> dict:
    with job_run("betnhl_snapshot") as ctx:
        r = betnhl.snapshot(date.today())
        con = connect()
        try:
            n = load.load_betnhl(con)
        finally:
            con.close()
        ctx["rows_affected"] = r["picks"]
        ctx["note"] = json.dumps({**r, "loaded_total": n})
        return r


JOBS = [
    (results_and_grade, "cron", dict(hour=3, minute=30), "results_and_grade"),
    (betnhl_snapshot, "cron", dict(hour=10, minute=0), "betnhl_snapshot"),
    (run_picks, "cron", dict(hour=9, minute=30), "picks_morning"),
    (run_picks, "cron", dict(hour=18, minute=30), "picks_evening"),
    (capture_closes, "cron", dict(hour="9-20", minute="*/15"), "capture_closes"),
    (heartbeat, "interval", dict(minutes=60), "heartbeat"),
]
