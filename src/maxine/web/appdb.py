"""SQLite for app state (users, invites, resets, paper picks, job runs). Same layer as the-algo.

Hockey data lives in DuckDB (maxine.warehouse); this file holds only what the web app owns.
WAL mode; migrations run once each and are recorded. Single replica only.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Sequence

from maxine.web.settings import settings

log = logging.getLogger(__name__)
_BUSY_TIMEOUT_MS = 15_000


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.APP_DB_PATH, timeout=_BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    return conn


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def query(sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
    with db() as conn:
        return conn.execute(sql, params).fetchall()


def query_one(sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
    with db() as conn:
        return conn.execute(sql, params).fetchone()


def insert_row(conn: sqlite3.Connection, table: str, row: dict[str, Any], on_conflict: str = "") -> int:
    cols = list(row.keys())
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) {on_conflict}"
    cur = conn.execute(sql, [row[c] for c in cols])
    return cur.lastrowid or 0


def insert_rows(conn: sqlite3.Connection, table: str, rows: Iterable[dict[str, Any]], on_conflict: str = "") -> int:
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0].keys())
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) {on_conflict}"
    conn.executemany(sql, [[r.get(c) for c in cols] for r in rows])
    return len(rows)


def set_kv(key: str, value: str) -> None:
    with db() as conn:
        conn.execute("INSERT INTO kv (key, value, updated_at) VALUES (?,?,?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                     (key, value, utcnow()))


def get_kv(key: str) -> sqlite3.Row | None:
    return query_one("SELECT * FROM kv WHERE key=?", (key,))


@contextmanager
def job_run(job_name: str) -> Iterator[dict[str, Any]]:
    """Wrap a scheduled job so success/failure is recorded for /health."""
    started = utcnow()
    t0 = time.monotonic()
    ctx: dict[str, Any] = {"rows_affected": 0}
    status, error = "success", None
    try:
        yield ctx
    except Exception as exc:  # noqa: BLE001 — every failure recorded
        status, error = "error", f"{type(exc).__name__}: {exc}"
        log.exception("job %s failed", job_name)
        raise
    finally:
        with db() as conn:
            insert_row(conn, "job_runs", {
                "job_name": job_name, "started_at": started, "finished_at": utcnow(),
                "duration_s": round(time.monotonic() - t0, 3), "status": status,
                "rows_affected": int(ctx.get("rows_affected") or 0),
                "note": ctx.get("note"), "error": error,
            })


def last_run(job_name: str, status: str | None = None) -> sqlite3.Row | None:
    if status:
        return query_one("SELECT * FROM job_runs WHERE job_name=? AND status=? ORDER BY id DESC LIMIT 1",
                         (job_name, status))
    return query_one("SELECT * FROM job_runs WHERE job_name=? ORDER BY id DESC LIMIT 1", (job_name,))


MIGRATIONS: list[tuple[int, str]] = [
    (1, """
    CREATE TABLE IF NOT EXISTS users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT UNIQUE NOT NULL,
        phone         TEXT,
        password_hash TEXT NOT NULL,
        role          TEXT DEFAULT 'user',
        tier          TEXT DEFAULT 'member',
        sms_opt_in    INTEGER DEFAULT 0,
        created_at    TEXT,
        last_login    TEXT
    );
    CREATE TABLE IF NOT EXISTS invites (
        code        TEXT PRIMARY KEY,
        created_by  INTEGER,
        note        TEXT,
        created_at  TEXT,
        expires_at  TEXT,
        used_by     INTEGER,
        used_at     TEXT,
        revoked     INTEGER DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_invites_open ON invites(used_by, revoked);
    CREATE TABLE IF NOT EXISTS invite_sends (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        code       TEXT, channel TEXT, recipient TEXT, status TEXT, error TEXT, sent_at TEXT
    );
    CREATE TABLE IF NOT EXISTS password_resets (
        token       TEXT PRIMARY KEY,
        user_id     INTEGER NOT NULL,
        created_at  TEXT, expires_at TEXT, used_at TEXT, issued_by TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_resets_user ON password_resets(user_id, used_at);
    CREATE TABLE IF NOT EXISTS job_runs (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        job_name      TEXT NOT NULL,
        started_at    TEXT NOT NULL,
        finished_at   TEXT,
        duration_s    REAL,
        status        TEXT NOT NULL,
        rows_affected INTEGER DEFAULT 0,
        note          TEXT,
        error         TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_jobruns_name ON job_runs(job_name, id DESC);
    CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
    -- Paper trades at the early line. One per game: the first price we saw is the bet.
    CREATE TABLE IF NOT EXISTS paper_picks (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at   TEXT NOT NULL,
        game_id      INTEGER UNIQUE NOT NULL,
        game_date    TEXT, start_utc TEXT, home TEXT, away TEXT,
        side         TEXT, pick_team TEXT,
        p_model      REAL, p_market REAL, edge REAL,
        best_price   REAL, best_book TEXT, med_price REAL, books INTEGER,
        close_p_side REAL, close_price REAL, close_at TEXT, clv REAL,
        result       TEXT, profit REAL, graded_at TEXT
    );
    """),
    (2, """
    -- early (day-before line, backtested edge) vs game-day (no backtested edge): separate records
    ALTER TABLE paper_picks ADD COLUMN line_type TEXT DEFAULT 'early';
    """),
    (3, """
    -- Every market. One bet per game per market; the first price seen is the bet.
    -- side: a/b for two-way markets (a = home / home covers / over), home/draw/away for reg3.
    CREATE TABLE IF NOT EXISTS paper_bets (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at   TEXT NOT NULL,
        game_id      INTEGER NOT NULL,
        market       TEXT NOT NULL,
        line_type    TEXT,
        validated    INTEGER,
        game_date    TEXT, start_utc TEXT, home TEXT, away TEXT,
        side         TEXT, selection TEXT, line REAL,
        p_model      REAL, p_market REAL, edge REAL,
        best_price   REAL, best_book TEXT, med_price REAL, books INTEGER,
        close_p_side REAL, close_price REAL, close_at TEXT, clv REAL,
        result       TEXT, profit REAL, graded_at TEXT,
        UNIQUE (game_id, market)
    );
    INSERT OR IGNORE INTO paper_bets (created_at, game_id, market, line_type, validated, game_date, start_utc, home,
        away, side, selection, line, p_model, p_market, edge, best_price, best_book, med_price, books, close_p_side,
        close_price, close_at, clv, result, profit, graded_at)
    SELECT created_at, game_id, 'h2h', COALESCE(line_type, 'early'), CASE WHEN COALESCE(line_type,'early')='early' THEN 1 ELSE 0 END,
        game_date, start_utc, home, away, CASE WHEN side='home' THEN 'a' ELSE 'b' END, pick_team, 0, p_model,
        p_market, edge, best_price, best_book, med_price, books, close_p_side, close_price, close_at, clv, result,
        profit, graded_at
    FROM paper_picks;
    """),
    (4, """
    -- invite by name and phone: the person only sets a password; login by name or phone
    ALTER TABLE invites ADD COLUMN name TEXT;
    ALTER TABLE invites ADD COLUMN phone TEXT;
    """),
]


def run_migrations() -> int:
    with db() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at TEXT)")
        done = {r[0] for r in conn.execute("SELECT version FROM schema_version").fetchall()}
    for version, sql in MIGRATIONS:
        if version in done:
            continue
        with db() as conn:
            conn.executescript(sql)
            conn.execute("INSERT INTO schema_version VALUES (?,?)", (version, utcnow()))
        log.info("applied migration %s", version)
    return max(v for v, _ in MIGRATIONS)
