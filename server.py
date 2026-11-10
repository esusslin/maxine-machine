"""Maxine Machine — hockey model + early-line picks. Serving plane.

Single FastAPI service; scheduled jobs run in-process via APScheduler, one at a time.
Same shape as the-algo:
  * SQLite (app state) + DuckDB (hockey warehouse) on the /data volume -> NEVER more than one replica.
  * APScheduler is in-process -> if the process dies, jobs stop silently. /health reports per-job
    last success so an external pinger can catch it.
Login, invites and password resets are copied from the-algo.

    uvicorn server:app --reload
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from maxine.web import auth, jobs
from maxine.web.appdb import db, get_kv, last_run, query, run_migrations, utcnow
from maxine.web.settings import settings

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("maxine")

scheduler = BackgroundScheduler(timezone=settings.TZ, executors={"default": ThreadPoolExecutor(1)})
ROOT = Path(__file__).parent
TEMPLATES = ROOT / "templates"
SCORECARD = ROOT / "reports" / "scorecard.json"

EXPECTED_FRESHNESS = {
    "heartbeat": timedelta(hours=2),
    "results_and_grade": timedelta(hours=30),
    "picks": timedelta(hours=26),      # one run a day at 2 PM
    "capture_closes": timedelta(hours=14),
}
STARTUP_GRACE = timedelta(hours=3)
BOOT_TIME = datetime.now(timezone.utc)
MANUAL = {
    "picks": jobs.run_picks,
    "closes": lambda: jobs.capture_closes(force=True),
    "evaluate": jobs.evaluate_season,
    "ladder": jobs.refresh_ladder,
    "results": jobs.results_and_grade,
    "bootstrap": jobs.bootstrap,
}


def register_jobs() -> None:
    for func, trigger, kwargs, job_id in jobs.JOBS:
        scheduler.add_job(func, trigger, id=job_id, replace_existing=True, coalesce=True,
                          max_instances=1, misfire_grace_time=3600, **kwargs)
        log.info("registered job %s (%s %s)", job_id, trigger, kwargs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    version = run_migrations()
    log.info("app schema version %s at %s", version, settings.APP_DB_PATH)
    for p in settings.validate():
        log.error("CONFIG: %s", p)
    try:
        admin_id = auth.bootstrap_admin()
        if admin_id:
            log.info("admin user ready (id=%s)", admin_id)
    except Exception as exc:  # noqa: BLE001 — never block startup on seeding
        log.error("admin bootstrap failed: %s", exc)
    if settings.ENABLE_JOBS:
        register_jobs()
        try:
            if settings.BOOTSTRAP_ON_EMPTY and jobs.warehouse_games() == 0:
                log.warning("warehouse is empty: starting NHL backfill (about 2 hours)")
                scheduler.add_job(jobs.bootstrap, id="bootstrap", replace_existing=True)
        except Exception:  # noqa: BLE001
            log.exception("could not check warehouse")
        scheduler.start()
        log.info("scheduler started (tz=%s)", settings.TZ)
    try:
        yield
    finally:
        if scheduler.running:
            scheduler.shutdown(wait=False)


app = FastAPI(title="Maxine Machine", version="0.1.0", lifespan=lifespan)


# ==========================================================================
# health + shell
# ==========================================================================
@app.get("/health")
def health() -> JSONResponse:
    now = datetime.now(timezone.utc)
    warming_up = (now - BOOT_TIME) < STARTUP_GRACE
    out, degraded = {}, []
    for name, max_age in EXPECTED_FRESHNESS.items():
        ok_run = last_run(name, "success")
        last = last_run(name)
        age = (now - datetime.fromisoformat(ok_run["finished_at"])) if ok_run else None
        stale = age is None or age > max_age
        if stale and not (warming_up and ok_run is None):
            degraded.append(name)
        out[name] = {"last_success": ok_run["finished_at"] if ok_run else None,
                     "last_status": last["status"] if last else None,
                     "last_error": last["error"] if last and last["status"] == "error" else None,
                     "stale": stale}
    boot = last_run("bootstrap")
    alive = scheduler.running or not settings.ENABLE_JOBS
    return JSONResponse(status_code=200 if alive else 503, content={
        "ok": alive and not degraded, "alive": alive, "warming_up": warming_up,
        "scheduler_running": scheduler.running, "degraded": degraded, "jobs": out,
        "bootstrap": dict(boot) if boot else None, "checked_at": utcnow(),
    })


@app.get("/", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
def app_page() -> HTMLResponse:
    return HTMLResponse((TEMPLATES / "app.html").read_text(encoding="utf-8"))


@app.get("/static/manifest.json")
def manifest() -> dict:
    return {"name": "Maxine Machine", "short_name": "Maxine", "start_url": "/app", "display": "standalone",
            "background_color": "#0b0f14", "theme_color": "#0b0f14", "icons": []}


# ==========================================================================
# auth (copied from the-algo)
# ==========================================================================
class LoginBody(BaseModel):
    username: str
    password: str


class RegisterBody(BaseModel):
    invite: str
    username: str
    password: str
    phone: str | None = None


class InviteBody(BaseModel):
    note: str = ""
    phone: str | None = None


class ResetRequestBody(BaseModel):
    identifier: str


class ResetBody(BaseModel):
    token: str
    password: str


class ProfileBody(BaseModel):
    phone: str | None = None
    sms_opt_in: bool | None = None


@app.post("/api/auth/login")
def login(body: LoginBody) -> dict:
    user = auth.authenticate(body.username, body.password)
    if not user:
        raise HTTPException(401, "invalid username or password")
    return {"token": auth.create_token(user["id"], user["username"], user["role"]),
            "user": {"id": user["id"], "username": user["username"], "role": user["role"]}}


@app.get("/api/auth/invite/{code}")
def check_invite(code: str) -> dict:
    ok, reason = auth.check_invite(code)
    return {"valid": ok, "reason": reason}


@app.post("/api/auth/register")
def register(body: RegisterBody) -> dict:
    try:
        user = auth.redeem_invite(body.invite, body.username, body.password, body.phone)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"token": auth.create_token(user["user_id"], user["username"], user["role"]), "user": user}


@app.post("/api/auth/forgot")
def forgot_password(request: Request, body: ResetRequestBody) -> dict:
    r = auth.request_reset(body.identifier)
    if r.get("sent"):
        from maxine.web.sms import send_reset
        sms = send_reset(r["token"], r["phone"], str(request.base_url))
        if sms["status"] != "sent":
            log.warning("reset SMS failed: %s", sms.get("error"))
    return {"ok": True, "message": "If that account exists and has a phone on file, a reset link has been sent."}


@app.get("/api/auth/reset/{token}")
def check_reset(token: str) -> dict:
    ok, reason = auth.check_reset(token)
    return {"valid": ok, "reason": reason}


@app.post("/api/auth/reset")
def do_reset(body: ResetBody) -> dict:
    try:
        user = auth.redeem_reset(body.token, body.password)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"token": auth.create_token(user["user_id"], user["username"], user["role"]), "user": user}


@app.post("/api/admin/reset-link/{user_id}")
def admin_reset_link(request: Request, user_id: int, admin: dict = Depends(auth.current_admin)) -> dict:
    u = auth.get_user_by_id(user_id)
    if not u:
        raise HTTPException(404, "user not found")
    r = auth.create_reset(user_id, issued_by=admin["username"])
    link = f"{str(request.base_url).rstrip('/')}/app?reset={r['token']}"
    out = {"username": u["username"], "link": link, "expires_at": r["expires_at"]}
    if u["phone"]:
        from maxine.web.sms import send_reset
        out["sms"] = send_reset(r["token"], u["phone"], str(request.base_url))
    return out


@app.get("/api/auth/me")
def me(user: dict = Depends(auth.current_user)) -> dict:
    return {"id": user["id"], "username": user["username"], "role": user["role"], "phone": user["phone"],
            "sms_opt_in": user["sms_opt_in"], "can_self_reset": bool(user["phone"])}


@app.patch("/api/auth/me")
def update_profile(body: ProfileBody, user: dict = Depends(auth.current_user)) -> dict:
    from maxine.web.sms import normalize_phone
    with db() as conn:
        if body.phone is not None:
            conn.execute("UPDATE users SET phone=? WHERE id=?", (normalize_phone(body.phone) or None, user["id"]))
        if body.sms_opt_in is not None:
            conn.execute("UPDATE users SET sms_opt_in=? WHERE id=?", (1 if body.sms_opt_in else 0, user["id"]))
    u = auth.get_user_by_id(user["id"])
    return {"phone": u["phone"], "sms_opt_in": u["sms_opt_in"], "can_self_reset": bool(u["phone"])}


@app.post("/api/admin/invites")
def make_invite(request: Request, body: InviteBody, admin: dict = Depends(auth.current_admin)) -> dict:
    inv = auth.create_invite(admin["id"], note=body.note)
    inv["link"] = f"{str(request.base_url).rstrip('/')}/app?invite={inv['code']}"
    if body.phone:
        from maxine.web.sms import send_invite
        inv["sms"] = send_invite(inv["code"], body.phone, str(request.base_url), inviter=admin["username"])
    return inv


@app.get("/api/admin/invites")
def get_invites(admin: dict = Depends(auth.current_admin)) -> dict:
    return {"invites": auth.list_invites()}


@app.delete("/api/admin/invites/{code}")
def kill_invite(code: str, admin: dict = Depends(auth.current_admin)) -> dict:
    return {"ok": auth.revoke_invite(code)}


@app.get("/api/admin/users")
def list_users(admin: dict = Depends(auth.current_admin)) -> dict:
    rows = query("SELECT id, username, role, phone, created_at, last_login FROM users ORDER BY id")
    return {"users": [dict(r) for r in rows]}


# ==========================================================================
# hockey
# ==========================================================================
@app.get("/api/board")
def board(user: dict = Depends(auth.current_user)) -> dict:
    row = get_kv("board")
    data = json.loads(row["value"]) if row else {"records": [], "min_edges": settings.MIN_EDGES}
    data["updated_at"] = row["updated_at"] if row else None
    return data


@app.get("/api/parlays")
def parlays_today(user: dict = Depends(auth.current_user)) -> dict:
    row = get_kv("parlays")
    data = json.loads(row["value"]) if row else {"parlays": []}
    data["updated_at"] = row["updated_at"] if row else None
    return data


@app.get("/api/evaluation")
def evaluation(user: dict = Depends(auth.current_user)) -> dict:
    row = get_kv("evaluation")
    data = json.loads(row["value"]) if row else {}
    data["updated_at"] = row["updated_at"] if row else None
    return data


@app.get("/api/ladder")
def ladder_parlays(user: dict = Depends(auth.current_user)) -> dict:
    row = get_kv("ladder")
    data = json.loads(row["value"]) if row else {"options": []}
    data["updated_at"] = row["updated_at"] if row else None
    return data


@app.get("/api/paper")
def paper_history(user: dict = Depends(auth.current_user)) -> dict:
    from maxine import paper
    return paper.summary()


@app.get("/api/betnhl")
def betnhl_picks(user: dict = Depends(auth.current_user)) -> dict:
    from maxine.warehouse.db import connect
    con = connect()
    try:
        rows = con.execute("""SELECT pick_date, idx, pick_text, first_decimal_odds, first_american_odds
                              FROM betnhl_picks ORDER BY pick_date DESC, idx LIMIT 100""").fetchall()
    finally:
        con.close()
    return {"picks": [{"date": str(r[0]), "idx": r[1], "pick": r[2], "decimal": r[3], "american": r[4]}
                      for r in rows]}


@app.get("/api/model")
def model(user: dict = Depends(auth.current_user)) -> dict:
    if not SCORECARD.exists():
        return {"missing": True}
    return json.loads(SCORECARD.read_text())


@app.get("/api/algorithm")
def algorithm(user: dict = Depends(auth.current_user)) -> dict:
    from maxine.picks import VALIDATED
    return {"min_edges": settings.MIN_EDGES, "game_day_picks": settings.GAME_DAY_PICKS,
            "reg3_feed": settings.REG3_FEED, "validated": sorted(f"{m}|{lt}" for m, lt in VALIDATED)}


@app.get("/api/admin/status")
def admin_status(admin: dict = Depends(auth.current_admin)) -> dict:
    from maxine.warehouse.db import connect
    counts = {}
    try:
        con = connect()
        for t in ("games", "shots", "shot_xg", "betnhl_picks"):
            try:
                counts[t] = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            except Exception:  # noqa: BLE001
                counts[t] = None
        counts["latest_game"] = str(con.execute("SELECT max(game_date) FROM games").fetchone()[0])
        con.close()
    except Exception as exc:  # noqa: BLE001
        counts["error"] = str(exc)
    runs = [dict(r) for r in query("SELECT job_name, started_at, finished_at, duration_s, status, rows_affected, "
                                   "note, error FROM job_runs ORDER BY id DESC LIMIT 25")]
    credits = get_kv("odds_credits_remaining")
    return {"warehouse": counts, "runs": runs, "odds_credits_remaining": credits["value"] if credits else None,
            "min_edges": settings.MIN_EDGES, "config_problems": settings.validate()}


@app.post("/api/admin/run/{name}")
def admin_run(name: str, admin: dict = Depends(auth.current_admin)) -> dict:
    if name not in MANUAL:
        raise HTTPException(404, f"unknown job {name}; one of {sorted(MANUAL)}")
    if not scheduler.running:
        raise HTTPException(503, "scheduler not running")
    scheduler.add_job(MANUAL[name], id=f"manual_{name}", replace_existing=True)
    return {"queued": name}
