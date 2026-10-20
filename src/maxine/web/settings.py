"""Serving-plane configuration. Every env var the web app reads is declared here.

Mirrors the-algo's config. The research CLI (maxine ...) keeps using maxine.config.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


PLACEHOLDERS = {"change-me", "changeme", "replace-me", "your-secret", "<generate>", "xxx"}
MIN_SECRET_LENGTH = 32


def _is_placeholder(value: str) -> bool:
    return value.strip().lower() in PLACEHOLDERS or value.strip().startswith("<")


class Settings:
    ENV: str = os.getenv("ENV", "development")
    IS_PROD: bool = ENV == "production"
    TZ: str = os.getenv("TZ", "America/Los_Angeles")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    DATA_DIR: Path = Path(os.getenv("MAXINE_DATA_DIR", "./data"))
    APP_DB_PATH: Path = Path(os.getenv("APP_DB_PATH", str(DATA_DIR / "app.db")))

    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = _int("JWT_EXPIRE_MINUTES", 43200)

    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_FROM_NUMBER: str = os.getenv("TWILIO_FROM_NUMBER", "")
    ENABLE_SMS: bool = _bool("ENABLE_SMS", False)

    ADMIN_PHONE: str = os.getenv("ADMIN_PHONE", "")
    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "")

    ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")
    # minimum edge per market: the lowest edge whose early-line picks beat the fair closing price (docs/experiments.md)
    MIN_EDGE_H2H: float = _float("MIN_EDGE_H2H", _float("MIN_EDGE", 0.05))
    MIN_EDGE_TOTALS: float = _float("MIN_EDGE_TOTALS", 0.05)
    MIN_EDGE_SPREADS: float = _float("MIN_EDGE_SPREADS", 0.07)
    MIN_EDGE_REG3: float = _float("MIN_EDGE_REG3", 0.05)
    REG3_FEED: bool = _bool("REG3_FEED", True)

    @property
    def MIN_EDGES(self) -> dict:
        return {"h2h": self.MIN_EDGE_H2H, "totals": self.MIN_EDGE_TOTALS,
                "spreads": self.MIN_EDGE_SPREADS, "reg3": self.MIN_EDGE_REG3}
    GAME_DAY_PICKS: bool = _bool("GAME_DAY_PICKS", True)
    ENABLE_JOBS: bool = _bool("ENABLE_JOBS", True)
    BOOTSTRAP_ON_EMPTY: bool = _bool("BOOTSTRAP_ON_EMPTY", True)

    def ensure_dirs(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.APP_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    def validate(self) -> list[str]:
        problems = []
        if self.IS_PROD:
            if not self.JWT_SECRET_KEY:
                problems.append("JWT_SECRET_KEY is not set")
            elif _is_placeholder(self.JWT_SECRET_KEY):
                problems.append("JWT_SECRET_KEY is still a placeholder")
            elif len(self.JWT_SECRET_KEY) < MIN_SECRET_LENGTH:
                problems.append(f"JWT_SECRET_KEY is {len(self.JWT_SECRET_KEY)} characters; want {MIN_SECRET_LENGTH}+")
            if not (self.ADMIN_USERNAME and self.ADMIN_PASSWORD):
                problems.append("ADMIN_USERNAME / ADMIN_PASSWORD not set: nobody can sign in")
            if not self.ODDS_API_KEY:
                problems.append("ODDS_API_KEY not set: picks and closing lines cannot run")
            if self.ENABLE_SMS and not self.TWILIO_ACCOUNT_SID:
                problems.append("ENABLE_SMS is true but Twilio is not configured")
        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
