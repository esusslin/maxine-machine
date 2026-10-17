"""Polite HTTP client. Every response is written to the raw layer before parsing."""
import json
import time
from pathlib import Path

import httpx

from maxine.config import RAW_DIR, USER_AGENT

_client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True)
_last_call: dict[str, float] = {}


def _throttle(host: str, min_interval: float) -> None:
    wait = _last_call.get(host, 0.0) + min_interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call[host] = time.monotonic()


def get(url: str, params: dict | None = None, min_interval: float = 0.5, retries: int = 3) -> httpx.Response:
    host = httpx.URL(url).host
    for attempt in range(retries):
        _throttle(host, min_interval)
        try:
            resp = _client.get(url, params=params)
        except httpx.TransportError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            continue
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(2 ** (attempt + 1))
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError(f"unreachable: {url}")


def raw_path(*parts: str) -> Path:
    path = RAW_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def fetch_json(url: str, save_to: Path, params: dict | None = None, refresh: bool = False, **kw) -> dict:
    """Return cached raw JSON if present (unless refresh), else fetch, save, return."""
    if save_to.exists() and not refresh:
        return json.loads(save_to.read_text())
    data = get(url, params=params, **kw).json()
    save_to.write_text(json.dumps(data))
    return data
