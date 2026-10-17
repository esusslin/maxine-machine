# Maxine Machine

Daily win, puck-line and total probabilities for NHL, AHL, ECHL and NCAA D-I hockey games.
Plan: see `docs/plan.md` (exported from the Claude Doc "Maxine Machine — Plan & Research").

## Setup (macOS)

```
cd ~/dev/maxine-machine
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env        # then put your Odds API key in .env
pytest -q                   # 5 parser tests
```

## Phase 0 commands

| Command | What it does |
| --- | --- |
| `maxine nhl-season 20212022 20222023 20232024 20242025 20252026` | Backfill five NHL seasons: schedule, box scores, play-by-play, final standings (raw JSON in `data/raw/nhl/`). Re-runs skip files already saved. |
| `maxine nhl-season 20242025 --no-pbp` | Same, without play-by-play (faster) |
| `maxine load` | Parse every raw file into `data/maxine.duckdb` |
| `maxine validate` | Phase 0 gate: standings rebuilt from games must match official standings for each season |
| `maxine odds --league nhl` | Snapshot current NHL odds (uses Odds API credits; prints credits used and remaining) |
| `maxine betnhl` | Save today's BetNHL tips page and extract picks (`--date YYYY-MM-DD` for another day) |
| `maxine daily` | Morning job: yesterday's NHL results, today's odds (NHL + AHL), today's BetNHL picks, then load |

## Layout

```
src/maxine/
  config.py          paths, API bases, .env
  http.py            throttled client; raw files saved before parsing
  ingest/nhl.py      api-web.nhle.com/v1
  ingest/odds.py     The Odds API (current + historical snapshots)
  ingest/betnhl.py   BetNHL free tips (raw HTML + v0 pick extractor)
  warehouse/         DuckDB schema and loaders
  validate.py        standings check
  cli.py             `maxine` command
data/                raw files + warehouse (gitignored)
```
