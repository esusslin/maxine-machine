# Deploying Maxine Machine to Railway

Single service, single replica, SQLite (app state) + DuckDB (hockey warehouse) on a volume at `/data`.
Same shape as the-algo.

Research stays on your Mac: backfills of odds history, backtests, `maxine scorecard`.
Railway serves the app and runs the daily jobs.

## 1. Before the first deploy (on your Mac)

```
maxine scorecard                      # writes reports/scorecard.json for the Model tab
git add -A && git commit -m "Railway serving plane" && git push
```

## 2. Create the service

1. Railway → New Project → Deploy from GitHub repo → `esusslin/maxine-machine`
2. Nixpacks reads `requirements.txt` (installs the package) and `Procfile`
3. Leave replicas at 1 (`railway.toml` sets `numReplicas = 1`)

## 3. Add the volume

Settings → Volumes → New Volume, mount path **`/data`**.

On first boot with an empty volume the app backfills five NHL seasons from the free NHL API
(about 2 hours, about 1.2 GB). Jobs queue behind it. Odds history is not needed on Railway.

## 4. Environment variables

```
ENV=production
TZ=America/Los_Angeles
LOG_LEVEL=INFO
MAXINE_DATA_DIR=/data
APP_DB_PATH=/data/app.db

ODDS_API_KEY=<your Odds API key>

JWT_SECRET_KEY=<generate: python -c "import secrets;print(secrets.token_urlsafe(48))">
JWT_EXPIRE_MINUTES=43200
ADMIN_USERNAME=emmet
ADMIN_PASSWORD=<8+ characters>
ADMIN_PHONE=+1...

MIN_EDGE=0.05
ENABLE_JOBS=true
BOOTSTRAP_ON_EMPTY=true

ENABLE_SMS=false
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=
```

The admin account is re-synced from `ADMIN_USERNAME` / `ADMIN_PASSWORD` on every boot (as in the-algo),
so changing the variable is the recovery path if you forget the password.

## 5. First boot

- `https://<app>.up.railway.app/health` → 200; `warming_up: true` for 3 hours; `bootstrap` shows progress
- Open `/app`, sign in as the admin
- Admin tab → Status shows warehouse game count climbing during the backfill

## Jobs (Pacific time)

| Time | Job | Odds credits |
|---|---|---|
| 03:30 | Results, xG, grade paper picks | 0 |
| 10:00 | BetNHL tips snapshot | 0 |
| 18:30 | Early-line picks for tomorrow | 3 |
| every 15 min 09:00–20:45 | Closing price, only when a picked game starts within 30 min | 3 per capture |

About 15–20 credits a day: the 20K Odds API plan covers it.

## Monitoring

Alert on the `ok` field of `/health`, not the status code. Admin tab shows the last 25 job runs.
