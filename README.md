# Maxine Machine

NHL game model and betting picks. Once a day at **2:00 PM Pacific** it prices that day's games for four
markets (moneyline, total, puck line, regulation 3-way), compares the model with the live odds, flags
picks, and builds best-chance parlays. Picks are logged, checked against the closing price, and graded
the next morning. Deployed on Railway (see `DEPLOY.md`); every test behind the numbers below is in
`docs/experiments.md`.

Live parameter values (weights, calibration) are on the app's **Model** tab and in `reports/scorecard.json`.
Values quoted below are from the 2026-10-20 scorecard.

---

## The formula, step by step

Every run, scheduled or from **Admin → Run full formula**, does all of this in about a minute.

### 1. Data

- **NHL API** (`api-web.nhle.com`): every regular-season game since 2021-22. Schedule, final score, how it
  ended (regulation / OT / shootout), goalies (who started, shots, saves) and every shot with its location.
  Each run first pulls any newly finished games and reloads them.
- **The Odds API**: one live snapshot of US books (moneyline, puck line, totals: 3 credits) and one of
  European books (moneyline with a Draw = regulation 3-way: 1 credit).

### 2. Expected goals (xG) for every shot

The chance an unblocked shot becomes a goal. Logistic regression on:

| Input | Detail |
|---|---|
| distance, angle | to the centre of the net (x = ±89 ft) |
| shot type | wrist, slap, snap, backhand, tip-in, deflection, wrap-around, ... |
| rebound | same team's previous unblocked shot within 3 seconds |
| rush | the other team had the last event within 4 seconds |
| skater difference | power play / short-handed |
| third period, overtime | flags |

Blocked shots, shootout attempts and shots at an empty net are excluded. Walk-forward: each season's
shots are scored by a model fit only on earlier seasons.

### 3. Team form before each game

For each team, an exponentially weighted average of its past games with a **15-game half-life** (the most
recent game counts most; a game 15 games ago counts half as much). Carried across seasons, so October leans
on last year; Utah continues Arizona.

- xG for and against per game (all situations)
- 5-on-5 xG share = xG for ÷ (xG for + xG against) at 5-on-5
- goals for and against per game
- rest: days since the previous game (capped at 4), and a back-to-back flag

### 4. Expected starting goalie

The starter is often not confirmed at 2 PM, so each team's goalie input is a **probability-weighted average
of its possible starters**:

1. **Candidates:** every goalie who started one of the team's last 20 games.
2. **Chance each one starts:** logistic regression on his share of the team's last 5, 10 and 20 starts,
   whether he started the previous game, whether the team is on a back-to-back, the two together (the goalie
   who played last night usually sits), and days since his last start. Normalised so the chances add to 100%.
   The most likely candidate starts about 2 times in 3.
3. **Quality of each goalie:** career goals saved above expected per 30 shots, shrunk toward league average:
   `quality = 30 × GSAx ÷ (shots faced + 3000)`. A goalie with little history counts as about average.
4. **Expected goalie quality** = Σ chance × quality.

The board shows the most likely starter and his chance, e.g. "Gibson (81%)".

### 5. Moneyline model (who wins, overtime and shootout included)

Logistic regression on home-minus-away differences:

| Input | Weight (standardised) |
|---|---|
| intercept (home ice) | +0.159 |
| all-situation xG differential (for − against) | +0.312 |
| goal differential | +0.123 |
| expected goalie quality | +0.084 |
| 5-on-5 xG share | +0.059 |
| home on a back-to-back | −0.083 |
| away on a back-to-back | +0.124 |
| rest difference | +0.036 |

Then **calibrated**: `p = sigmoid(a + b × logit(p_raw))`, fit on the model's own out-of-sample predictions
(b = 0.80 < 1 pulls back overconfident calls).

### 6. Goal model (totals, puck line, regulation 3-way)

1. **Goals per team.** Each team's regulation goals ~ Poisson(λ). λ comes from a Poisson regression on its
   own xG for and goals for, the opponent's xG against and goals against, home ice, both teams'
   back-to-backs and rest, and the **opponent's expected goalie quality**.
2. **Goalies shift who scores, not how many.** A twin regression without the goalie input gives the
   expected total; both λs are scaled so λ_home + λ_away equals it. (With goalies in the total, totals got
   slightly worse in testing.)
3. **Score grid** P(home = i, away = j) for 0–12 goals each, with three corrections fit on past seasons:
   - **Ties:** regulation ties happen more often than independent goals predict; the tie cells are scaled
     by ×1.373, then the grid is renormalised.
   - **Empty-net goals:** 35.8% of one-goal regulation wins become two-goal wins (the trailing team pulls
     its goalie).
   - **Overtime/shootout:** P(home wins it) = logistic(0.036 + 0.909 × log(λ_home ÷ λ_away)). The winner is
     credited one goal, as books grade totals and puck lines.
4. **Markets from the one grid:**
   - regulation 3-way: P(home ahead after 60), P(tied), P(away ahead)
   - total L: P(final total > L), P(< L), P(= L) (push on whole-number lines)
   - puck line: P(home final margin + line > 0), push when = 0
5. **Calibrated** per market, `p = sigmoid(a + b × logit p)`, fit on out-of-sample predictions against
   actual results: totals (a −0.028, b 0.653), puck line (a −0.012, b 1.055).

### 7. Market price

For each book: remove its margin, `p = (1/price) ÷ Σ(1/price over its outcomes)`. The market probability
is the **median across books**. For totals and puck lines, the line used is the one the most books quote.
The best available price and its book are recorded.

### 8. Edge and what becomes a pick

- **Edge** = model probability − market probability, for the side the model rates higher than the market.
  Regulation 3-way: the outcome (home / tie / away) with the largest positive gap.
- **Minimum edge:** moneyline **5%**, total **5%**, puck line **7%**, regulation **5%** (Railway variables
  `MIN_EDGE_H2H`, `MIN_EDGE_TOTALS`, `MIN_EDGE_SPREADS`, `MIN_EDGE_REG3`).
- **Clean game:** neither team plays another game first.
- **One pick per team per game.** Moneyline, puck line and regulation on the same team win or lose together,
  so only the one with the largest edge is picked; the others show "same bet as". The total and the
  regulation tie are separate.
- A line must be posted and the game not started.

### 9. Parlays (3 to 8 legs, best chance to hit)

1. Candidates: both sides of every game's moneyline, total and puck line (US books; regulation 3-way is
   left out because it is priced at European books).
2. A leg qualifies only if the model rates it **at least as likely as the market** does.
3. **One leg per game** (same-game legs are correlated): the qualifying leg with the highest model chance.
4. The N-leg parlay = the N games whose best leg is most likely.
   **Chance to hit** = product of the leg chances.
5. Price: the single book that offers every leg at the best combined price; if none has them all, the best
   price per leg, marked "mixed books".

### 10. Bovada -130 parlay (Bovada tab)

1. Every Bovada line for today's games, one request per game: moneyline, main and alternate puck lines,
   main and alternate totals (about 5 credits a game). Whole-number lines are skipped (they can push).
2. Chance of each leg: moneyline from the win model; puck lines and totals from the goal model's score grid,
   **calibrated line by line** on past results (the grid underrates 3+ goal margins, so e.g. +2.5 is pulled
   down from 87% to about 82%).
3. One leg per game; every 2-, 3- and 4-game combination whose combined Bovada price is between −145 and −120.
4. **Best, 2nd best, 3rd best** = highest chance to hit, each on a different set of games. At a near-fixed
   price the highest chance is also the best value. Each shows chance to hit, break-even (−130 ≈ 56.5%) and
   expected value.

### 11. Tracking

- The first price seen is the paper bet, 1 unit.
- Every 15 minutes (9 AM–8:45 PM PT) the app checks for picks starting within 30 minutes and records the
  **closing price** for the same market and line. **CLV** = our side's no-margin probability at close minus
  when we bet: positive means the market moved our way.
- **3:30 AM**: results are loaded and picks graded (final score for moneyline, totals and puck line;
  regulation goals for the 3-way).

---

## What the evidence says

Out of sample, 2023-24 to 2025-26 (`docs/experiments.md`):

- The model is less accurate than the closing line overall (moneyline log loss 0.6688 vs 0.6651), as is
  every public model we have compared.
- **Day-before (early) line:** picks at the thresholds above beat the fair closing price: moneyline +1.1%,
  totals +1.1%, regulation 3-way +2.6% per bet, positive in every season.
- **Game-day line:** in the backtest, morning and ~75-minute-before picks did **not** beat the closing price
  (−0.8% to −1.6%), even when the actual starting goalie was known. The live 2 PM record will show how they do.
- **Puck line:** not shown to beat the close at any threshold with enough bets.

Tested and rejected: score-adjusted xG, season-start regression to the mean, special teams, travel and time
zones, Pinnacle as a sharp anchor, a confirmed-starter feed, anchoring the goal model to the moneyline model.

---

## Schedule (Pacific)

| Time | Job |
|---|---|
| 2:00 PM | **Full formula for today's games**: picks, board, parlays, Bovada -130 parlay |
| every 15 min, 9 AM–8:45 PM | closing prices for picks about to start (only spends credits when one is due) |
| 3:30 AM | results, xG, grade picks |

**Admin → Run full formula (today's games)** runs the 2 PM job on demand (about 4 odds credits, plus about 5 per game for Bovada).

---

## Local setup (macOS)

```
cd ~/dev/maxine-machine
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env        # then put your Odds API key in .env
pytest -q
```

| Command | What it does |
|---|---|
| `maxine nhl-season 20212022 20222023 20232024 20242025 20252026` | Backfill NHL seasons (raw JSON in `data/raw/nhl/`) |
| `maxine load` | Parse raw files into `data/maxine.duckdb` |
| `maxine validate` | Rebuilt standings must match official standings |
| `maxine xg` | Fit xG walk-forward and score every shot |
| `maxine picks` | The full formula for today's games, same as the 2 PM run (uses about 4 odds credits) |
| `maxine odds-backfill` | Historical odds for backtests (paid Odds API plan) |
| `maxine market` | Build opening / morning / closing lines from odds snapshots |
| `maxine backtest`, `maxine clv`, `maxine markets` | Backtests vs the closing line |
| `maxine scorecard` | Write `reports/scorecard.json` for the Model tab |

## Layout

```
src/maxine/
  ingest/          NHL API, The Odds API, BetNHL
  warehouse/       DuckDB schema and loaders
  models/xg.py     expected goals per shot
  features.py      team form, rest, goalie inputs
  goalies.py       expected starting goalie
  models/winprob.py  moneyline model + calibration
  models/goals.py  goal model: score grid, totals, puck line, regulation 3-way
  live_markets.py  live odds -> no-margin market probabilities
  picks.py         the daily run: board and picks
  parlays.py       best-chance parlays
  bovada.py        Bovada -130 parlay finder (alternate lines)
  paper.py         paper bets: store, closing price, grade
  web/             jobs, auth, app database
server.py          FastAPI app + scheduler
templates/app.html the web app
docs/experiments.md every model test, before/after
```
