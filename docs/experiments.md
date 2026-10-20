# Model experiments

Every change is judged out of sample on 2023-24 to 2025-26 (`maxine.experiments.evaluate`): moneyline log loss,
goal-model log loss (totals, puck line), and early-line CLV (bets with edge over the threshold; share where the
closing line moved toward the model). A change ships only if it improves both accuracy and early-line CLV, and
holds season by season.

## 2026-10-20 — baseline (half-life 25, raw xG)

| Metric | Value |
|---|---|
| Moneyline log loss | 0.6700 |
| Moneyline early, edge >= 5% | 739 bets, CLV +2.18 pts, moved our way 74.4% |
| Totals log loss / early >= 3% | 0.6927 / 224 bets, +1.38 pts, 67.4% |
| Puck line log loss / early >= 5% | 0.6570 / 130 bets, +0.52 pts, 56.9% |

## 1. Score-adjusted xG — rejected (no effect)

5v5 shots weighted by 0.5 / (shooting team's xG share in that score state), weights fit on 2021-23 only:
-2: 0.962, -1: 0.983, 0: 1.0, +1: 1.017, +2: 1.041. Every metric moved by <= 0.001. Score effects inflate shot
volume, but xG already discounts the low-quality shots trailing teams take. Column `xg_adj` kept; not used.

## 2a. Form half-life — adopted: 15 games (was 25)

| Half-life | ML log loss | ML early 5% | Totals log loss | Totals early 3% | Puck line early 5% |
|---|---|---|---|---|---|
| 10 | 0.6691 | 1021, +2.06, 73.8% | 0.6921 | 255, +1.75, 75.7% | 192, 61.5% |
| **15** | **0.6693** | **869, +2.27, 74.9%** | **0.6921** | **239, +1.82, 75.3%** | 154, 57.1% |
| 20 | 0.6696 | 783, +2.25, 75.0% | 0.6924 | 225, +1.59, 70.2% | 138, 59.4% |
| 25 | 0.6700 | 739, +2.18, 74.4% | 0.6927 | 224, +1.38, 67.4% | 130, 56.9% |
| 35 | 0.6707 | 745, +1.85, 69.9% | 0.6933 | 258, +1.10, 63.6% | 129, 54.3% |
| 50 | 0.6717 | 787, +1.36, 63.4% | 0.6940 | 279, +0.92, 62.0% | 149, 50.3% |

Season by season (15 vs 25): moneyline finds more edges in all four seasons at equal or better line movement;
log loss better in 2024-25 and 2025-26, within 0.0003 in 2022-23 and 2023-24. Totals log loss better in all three
seasons, early-line movement better or equal in all three. 15 chosen over 10 as the more conservative step.

## 2b. Season-start regression to the league mean — rejected

Form pulled 20% or 40% toward last season's league average at each team's first game. Compared on the same
code path (the loop EWMA), the regression was neutral to slightly worse (ML early 69.1% -> 69.9%, totals early
66% -> 64%). The loop EWMA itself (pandas adjust=False style) was worse than the pandas default, so the default
stays and the regression code was removed.

## 3. Expected starting goalie — adopted

At early-line time the starter is not announced, so the goalie input is the probability-weighted quality of
the likely starters (`goalies.py`). Starter model out of sample 2023-26: most likely candidate starts 67.5%;
97.7% of starters were among the candidates (the rest: call-ups, new signings).

Ceiling first (actual starter from the box score, look-ahead): moneyline log loss 0.6693 -> 0.6686; early
line moved our way 74.9% -> 78.2%; game-day (morning) line 52.2% -> 58.0%.

| Moneyline model | Log loss | Early >= 5% | Morning >= 5% |
|---|---|---|---|
| No goalie | 0.6693 | 869, +2.27, 74.9% | 1208, +0.26, 52.2% |
| **Expected starter** | **0.6688** | **840, +2.38, 76.0%** | 1149, +0.32, 52.9% |
| Actual starter (ceiling) | 0.6686 | 886, +2.49, 78.2% | 1148, +0.51, 58.0% |

By season, early line, expected vs none: 2023-24 79.6% -> 81.9%, 2024-25 70.6% -> 70.0%, 2025-26 73.0% -> 74.7%.

Goal model (opponent's expected goalie in each side's rate):

| | Totals log loss | Puck line log loss | Puck line early >= 5% |
|---|---|---|---|
| No goalie | 0.6921 | 0.6570 | 154, +0.53, 57.1% |
| Expected starter, goalie in totals too | 0.6927 | 0.6562 | 145, +0.75, 60.7% |
| **Expected starter, total held** | **0.6921** | **0.6563** | **145, +0.76, 60.7%** |

Even the actual starter made totals slightly worse, so the goalie only shifts goals between the teams: the
lambdas are rescaled to the no-goalie model's total. Totals early line: 239 bets either way (75.3% -> 74.5%, noise).

Game-day picks need the confirmed starter (the ceiling row): that requires a starter feed, not built yet.

## 4. Pinnacle as a sharp anchor (moneyline) — rejected

Pinnacle's 2-way moneyline is already in the EU feed (no new credits). 2023-26, same games:

| Bet time | Pinnacle has a line | Log loss US median | Log loss Pinnacle | Model |
|---|---|---|---|---|
| Early (~33 h) | 10% of games | 0.6768 | 0.6773 | 0.6699 |
| Morning | 100% | 0.6661 | 0.6661 | 0.6688 |
| Close | 100% | 0.6651 | 0.6651 | 0.6688 |

Pinnacle is not posted yet when we bet (early line), and once posted it is no sharper than the US median.
Blending the model with Pinnacle, or betting US prices that beat Pinnacle's no-vig line (no model), never
cleared +0.3% closing EV on a sample over 100 bets. Not worth the extra credits for Pinnacle totals/puck lines.

## 5. Confirmed starting goalie on game day — rejected (feed not built)

Proxy for a confirmed-starter feed: the actual starter from the box score. Strict measure: price taken (best
US book) vs the fair no-vig closing price (Pinnacle close; US median when missing).

| Bet time | Goalie input | Edge >= 5%: bets | Moved our way | Closing EV | Beat fair close |
|---|---|---|---|---|---|
| Morning | expected | 1152 | 56.7% | -1.16% | 37.2% |
| Morning | confirmed | 1148 | 60.6% | -0.78% | 40.7% |
| ~75 min before | expected | 741 | 48.2% | -1.58% | 18.8% |
| ~75 min before | confirmed | 715 | 51.2% | -1.48% | 21.0% |

Knowing the starter helps, but the market prices the starter as fast as it is announced, so game-day bets
still lose to the close. Game-day stays unproven; no starter feed.

## Benchmark: early-line moneyline picks on the strict measure

Edge >= 5% vs US median, bet at best US price, 2023-26 clean games: 840 bets, line moved our way 74.9%,
closing EV +1.09% vs the fair closing price, beat it 56.9%, ROI +4.4%. By season closing EV +1.29% / +0.98% /
+0.94% (all positive). Edge >= 7%: 433 bets, closing EV +2.09%, beat fair close 63.5%.
