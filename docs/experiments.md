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

## 6. Pick thresholds on the strict measure — totals 3% -> 5%, puck line 5% -> 7%

Early line, 2023-26, best US price (reg3: EU books, shown at the MEDIAN price since the best EU price is not
reachable from the US), closing EV vs the fair no-vig closing price:

| Market | Edge | Bets | Moved our way | Closing EV | By season |
|---|---|---|---|---|---|
| Moneyline | 5% | 840 | 74.9% | +1.09% | +1.29 / +0.98 / +0.94 |
| Moneyline | 7% | 433 | 78.1% | +2.09% | |
| Totals | 3% (old) | 236 | 72.9% | -0.34% | -0.8 / -0.7 / +0.3 |
| Totals | **5%** | 89 | 78.7% | +1.07% | +0.9 / +0.7 / +2.4 |
| Puck line | 5% (old) | 144 | 59.7% | -0.69% | |
| Puck line | **7%** | 51 | 72.5% | +0.61% | still unproven (51 bets) |
| Regulation 3-way | 5% (median price) | 645 | 81.1% | +2.59% | +3.6 / +3.5 / +0.8 |

Moneyline stays at 5% (positive every season; 7% doubles the value per bet but halves the count).

## 7. Special teams and travel — rejected

Moneyline model, out of sample 2023-26 (log loss / early-line strict closing EV at 5%):
current 0.6688 / +1.09%; + special-teams net xG (PP xGF - PK xGA, EWMA) 0.6689 / +1.08%;
+ travel km 0.6688 / +1.02%; + time-zone shift 0.6689 / +1.01%; + away team's zones from home 0.6687 / +0.96%;
all together 0.6688 / +0.67%. Goal model: special teams made totals worse (0.6922 -> 0.6925, closing EV
+0.62% -> +0.25%); travel was neutral to worse. Special teams are already inside all-situation xG, and the
market already prices travel. Code removed.

## 8. Anchor the goal model to the moneyline model — rejected

Shift each game's goal split (lam_h : lam_a, total unchanged) until the goal model's moneyline equals the
moneyline model's probability (weight 1), or the logit average of the two (weight 0.5). Early line, strict
closing EV at the MEDIAN price for every market (so the puck line reads lower than in section 6, which used
the best US price):

| Anchor weight | Goal-model ML log loss | Puck line log loss | Puck line 7%: bets, EV | Reg 3-way 5%: bets, EV |
|---|---|---|---|---|
| 0 (current) | 0.6687 | 0.6563 | 51, -1.27% | 645, +2.59% |
| 0.5 | 0.6691 | 0.6566 | 66, -1.87% | 664, +2.22% |
| 1 | 0.6702 | 0.6575 | 103, -2.38% | 783, +1.51% |

With the expected goalie added, the goal model already predicts winners as well as the moneyline model
(0.6687 vs 0.6688), so there is nothing to borrow; anchoring only imports the other model's errors.
The puck line's weakness is not the win probability; it stays unproven.

## 9. Staking — adopted: one pick per game per team (flat; stake sizing tested, not used by choice)

Early-line picks 2023-26 at the section-6 thresholds (1,626 picks; 431 games had 2-3 picks on the same team).
Closing EV per unit staked vs the fair closing price; ROI is about +6% for every scheme (noise).

| Staking | Bets | Staked | Profit | Closing EV / unit | Worst drawdown |
|---|---|---|---|---|---|
| Flat 1u, every pick | 1626 | 1626 | +97u | +1.33% | 33u |
| Flat 1u, best edge per game+team | 1171 | 1171 | +65u | +1.59% | 24u |
| Quarter Kelly cap 3u, best per game+team | 1171 | 2635 | +156u | +1.94% | 65u |
| **Edge / min edge cap 2u, best per game+team** | 1171 | 1709 | +100u | **+1.98%** | 41u |

Moneyline, puck line and regulation on the same team are one exposure; totals and the regulation tie are
separate. Live, a later run never adds a second bet on an exposure already bet for that game.

## 10. Alternate lines: calibrate per line — adopted (Bovada tab)

Goal model, out of sample 2023-26 (4,072 games), chance of each line vs how often it happened:

| Line | Actual | Raw model | One calibration per market | Per-line calibration |
|---|---|---|---|---|
| Home +2.5 | 82.0% | 87.3% | 88.2% | **82.3%** |
| Home -2.5 | 22.3% | 16.4% | 15.1% | **21.2%** |
| Over 4.5 | 78.5% | 75.9% | 67.4% | **77.2%** |
| Over 7.5 | 24.9% | 27.4% | 33.9% | **23.3%** |

The score grid is too narrow for 3+ goal margins, and the main-line totals calibration (fit on 5.5-6.5)
pushes far totals toward 50%. Per-line Platt calibration, fit walk-forward on earlier seasons, fixes both
(log loss better or equal on every line except ±1.5 and 5.5, where it is within 0.002). Used only for the
Bovada tab's alternate lines; main-line picks are unchanged.
