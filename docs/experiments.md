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
