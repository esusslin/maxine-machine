"""How much to trust the model vs the market, by leg type, for the Parlays and Ladder tabs.

Chance used = sigmoid(w * logit(model) + (1 - w) * logit(market)), market = no-margin consensus.
Weights chosen from the four-season check (docs/experiments.md section 11): the side the model leaned to,
3+ points off the closing line, vs what the market expected:
  puck line underdog (+1.5, +2.5...)  +0.4 / +0.7 / +1.4 / +4.9  -> positive every season    w = 0.60
  moneyline (either side)              mixed, +1.4 to +1.5 overall                          w = 0.40
  puck line favourite (-1.5, ...)      +3.2 / +4.3 / -1.8 / +0.3, small samples              w = 0.25
  totals (over or under)               unders +2.1 / +3.3 then -1.3; overs inconsistent      w = 0.20
The season report on the Picks tab shows whether these should move as this season's games come in.
"""
from __future__ import annotations

import math

WEIGHTS = {"puck line underdog": 0.60, "moneyline": 0.40, "puck line favourite": 0.25, "total": 0.20}


def leg_type(kind: str, team_line: float = 0.0) -> str:
    """kind: moneyline / puck line / total (or h2h / spreads / totals). team_line: the bet team's own line."""
    if kind in ("moneyline", "h2h"):
        return "moneyline"
    if kind in ("total", "totals"):
        return "total"
    return "puck line underdog" if team_line > 0 else "puck line favourite"


def blend(p_model: float, p_market: float | None, kind: str, team_line: float = 0.0) -> float:
    if p_market is None or p_market != p_market:      # no market chance: nothing to blend with
        return p_model
    w = WEIGHTS[leg_type(kind, team_line)]
    lg = lambda p: math.log(min(max(p, 1e-6), 1 - 1e-6) / (1 - min(max(p, 1e-6), 1 - 1e-6)))
    return 1 / (1 + math.exp(-(w * lg(p_model) + (1 - w) * lg(p_market))))
