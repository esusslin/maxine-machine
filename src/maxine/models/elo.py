"""Team Elo for the NHL, after Neil Paine's (ex-FiveThirtyEight) NHL Elo.

Online by construction: the probability for each game uses only ratings from games before it.
"""
import math
from dataclasses import dataclass

from maxine.teams import FRANCHISE_PREDECESSOR


@dataclass(frozen=True)
class EloParams:
    k: float = 6.0                 # rating change per game
    home_adv: float = 50.0         # Elo points added to the home team
    carryover: float = 0.70        # share of last season's rating kept; rest reverts to the mean
    mean: float = 1505.0
    use_mov: bool = True           # margin-of-victory multiplier: 0.6686 * ln(MOV) + 0.8048


def win_prob(diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def run(games: list[tuple], params: EloParams = EloParams()) -> dict[int, float]:
    """games: (game_id, season, home, away, home_score, away_score) sorted by start time.
    Returns {game_id: pre-game home win probability}."""
    ratings: dict[str, float] = {}
    season_of: dict[str, int] = {}
    preds: dict[int, float] = {}
    for game_id, season, home, away, hs, as_ in games:
        for team in (home, away):
            if team not in ratings:
                pred = FRANCHISE_PREDECESSOR.get(team)
                ratings[team] = ratings.get(pred, params.mean) if pred else params.mean
                season_of[team] = season_of.get(pred, season) if pred else season
            if season_of[team] != season:  # first game of a new season: revert toward the mean
                ratings[team] = params.carryover * ratings[team] + (1 - params.carryover) * params.mean
                season_of[team] = season
        diff = ratings[home] + params.home_adv - ratings[away]
        p = win_prob(diff)
        preds[game_id] = p
        result = 1.0 if hs > as_ else 0.0
        mult = (0.6686 * math.log(abs(hs - as_)) + 0.8048) if params.use_mov and hs != as_ else 1.0
        delta = params.k * mult * (result - p)
        ratings[home] += delta
        ratings[away] -= delta
    return preds
