from maxine.models import elo
from maxine.teams import ODDS_NAME_TO_NHL


def test_win_prob_symmetry():
    assert elo.win_prob(0) == 0.5
    assert abs(elo.win_prob(50) + elo.win_prob(-50) - 1) < 1e-12


def test_elo_is_pregame_and_updates():
    games = [(1, 20242025, "AAA", "BBB", 4, 1), (2, 20242025, "AAA", "BBB", 2, 3)]
    preds = elo.run(games, elo.EloParams(k=10, home_adv=0))
    assert preds[1] == 0.5                      # no information before the first game
    assert preds[2] > 0.5                       # AAA won game 1, so favored in game 2


def test_franchise_inherits_rating():
    games = [(1, 20232024, "ARI", "BBB", 5, 0), (2, 20242025, "UTA", "BBB", 1, 2)]
    preds = elo.run(games, elo.EloParams(k=10, home_adv=0, carryover=1.0))
    assert preds[2] > 0.5                       # Utah starts from Arizona's rating


def test_team_map_covers_utah_and_montreal():
    assert ODDS_NAME_TO_NHL["Utah Mammoth"] == ODDS_NAME_TO_NHL["Utah Hockey Club"] == "UTA"
    assert ODDS_NAME_TO_NHL["Montréal Canadiens"] == "MTL"
