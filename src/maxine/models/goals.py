"""Goal model: one score distribution per game, priced into every market.

1. Regulation goals for each team ~ Poisson(lambda). lambda from a Poisson regression on pre-game form:
   own xG for and goals for (EWMA), opponent xG against and goals against, home ice, back-to-backs, rest.
   Walk-forward by season: season S uses a model fit only on earlier seasons.
2. Score matrix P(home=i, away=j) from independent Poissons, with the diagonal (regulation ties) rescaled
   by tie_factor = observed / predicted tie rate in the training seasons, then renormalised.
   Empty-net shift: a share en_shift of 1-goal regulation wins become 2-goal wins (trailing team pulls its
   goalie late). en_shift is fit on the training seasons from observed vs predicted margins.
3. Tied after regulation -> OT/shootout. P(home wins it) = logistic(a + b * log(lam_h / lam_a)), fit on
   training-season games that went to OT/SO. The OT/SO winner is credited one goal (books grade it so):
   final margin is +-1 and the final total is regulation total + 1.
4. Markets, all from the same matrix:
     moneyline       P(home wins, incl. OT/SO)
     regulation 3-way P(home reg win), P(reg tie), P(away reg win)
     totals L         P(final total > L), P(< L), P(= L) (push, whole-number lines)
     spread h (home)  P(home final margin + h > 0), push when = 0
5. Goalies: each side's rate uses the opponent's expected starter quality (goalies.py; the starter is
   unknown at bet time). The goalie shifts who scores, not how many: lambdas are rescaled so their sum
   equals that of a twin model without the goalie input (adding it to totals made totals slightly worse).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from maxine import goalies
from maxine.features import team_form

MAX_GOALS = 12
TOTAL_FEATURES = ["own_pre_xgf", "own_pre_gf", "opp_pre_xga", "opp_pre_ga", "is_home",
                  "own_b2b", "opp_b2b", "own_rest", "opp_rest"]
SIDE_FEATURES = TOTAL_FEATURES + ["opp_exp_gsax30"]


def regulation_goals(con) -> pd.DataFrame:
    """Goals scored in periods 1-3 per team-game (empty-net goals included; shootouts excluded)."""
    return con.execute("""
        SELECT g.game_id, g.home AS team,
               sum(CASE WHEN s.event_type='goal' AND s.period<=3 AND s.owner_team_id=g.home_team_id THEN 1 ELSE 0 END) AS reg_goals
        FROM games g LEFT JOIN shots s USING (game_id) WHERE g.league='NHL' AND g.game_type=2 GROUP BY ALL
        UNION ALL
        SELECT g.game_id, g.away AS team,
               sum(CASE WHEN s.event_type='goal' AND s.period<=3 AND s.owner_team_id=g.away_team_id THEN 1 ELSE 0 END)
        FROM games g LEFT JOIN shots s USING (game_id) WHERE g.league='NHL' AND g.game_type=2 GROUP BY ALL
    """).df()


def side_frame(con) -> pd.DataFrame:
    """One row per team-game with pre-game features (own vs opponent) and regulation goals scored."""
    tf = team_form(con)
    base = tf[["game_id", "season", "start_utc", "team", "opp", "is_home", "pre_xgf", "pre_xga", "pre_gf",
               "pre_ga", "b2b", "rest_days"]]
    opp = base[["game_id", "team", "pre_xga", "pre_ga", "b2b", "rest_days"]].rename(
        columns={"team": "opp", "pre_xga": "opp_pre_xga", "pre_ga": "opp_pre_ga", "b2b": "opp_b2b",
                 "rest_days": "opp_rest"})
    df = base.merge(opp, on=["game_id", "opp"]).rename(
        columns={"pre_xgf": "own_pre_xgf", "pre_gf": "own_pre_gf", "b2b": "own_b2b", "rest_days": "own_rest"})
    df["is_home"] = df["is_home"].astype(int)
    df = df.merge(regulation_goals(con), on=["game_id", "team"], how="left")
    eg = goalies.expected(con).rename(columns={"team": "opp", "exp_gsax30": "opp_exp_gsax30"})
    df = df.merge(eg, on=["game_id", "opp"], how="left").fillna({"opp_exp_gsax30": 0.0})
    return df


@dataclass
class GoalModel:
    rate: object            # fitted Poisson regression pipeline (SIDE_FEATURES)
    total_rate: object      # twin without the goalie input (TOTAL_FEATURES): fixes lam_h + lam_a
    tie_factor: float
    en_shift: float
    ot_a: float
    ot_b: float



def _matrix(lh: float, la: float, tie_factor: float, en_shift: float = 0.0) -> np.ndarray:
    g = np.arange(MAX_GOALS + 1)
    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    m[np.diag_indices_from(m)] *= tie_factor
    m = m / m.sum()
    if en_shift:
        out = m.copy()
        for i in range(MAX_GOALS):
            for j in range(MAX_GOALS):
                if i - j == 1:            # home by one -> home by two
                    moved = m[i, j] * en_shift
                    out[i, j] -= moved
                    out[i + 1, j] += moved
                elif j - i == 1:          # away by one -> away by two
                    moved = m[i, j] * en_shift
                    out[i, j] -= moved
                    out[i, j + 1] += moved
        m = out
    return m


def _tie_prob(lh, la):
    g = np.arange(MAX_GOALS + 1)
    return float((poisson.pmf(g, lh) * poisson.pmf(g, la)).sum())


def fit(train_sides: pd.DataFrame, train_games: pd.DataFrame) -> GoalModel:
    """train_sides: side_frame rows; train_games: game_id, home_reg, away_reg, last_period_type, home_win."""
    usable = train_sides.dropna(subset=SIDE_FEATURES + ["reg_goals"])
    rate = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=1000))
    rate.fit(usable[SIDE_FEATURES], usable["reg_goals"])
    total_rate = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=1000))
    total_rate.fit(usable[TOTAL_FEATURES], usable["reg_goals"])
    lam = _game_lambdas(rate, total_rate, usable)
    g = train_games.merge(lam, on="game_id")
    pred_tie = np.array([_tie_prob(a, b) for a, b in zip(g["lam_h"], g["lam_a"])])
    tie_factor = float((g["home_reg"] == g["away_reg"]).mean() / pred_tie.mean())
    # empty-net shift: observed vs predicted share of 2+ goal margins among regulation wins
    one = ge2 = 0.0
    for lh, la in zip(g["lam_h"], g["lam_a"]):
        mm = _matrix(lh, la, tie_factor)
        i, j = np.indices(mm.shape)
        nontie = mm[i != j].sum()
        one += mm[np.abs(i - j) == 1].sum() / nontie
        ge2 += mm[np.abs(i - j) >= 2].sum() / nontie
    n = len(g)
    reg = g[g["home_reg"] != g["away_reg"]]
    obs_ge2 = float((np.abs(reg["home_reg"] - reg["away_reg"]) >= 2).mean())
    en_shift = float(np.clip((obs_ge2 - ge2 / n) / (one / n), 0.0, 0.9))
    ot = g[g["home_reg"] == g["away_reg"]]
    x = np.log(ot["lam_h"] / ot["lam_a"]).to_numpy().reshape(-1, 1)
    lr = LogisticRegression(C=1.0).fit(x, ot["home_win"].astype(int))
    return GoalModel(rate, total_rate, tie_factor, en_shift, float(lr.intercept_[0]), float(lr.coef_[0][0]))


def _game_lambdas(rate, total_rate, sides: pd.DataFrame) -> pd.DataFrame:
    s = sides.copy()
    s["lam"] = rate.predict(s[SIDE_FEATURES])
    s["lam0"] = total_rate.predict(s[TOTAL_FEATURES])
    h = s[s["is_home"] == 1][["game_id", "lam", "lam0"]].rename(columns={"lam": "lam_h", "lam0": "t_h"})
    a = s[s["is_home"] == 0][["game_id", "lam", "lam0"]].rename(columns={"lam": "lam_a", "lam0": "t_a"})
    g = h.merge(a, on="game_id")
    k = (g["t_h"] + g["t_a"]) / (g["lam_h"] + g["lam_a"])
    g["lam_h"], g["lam_a"] = g["lam_h"] * k, g["lam_a"] * k
    return g[["game_id", "lam_h", "lam_a"]]


def game_lambdas(model: GoalModel, sides: pd.DataFrame) -> pd.DataFrame:
    return _game_lambdas(model.rate, model.total_rate, sides.dropna(subset=SIDE_FEATURES))


def price(model: GoalModel, lam_h: float, lam_a: float) -> dict:
    """Full market set for one game."""
    m = _matrix(lam_h, lam_a, model.tie_factor, model.en_shift)
    q = 1 / (1 + np.exp(-(model.ot_a + model.ot_b * np.log(lam_h / lam_a))))   # P(home wins OT/SO)
    i, j = np.indices(m.shape)
    reg_h, reg_t, reg_a = m[i > j].sum(), m[i == j].sum(), m[i < j].sum()
    return {"m": m, "q": float(q), "reg_home": float(reg_h), "reg_tie": float(reg_t), "reg_away": float(reg_a),
            "ml_home": float(reg_h + reg_t * q)}


def prob_total(p: dict, line: float) -> tuple[float, float, float]:
    """(over, under, push) for the final total, OT/SO winner credited one goal."""
    m = p["m"]
    i, j = np.indices(m.shape)
    total = i + j + (i == j)
    return float(m[total > line].sum()), float(m[total < line].sum()), float(m[total == line].sum())


def prob_spread(p: dict, home_line: float) -> tuple[float, float, float]:
    """(home covers, away covers, push) for home spread `home_line` (e.g. -1.5)."""
    m, q = p["m"], p["q"]
    i, j = np.indices(m.shape)
    margin = (i - j).astype(float)
    reg = i != j
    tie = m[~reg].sum()
    cover = m[reg & (margin + home_line > 0)].sum() + tie * (q * (1 + home_line > 0) + (1 - q) * (-1 + home_line > 0))
    lose = m[reg & (margin + home_line < 0)].sum() + tie * (q * (1 + home_line < 0) + (1 - q) * (-1 + home_line < 0))
    return float(cover), float(lose), float(1 - cover - lose)


def game_outcomes(con) -> pd.DataFrame:
    rg = regulation_goals(con)
    g = con.execute("""SELECT game_id, season, home, away, home_score, away_score, last_period_type
                       FROM games WHERE league='NHL' AND game_type=2""").df()
    g = g.merge(rg.rename(columns={"team": "home", "reg_goals": "home_reg"}), on=["game_id", "home"])
    g = g.merge(rg.rename(columns={"team": "away", "reg_goals": "away_reg"}), on=["game_id", "away"])
    g["home_win"] = g["home_score"] > g["away_score"]
    return g


def walk_forward(con, first_test_season: int = 20222023) -> tuple[pd.DataFrame, dict]:
    """Out-of-sample lambdas per game for each test season. Returns (game_id, season, lam_h, lam_a) and models."""
    sides, games = side_frame(con), game_outcomes(con)
    out, models = [], {}
    for season in sorted(s for s in games["season"].unique() if s >= first_test_season):
        model = fit(sides[sides["season"] < season], games[games["season"] < season])
        lam = game_lambdas(model, sides[sides["season"] == season])
        lam["season"] = season
        out.append(lam)
        models[int(season)] = model
    return pd.concat(out, ignore_index=True), models


# ---------------------------------------------------------------------------
# live: fit on everything, calibrate two-way markets, price upcoming fixtures
# ---------------------------------------------------------------------------
CAL_TOTAL_LINES = (5.5, 6.0, 6.5)
CAL_SPREAD_LINES = (-1.5, 1.5)


def _platt(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    lr = LogisticRegression(C=1e6).fit(np.log(p / (1 - p)).reshape(-1, 1), np.asarray(y, dtype=int))
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def calibrators(con) -> dict:
    """Per-market Platt (a, b) for totals and spreads, fit on out-of-sample predictions for standard lines
    against actual results (no odds needed, so it also runs on Railway). Pushes excluded."""
    lam, models = walk_forward(con)
    g = game_outcomes(con).merge(lam, on=["game_id", "season"])
    tot_p, tot_y, spr_p, spr_y = [], [], [], []
    for r in g.itertuples():
        pr = price(models[int(r.season)], r.lam_h, r.lam_a)
        T, M = r.home_score + r.away_score, r.home_score - r.away_score
        for L in CAL_TOTAL_LINES:
            o, u, _ = prob_total(pr, L)
            if T != L:
                tot_p.append(o / (o + u)); tot_y.append(T > L)
        for h in CAL_SPREAD_LINES:
            c, l_, _ = prob_spread(pr, h)
            if M + h != 0:
                spr_p.append(c / (c + l_)); spr_y.append(M + h > 0)
    return {"totals": _platt(tot_p, tot_y), "spreads": _platt(spr_p, spr_y)}


def apply_cal(p: float, ab: tuple[float, float] | None) -> float:
    if ab is None:
        return p
    p = min(max(p, 1e-6), 1 - 1e-6)
    z = ab[0] + ab[1] * np.log(p / (1 - p))
    return float(1 / (1 + np.exp(-z)))


def fit_final(con) -> GoalModel:
    return fit(side_frame(con), game_outcomes(con))


def fixture_sides(form: dict, fixtures: pd.DataFrame, rest_cap: int = 4,
                  goalie_exp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Two side rows per upcoming fixture from each team's latest form (features.latest_form) and the
    expected starters (goalies.upcoming; average goalie if not given)."""
    eg = {} if goalie_exp is None else {(r.game_id, r.team): r.exp_gsax30 for r in goalie_exp.itertuples()}
    rows = []
    for f in fixtures.itertuples():
        h, a = form.get(f.home), form.get(f.away)
        if h is None or a is None:
            continue
        date = pd.to_datetime(f.game_date)
        rest = {t: min(max((date - form[t]["last_game_date"]).days - 1, 0), rest_cap) for t in (f.home, f.away)}
        for own, opp, is_home in ((f.home, f.away, 1), (f.away, f.home, 0)):
            rows.append({"game_id": f.game_id, "is_home": is_home,
                         "own_pre_xgf": form[own]["xgf"], "own_pre_gf": form[own]["gf"],
                         "opp_pre_xga": form[opp]["xga"], "opp_pre_ga": form[opp]["ga"],
                         "own_b2b": int(rest[own] == 0), "opp_b2b": int(rest[opp] == 0),
                         "own_rest": rest[own], "opp_rest": rest[opp],
                         "opp_exp_gsax30": eg.get((f.game_id, opp), 0.0)})
    return pd.DataFrame(rows)
