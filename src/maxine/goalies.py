"""Expected starting goalie, before the starter is confirmed.

Early-line bets are placed ~1-2 days before puck drop, when teams have not named a starter. So instead of
the actual starter (which the box score gives us only after the game), each team's goalie input is the
probability-weighted quality of the goalies who might start:

  1. Candidates: every goalie who started one of the team's last 20 games.
  2. P(start) per candidate: logistic regression on his share of the team's last 5 / 10 / 20 starts, whether
     he started the team's previous game, whether the team is on a back-to-back (and the interaction: the
     goalie who played last night usually sits), and days since his last start. Scores are normalised
     within each team-game so they sum to 1. Walk-forward: season S uses a model fit on earlier seasons.
  3. Quality: each goalie's career goals saved above expected per 30 shots before the game, shrunk toward
     0 (league average) with 3000 pseudo-shots, as in features.goalie_form.
  4. Expected quality = sum over candidates of P(start) x quality.

Out of sample 2023-26 the most likely candidate starts 67.5% of the time (docs/experiments.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

GSAX_PSEUDO_SHOTS = 3000
WINDOW = 20
STARTER_FEATURES = ["share5", "share10", "share20", "started_last", "b2b", "b2b_started_last", "days_since"]
DAYS_CAP = 30


def _team_starts(con) -> pd.DataFrame:
    """One row per team per completed regular-season game, with the team's starting goalie."""
    df = con.execute("""
        SELECT t.game_id, t.team, g.season, g.start_utc, g.game_date, gg.player_id AS starter
        FROM (SELECT game_id, home AS team FROM games UNION ALL SELECT game_id, away FROM games) t
        JOIN games g USING (game_id)
        LEFT JOIN goalie_games gg ON gg.game_id = t.game_id AND gg.team = t.team AND gg.starter
        WHERE g.league = 'NHL' AND g.game_type = 2 AND g.home_score IS NOT NULL
        ORDER BY t.team, g.start_utc, t.game_id
    """).df()
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _candidate_rows(starters: list, dates: list, last_start: dict, date, base: dict) -> list[dict]:
    """Candidate rows for one team-game given the team's starters/dates before it."""
    prev = [p for p in starters[-WINDOW:] if p == p]
    if not prev:
        return []
    b2b = int(bool(dates) and (date - dates[-1]).days == 1)
    last = starters[-1] if starters else None
    l10, l5 = starters[-10:], starters[-5:]
    rows = []
    for p in set(prev):
        ls = last_start.get(p)
        started_last = int(last == p)
        rows.append({**base, "player_id": int(p),
                     "share5": sum(x == p for x in l5) / len(l5),
                     "share10": sum(x == p for x in l10) / len(l10),
                     "share20": sum(x == p for x in prev) / len(prev),
                     "started_last": started_last, "b2b": b2b, "b2b_started_last": b2b * started_last,
                     "days_since": min((date - ls).days, DAYS_CAP) if ls is not None else DAYS_CAP})
    return rows


def candidate_frame(con) -> pd.DataFrame:
    """Historical candidates per team-game with features and y = 1 if he started."""
    ts = _team_starts(con)
    rows = []
    for team, g in ts.groupby("team", sort=False):
        starters, dates, last_start = [], [], {}
        for r in g.itertuples(index=False):
            for c in _candidate_rows(starters, dates, last_start, r.game_date,
                                     {"game_id": r.game_id, "team": team, "season": r.season, "start_utc": r.start_utc}):
                c["y"] = int(c["player_id"] == r.starter)
                rows.append(c)
            starters.append(r.starter)
            dates.append(r.game_date)
            if r.starter == r.starter:
                last_start[r.starter] = r.game_date
    return pd.DataFrame(rows)


def fit_starter(train: pd.DataFrame) -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=2000).fit(train[STARTER_FEATURES], train["y"])


def _normalise(model, c: pd.DataFrame) -> pd.Series:
    s = np.exp(model.decision_function(c[STARTER_FEATURES]))
    s = pd.Series(s, index=c.index)
    return s / s.groupby([c["game_id"], c["team"]]).transform("sum")


def _ratings(con) -> pd.DataFrame:
    """Cumulative shots faced and GSAx per goalie after each game (for as-of lookups)."""
    f = con.execute("""
        SELECT x.goalie_id AS player_id, g.start_utc, count(*) AS shots, sum(x.xg) - sum(x.goal) AS gsax
        FROM shot_xg x JOIN games g USING (game_id)
        WHERE x.xg IS NOT NULL AND x.goalie_id IS NOT NULL
        GROUP BY ALL ORDER BY player_id, start_utc
    """).df()
    grp = f.groupby("player_id")
    f["cum_shots"], f["cum_gsax"] = grp["shots"].cumsum(), grp["gsax"].cumsum()
    return f[["player_id", "start_utc", "cum_shots", "cum_gsax"]]


def _rate(c: pd.DataFrame, ratings: pd.DataFrame) -> pd.Series:
    """Each candidate's shrunk GSAx/30 from games strictly before c.start_utc."""
    c = c.reset_index()
    c["start_utc"] = pd.to_datetime(c["start_utc"]).astype("datetime64[ns]")
    c["player_id"] = c["player_id"].astype("int64")
    c = c.sort_values("start_utc")
    r = ratings.rename(columns={"start_utc": "t"})
    r["t"] = pd.to_datetime(r["t"]).astype("datetime64[ns]")
    r["player_id"] = r["player_id"].astype("int64")
    r = r.sort_values("t")
    m = pd.merge_asof(c, r, left_on="start_utc", right_on="t", by="player_id", allow_exact_matches=False)
    q = 30 * m["cum_gsax"].fillna(0) / (m["cum_shots"].fillna(0) + GSAX_PSEUDO_SHOTS)
    return pd.Series(q.to_numpy(), index=m["index"]).sort_index()


def expected(con) -> pd.DataFrame:
    """(game_id, team, exp_gsax30) for every completed game, starter model walk-forward by season."""
    c = candidate_frame(con)
    seasons = sorted(c["season"].unique())
    c["p"] = np.nan
    for s in seasons:
        train = c[c["season"] < s]
        if train["season"].nunique() < 1:
            train = c[c["season"] == s]         # first season only: in-sample (it is never a test season)
        model = fit_starter(train)
        idx = c["season"] == s
        c.loc[idx, "p"] = _normalise(model, c[idx])
    c["q"] = _rate(c, _ratings(con))
    c["pq"] = c["p"] * c["q"]
    return c.groupby(["game_id", "team"], as_index=False)["pq"].sum().rename(columns={"pq": "exp_gsax30"})


def upcoming(con, fixtures: pd.DataFrame) -> pd.DataFrame:
    """(game_id, team, exp_gsax30, likely_goalie, likely_p) for upcoming fixtures (game_id, game_date, home, away).

    Uses each team's completed games only, so a team that plays again before the fixture (not a clean game)
    is priced as if that game had not happened.
    """
    ts = _team_starts(con)
    model = fit_starter(candidate_frame(con))
    names = dict(con.execute("SELECT player_id, any_value(name) FROM goalie_games GROUP BY player_id").fetchall())
    ratings = _ratings(con)
    latest = ratings.groupby("player_id").last()
    by_team = {t: g for t, g in ts.groupby("team")}
    rows = []
    for f in fixtures.itertuples():
        date = pd.to_datetime(f.game_date)
        for team in (f.home, f.away):
            g = by_team.get(team)
            if g is None:
                continue
            starters, dates = g["starter"].tolist(), g["game_date"].tolist()
            last_start = {}
            for p, d in zip(starters, dates):
                if p == p:
                    last_start[p] = d
            cands = _candidate_rows(starters, dates, last_start, date, {"game_id": f.game_id, "team": team})
            if not cands:
                continue
            c = pd.DataFrame(cands)
            c["p"] = _normalise(model, c)
            c["q"] = [30 * latest["cum_gsax"].get(p, 0.0) / (latest["cum_shots"].get(p, 0.0) + GSAX_PSEUDO_SHOTS)
                      for p in c["player_id"]]
            top = c.loc[c["p"].idxmax()]
            rows.append({"game_id": f.game_id, "team": team, "exp_gsax30": float((c["p"] * c["q"]).sum()),
                         "likely_goalie": names.get(int(top.player_id), str(int(top.player_id))),
                         "likely_p": round(float(top.p), 2)})
    return pd.DataFrame(rows, columns=["game_id", "team", "exp_gsax30", "likely_goalie", "likely_p"])
