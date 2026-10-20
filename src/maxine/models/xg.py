"""Expected goals (xG) for unblocked shots, NHL play-by-play.

Features: distance and angle to the net, shot type, rebound (team's previous unblocked shot within
3 seconds), rush proxy (previous event in the game within 4 seconds was by the other team),
manpower for the shooting team, period. Logistic regression.

Walk-forward: shots in season S are scored by a model fit only on seasons before S.
The first season is scored by a model fit on itself (it is the warm-up season, not evaluated).
Shootout attempts and shots at an empty net are excluded.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

NUMERIC = ["distance", "angle", "rebound", "rush", "skater_diff", "period_3", "ot"]
CATEGORICAL = ["shot_type"]


def shot_frame(con) -> pd.DataFrame:
    df = con.execute("""
        SELECT s.game_id, s.event_id, g.season, s.period, s.time_in_period, s.situation_code, s.event_type,
               s.owner_team_id, s.goalie_id, s.shot_type, s.x, s.y,
               g.home_team_id, g.away_team_id, g.home, g.away
        FROM shots s JOIN games g USING (game_id)
        WHERE g.league = 'NHL' AND g.game_type = 2
    """).df()
    t = df["time_in_period"].fillna("00:00").str.split(":", expand=True).astype(int)
    df["t"] = (df["period"] - 1) * 1200 + t[0] * 60 + t[1]
    df = df.sort_values(["game_id", "t", "event_id"]).reset_index(drop=True)

    sc = df["situation_code"].fillna("1551").str.zfill(4)
    away_g, away_sk = sc.str[0].astype(int), sc.str[1].astype(int)
    home_sk, home_g = sc.str[2].astype(int), sc.str[3].astype(int)
    df["is_home"] = df["owner_team_id"] == df["home_team_id"]
    df["team"] = np.where(df["is_home"], df["home"], df["away"])
    df["opp"] = np.where(df["is_home"], df["away"], df["home"])
    df["skater_diff"] = np.where(df["is_home"], home_sk - away_sk, away_sk - home_sk)
    df["own_skaters"] = np.where(df["is_home"], home_sk, away_sk)
    df["opp_skaters"] = np.where(df["is_home"], away_sk, home_sk)
    df["opp_goalie_in"] = np.where(df["is_home"], away_g, home_g)
    df["strength"] = np.select(
        [(df["own_skaters"] == 5) & (df["opp_skaters"] == 5), df["skater_diff"] > 0, df["skater_diff"] < 0],
        ["5v5", "pp", "sh"], "other")
    df["shootout"] = df["situation_code"].eq("1010") | (df["period"] >= 5)

    # rebound / rush from the previous shot-type event in the same game
    prev_team = df.groupby("game_id")["owner_team_id"].shift(1)
    prev_t = df.groupby("game_id")["t"].shift(1)
    prev_type = df.groupby("game_id")["event_type"].shift(1)
    dt = df["t"] - prev_t
    df["rebound"] = ((prev_team == df["owner_team_id"]) & (dt <= 3) & (prev_type != "blocked-shot")).astype(int)
    df["rush"] = ((prev_team != df["owner_team_id"]) & (dt <= 4)).astype(int)

    ax = df["x"].abs()
    df["distance"] = np.sqrt((89 - ax) ** 2 + df["y"] ** 2)
    df["angle"] = np.degrees(np.arctan2(df["y"].abs(), (89 - ax).clip(lower=0.1)))
    df["period_3"] = (df["period"] == 3).astype(int)
    df["ot"] = (df["period"] >= 4).astype(int)
    df["shot_type"] = df["shot_type"].fillna("unknown")
    df["goal"] = (df["event_type"] == "goal").astype(int)

    # score state before each event, from the shooting team's point of view (regulation + OT, not shootout)
    g = (df["event_type"] == "goal") & ~df["shootout"]
    home_goals = (g & df["is_home"]).groupby(df["game_id"]).cumsum() - (g & df["is_home"]).astype(int)
    away_goals = (g & ~df["is_home"]).groupby(df["game_id"]).cumsum() - (g & ~df["is_home"]).astype(int)
    diff = np.where(df["is_home"], home_goals - away_goals, away_goals - home_goals)
    df["score_diff"] = np.clip(diff, -2, 2)
    return df


def _model():
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return make_pipeline(pre, LogisticRegression(max_iter=1000))


SCORE_ADJ_FIT_BEFORE = 20232024   # weights fit on seasons before this one only (no look-ahead into tests)


def score_weights(scored: pd.DataFrame) -> dict[int, float]:
    """5v5 score-state weights: 0.5 / (shooting team's average share of xG in that score state).
    Trailing teams pile up shots (share > 50%) so their shots count less; leading teams' count more."""
    five = scored[(scored["strength"] == "5v5") & ~scored["empty_net"] & (scored["season"] < SCORE_ADJ_FIT_BEFORE)]
    by = five.groupby("score_diff")["xg"].sum()
    return {int(d): float(0.5 / (by[d] / (by[d] + by[-d]))) for d in by.index if -d in by.index}


def score_shots(con) -> dict:
    """Write shot_xg(game_id, event_id, ..., xg) for every unblocked, non-shootout, goalie-in shot."""
    df = shot_frame(con)
    unblocked = df[(df["event_type"] != "blocked-shot") & ~df["shootout"]].copy()
    unblocked["empty_net"] = unblocked["opp_goalie_in"] == 0
    fit_set = unblocked[~unblocked["empty_net"]]
    seasons = sorted(fit_set["season"].unique())
    parts = []
    for i, season in enumerate(seasons):
        train = fit_set[fit_set["season"].isin(seasons[:i] if i > 0 else seasons[:1])]
        model = _model().fit(train[NUMERIC + CATEGORICAL], train["goal"])
        cur = unblocked[unblocked["season"] == season].copy()
        cur["xg"] = model.predict_proba(cur[NUMERIC + CATEGORICAL])[:, 1]
        cur.loc[cur["empty_net"], "xg"] = np.nan          # not a goalie-facing shot
        parts.append(cur)
    out = pd.concat(parts)
    w = score_weights(out)
    adj = out["score_diff"].map(w).fillna(1.0)
    out["xg_adj"] = np.where(out["strength"] == "5v5", out["xg"] * adj, out["xg"])
    out = out[["game_id", "event_id", "season", "team", "opp", "is_home", "strength", "score_diff",
               "empty_net", "goalie_id", "goal", "xg", "xg_adj"]]
    con.register("_xg", out)
    con.execute("CREATE OR REPLACE TABLE shot_xg AS SELECT * FROM _xg")
    con.unregister("_xg")
    ev = out.dropna(subset=["xg"])
    by_season = ev.groupby("season").agg(shots=("goal", "size"), goals=("goal", "sum"), xg=("xg", "sum"))
    return {"shots_scored": int(len(ev)), "score_weights": {k: round(v, 3) for k, v in w.items()},
            "goals_vs_xg_by_season": {int(s): (int(r.goals), round(float(r.xg))) for s, r in by_season.iterrows()}}
