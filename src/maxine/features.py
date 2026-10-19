"""Pre-game features per NHL game. Every value uses only games before puck drop.

Team form: exponentially weighted (half-life in games) 5v5 xG share, all-situation xG for/against
per game, goal share. Continues across seasons (no reset), so early-season games lean on last season.
Goalie: starter's career GSAx per shot faced before this game, shrunk toward 0 with a pseudo-count.
Rest: days since the team's previous game (capped), back-to-back flag.

Known look-ahead: the starting goalie comes from the box score. Live, it must come from confirmed
starters (Daily Faceoff / news agent) before puck drop.
"""
import numpy as np
import pandas as pd

from maxine.models import elo
from maxine.teams import FRANCHISE_PREDECESSOR

HALF_LIFE_GAMES = 25
GSAX_PSEUDO_SHOTS = 3000
REST_CAP = 4


def _team_games(con) -> pd.DataFrame:
    games = con.execute("""
        SELECT game_id, season, start_utc, game_date, home, away, home_score, away_score
        FROM games WHERE league = 'NHL' AND game_type = 2 ORDER BY start_utc, game_id
    """).df()
    xg = con.execute("""
        SELECT game_id, team,
               sum(xg) AS xgf, sum(CASE WHEN strength = '5v5' THEN xg END) AS xgf_5v5
        FROM shot_xg WHERE xg IS NOT NULL GROUP BY game_id, team
    """).df()
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        part = games[["game_id", "season", "start_utc", "game_date", side, opp,
                      f"{side}_score", f"{opp}_score"]].copy()
        part.columns = ["game_id", "season", "start_utc", "game_date", "team", "opp", "gf", "ga"]
        part["is_home"] = side == "home"
        rows.append(part)
    tg = pd.concat(rows, ignore_index=True)
    tg = tg.merge(xg, on=["game_id", "team"], how="left")
    opp_xg = xg.rename(columns={"team": "opp", "xgf": "xga", "xgf_5v5": "xga_5v5"})
    tg = tg.merge(opp_xg, on=["game_id", "opp"], how="left")
    return tg.fillna({"xgf": 0, "xga": 0, "xgf_5v5": 0, "xga_5v5": 0}).sort_values(["team", "start_utc"])


def _ewm_pregame(tg: pd.DataFrame, col: str) -> pd.Series:
    # franchise continuity: Utah's form continues from Arizona's
    franchise = tg["team"].map(lambda t: {v: k for k, v in FRANCHISE_PREDECESSOR.items()}.get(t, t))
    return tg.groupby(franchise)[col].transform(
        lambda s: s.shift(1).ewm(halflife=HALF_LIFE_GAMES, min_periods=5).mean())


def team_form(con) -> pd.DataFrame:
    tg = _team_games(con)
    tg = tg.sort_values("start_utc")
    for col in ("xgf", "xga", "xgf_5v5", "xga_5v5", "gf", "ga"):
        tg[f"pre_{col}"] = _ewm_pregame(tg, col)
    tg["pre_xg_share_5v5"] = tg["pre_xgf_5v5"] / (tg["pre_xgf_5v5"] + tg["pre_xga_5v5"])
    tg["pre_xg_net"] = tg["pre_xgf"] - tg["pre_xga"]
    tg["pre_goal_net"] = tg["pre_gf"] - tg["pre_ga"]
    prev_date = tg.groupby("team")["game_date"].shift(1)
    tg["rest_days"] = (pd.to_datetime(tg["game_date"]) - pd.to_datetime(prev_date)).dt.days.sub(1)
    tg["rest_days"] = tg["rest_days"].clip(upper=REST_CAP).fillna(REST_CAP)
    tg["b2b"] = (tg["rest_days"] == 0).astype(int)
    return tg


def goalie_form(con) -> pd.DataFrame:
    """Starter per team-game with pre-game shrunk GSAx per 30 shots."""
    faced = con.execute("""
        SELECT x.goalie_id AS player_id, x.game_id, g.start_utc,
               count(*) AS shots, sum(x.xg) AS xga, sum(x.goal) AS ga
        FROM shot_xg x JOIN games g USING (game_id)
        WHERE x.xg IS NOT NULL AND x.goalie_id IS NOT NULL
        GROUP BY ALL ORDER BY player_id, start_utc
    """).df()
    faced["gsax"] = faced["xga"] - faced["ga"]
    grp = faced.groupby("player_id")
    faced["cum_gsax"] = grp["gsax"].cumsum() - faced["gsax"]
    faced["cum_shots"] = grp["shots"].cumsum() - faced["shots"]
    faced["pre_gsax30"] = 30 * faced["cum_gsax"] / (faced["cum_shots"] + GSAX_PSEUDO_SHOTS)
    starters = con.execute("""
        SELECT game_id, team, player_id FROM goalie_games WHERE starter
    """).df()
    out = starters.merge(faced[["player_id", "game_id", "pre_gsax30", "cum_shots"]],
                         on=["player_id", "game_id"], how="left")
    return out.fillna({"pre_gsax30": 0.0, "cum_shots": 0})


# elo_logit is computed but not used by default: it added nothing out of sample once xG and goal form were in
FEATURES = ["xg_share_5v5_diff", "xg_net_diff", "goal_net_diff", "gsax30_diff",
            "home_b2b", "away_b2b", "rest_diff"]


def game_features(con) -> pd.DataFrame:
    tf = team_form(con)
    gf = goalie_form(con)
    tf = tf.merge(gf[["game_id", "team", "pre_gsax30"]], on=["game_id", "team"], how="left")
    tf["pre_gsax30"] = tf["pre_gsax30"].fillna(0.0)
    cols = ["game_id", "pre_xg_share_5v5", "pre_xg_net", "pre_goal_net", "pre_gsax30", "rest_days", "b2b"]
    h = tf[tf["is_home"]][cols + ["season", "start_utc", "gf", "ga"]].add_prefix("h_")
    a = tf[~tf["is_home"]][cols].add_prefix("a_")
    df = h.merge(a, left_on="h_game_id", right_on="a_game_id")
    df = df.rename(columns={"h_game_id": "game_id", "h_season": "season", "h_start_utc": "start_utc"})
    df["home_win"] = (df["h_gf"] > df["h_ga"]).astype(int)

    games = con.execute("""SELECT game_id, season, home, away, home_score, away_score
                           FROM games WHERE league='NHL' AND game_type=2 ORDER BY start_utc, game_id""").fetchall()
    elo_p = elo.run(games, elo.EloParams())
    df["elo_p"] = df["game_id"].map(elo_p)
    df["elo_logit"] = np.log(df["elo_p"] / (1 - df["elo_p"]))
    df["xg_share_5v5_diff"] = df["h_pre_xg_share_5v5"] - df["a_pre_xg_share_5v5"]
    df["xg_net_diff"] = df["h_pre_xg_net"] - df["a_pre_xg_net"]
    df["goal_net_diff"] = df["h_pre_goal_net"] - df["a_pre_goal_net"]
    df["gsax30_diff"] = df["h_pre_gsax30"] - df["a_pre_gsax30"]
    df["home_b2b"], df["away_b2b"] = df["h_b2b"], df["a_b2b"]
    df["rest_diff"] = df["h_rest_days"] - df["a_rest_days"]
    return df.sort_values("start_utc").reset_index(drop=True)


def latest_form(con) -> dict[str, dict]:
    """Each team's form after its most recent completed game (for pricing upcoming games)."""
    tg = _team_games(con).sort_values("start_utc")
    franchise = tg["team"].map(lambda t: {v: k for k, v in FRANCHISE_PREDECESSOR.items()}.get(t, t))
    tg["franchise"] = franchise
    out = {}
    for team, g in tg.groupby("franchise"):
        ew = {c: g[c].ewm(halflife=HALF_LIFE_GAMES).mean().iloc[-1]
              for c in ("xgf", "xga", "xgf_5v5", "xga_5v5", "gf", "ga")}
        out[team] = {
            "xg_share_5v5": ew["xgf_5v5"] / (ew["xgf_5v5"] + ew["xga_5v5"]),
            "xg_net": ew["xgf"] - ew["xga"],
            "goal_net": ew["gf"] - ew["ga"],
            "xgf": ew["xgf"], "xga": ew["xga"], "gf": ew["gf"], "ga": ew["ga"],
            "last_game_date": pd.to_datetime(g["game_date"].iloc[-1]),
            "games": len(g),
        }
    return out


def upcoming_features(con, fixtures: pd.DataFrame) -> pd.DataFrame:
    """fixtures: game_id, game_date, home, away. Same feature definitions as game_features."""
    form = latest_form(con)
    rows = []
    for f in fixtures.itertuples():
        h, a = form.get(f.home), form.get(f.away)
        if h is None or a is None:
            continue
        date = pd.to_datetime(f.game_date)
        rest_h = min(max((date - h["last_game_date"]).days - 1, 0), REST_CAP)
        rest_a = min(max((date - a["last_game_date"]).days - 1, 0), REST_CAP)
        rows.append({
            "game_id": f.game_id,
            "xg_share_5v5_diff": h["xg_share_5v5"] - a["xg_share_5v5"],
            "xg_net_diff": h["xg_net"] - a["xg_net"],
            "goal_net_diff": h["goal_net"] - a["goal_net"],
            "gsax30_diff": 0.0,
            "home_b2b": int(rest_h == 0), "away_b2b": int(rest_a == 0),
            "rest_diff": rest_h - rest_a,
        })
    return pd.DataFrame(rows)
