"""Home win probability (moneyline, OT/SO included) from pre-game features. Logistic regression.

Walk-forward by season: season S is predicted by a model fit only on earlier seasons.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from maxine.features import FEATURES


def walk_forward(df: pd.DataFrame, first_test_season: int, features: list[str] = FEATURES):
    """Returns ({game_id: p_home}, {season: coefficient dict})."""
    usable = df.dropna(subset=features)
    preds, coefs = {}, {}
    for season in sorted(s for s in usable["season"].unique() if s >= first_test_season):
        train = usable[usable["season"] < season]
        test = usable[usable["season"] == season]
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        model.fit(train[features], train["home_win"])
        p = model.predict_proba(test[features])[:, 1]
        preds.update(dict(zip(test["game_id"], p)))
        lr = model[-1]
        coefs[int(season)] = {"intercept": round(float(lr.intercept_[0]), 3),
                              **{f: round(float(c), 3) for f, c in zip(features, lr.coef_[0])}}
    return preds, coefs
