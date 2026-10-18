"""Home win probability (moneyline, OT/SO included) from pre-game features. Logistic regression.

Walk-forward by season: season S is predicted by a model fit only on earlier seasons.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from maxine.features import FEATURES


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def walk_forward(df: pd.DataFrame, first_test_season: int, features: list[str] = FEATURES,
                 calibrate: bool = True):
    """Returns ({game_id: p_home}, {season: coefficient dict}).

    Calibration: for season S, refit p -> sigmoid(a + b*logit(p)) on the model's own out-of-sample
    predictions for earlier seasons (b < 1 shrinks overconfident predictions). The first test season
    has no earlier out-of-sample predictions and is left uncalibrated.
    """
    usable = df.dropna(subset=features)
    preds, coefs = {}, {}
    oos = []  # (raw_p, y) from earlier test seasons
    for season in sorted(s for s in usable["season"].unique() if s >= first_test_season):
        train = usable[usable["season"] < season]
        test = usable[usable["season"] == season]
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        model.fit(train[features], train["home_win"])
        raw = model.predict_proba(test[features])[:, 1]
        p = raw
        cal = None
        if calibrate and oos:
            prior_p, prior_y = zip(*oos)
            cal = LogisticRegression(C=1e6).fit(_logit(prior_p).reshape(-1, 1), np.array(prior_y))
            p = cal.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
        oos += list(zip(raw, test["home_win"]))
        preds.update(dict(zip(test["game_id"], p)))
        lr = model[-1]
        coefs[int(season)] = {"intercept": round(float(lr.intercept_[0]), 3),
                              **{f: round(float(c), 3) for f, c in zip(features, lr.coef_[0])},
                              **({"calib_slope": round(float(cal.coef_[0][0]), 3),
                                  "calib_intercept": round(float(cal.intercept_[0]), 3)} if cal else {})}
    return preds, coefs
