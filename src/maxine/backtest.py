"""Walk-forward scorecard: models vs the betting market, per season.

Metrics on home-win probability (moneyline result, OT/SO included):
log loss (lower is better), Brier score, accuracy. Every model is scored on the same games:
those that have a closing line.
"""
import math

from maxine.models import elo

EPS = 1e-6


def _metrics(pairs: list[tuple[float, int]]) -> dict:
    n = len(pairs)
    ll = -sum(y * math.log(max(p, EPS)) + (1 - y) * math.log(max(1 - p, EPS)) for p, y in pairs) / n
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    acc = sum((p > 0.5) == (y == 1) for p, y in pairs) / n
    return {"n": n, "log_loss": round(ll, 4), "brier": round(brier, 4), "accuracy": round(acc, 3)}


def calibration(pairs: list[tuple[float, int]], bins: int = 10) -> list[tuple[str, int, float, float]]:
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(p, y) for p, y in pairs if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if sel:
            out.append((f"{lo:.1f}-{hi:.1f}", len(sel), round(sum(p for p, _ in sel) / len(sel), 3),
                        round(sum(y for _, y in sel) / len(sel), 3)))
    return out


def load_games(con) -> list[tuple]:
    return con.execute("""
        SELECT game_id, season, home, away, home_score, away_score
        FROM games WHERE league = 'NHL' AND game_type = 2 ORDER BY start_utc, game_id
    """).fetchall()


def tune_elo(games, market: dict, train_seasons: set[int]) -> elo.EloParams:
    """Grid-search K and home advantage on the training seasons only."""
    best, best_ll = None, float("inf")
    for k in (4, 6, 8, 10, 12, 16):
        for ha in (20, 30, 40, 50, 60):
            for carry in (0.5, 0.7, 0.85):
                params = elo.EloParams(k=k, home_adv=ha, carryover=carry)
                preds = elo.run(games, params)
                pairs = [(preds[g[0]], int(g[4] > g[5])) for g in games if g[1] in train_seasons and g[0] in market]
                ll = _metrics(pairs)["log_loss"]
                if ll < best_ll:
                    best, best_ll = params, ll
    return best


def run(con, warmup_season: int = 20212022) -> dict:
    games = load_games(con)
    market = dict(con.execute("SELECT game_id, p_home_close FROM market_lines").fetchall())
    market_open = dict(con.execute("SELECT game_id, p_home_open FROM market_lines").fetchall())
    seasons = sorted({g[1] for g in games if g[1] != warmup_season})
    train, test = set(seasons[:2]), set(seasons[2:])

    tuned = tune_elo(games, market, train)
    models = {
        "elo_default": elo.run(games, elo.EloParams()),
        "elo_tuned": elo.run(games, tuned),
        "market_open": market_open,
        "market_close": market,
    }
    # home baseline: home-win rate of all earlier games (no look-ahead)
    home_base, wins, seen = {}, 0, 0
    for g in games:
        home_base[g[0]] = wins / seen if seen else 0.54
        wins += int(g[4] > g[5])
        seen += 1
    models = {"home_rate": home_base, **models}

    by_season = {}
    for season in seasons:
        sg = [g for g in games if g[1] == season and g[0] in market]
        by_season[season] = {name: _metrics([(preds[g[0]], int(g[4] > g[5])) for g in sg])
                             for name, preds in models.items()}
    test_games = [g for g in games if g[1] in test and g[0] in market]
    overall_test = {name: _metrics([(preds[g[0]], int(g[4] > g[5])) for g in test_games])
                    for name, preds in models.items()}
    calib = calibration([(models["elo_tuned"][g[0]], int(g[4] > g[5])) for g in test_games])
    return {"tuned_params": tuned, "train_seasons": sorted(train), "test_seasons": sorted(test),
            "by_season": by_season, "test_overall": overall_test, "elo_tuned_calibration_test": calib}


def print_report(r: dict) -> None:
    p = r["tuned_params"]
    print(f"Elo tuned on {r['train_seasons']}: K={p.k} home_adv={p.home_adv} carryover={p.carryover}")
    print(f"Test seasons (out of sample for tuning): {r['test_seasons']}\n")
    for season, rows in r["by_season"].items():
        tag = "test" if season in r["test_seasons"] else "train"
        print(f"{season} ({tag})")
        print(f"  {'model':<14}{'n':>6}{'log_loss':>10}{'brier':>8}{'acc':>7}")
        for name, m in rows.items():
            print(f"  {name:<14}{m['n']:>6}{m['log_loss']:>10}{m['brier']:>8}{m['accuracy']:>7}")
    print("\nTest seasons combined")
    for name, m in r["test_overall"].items():
        print(f"  {name:<14}{m['n']:>6}{m['log_loss']:>10}{m['brier']:>8}{m['accuracy']:>7}")
    print("\nElo (tuned) calibration, test seasons: bin, games, predicted, actual")
    for row in r["elo_tuned_calibration_test"]:
        print("  ", row)
