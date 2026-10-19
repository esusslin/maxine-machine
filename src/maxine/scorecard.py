"""Research-plane scorecard, written to reports/scorecard.json and committed.

Railway has no odds history (that stays local), so the Model tab shows this file.
Regenerate locally after model changes:  maxine scorecard
"""
import dataclasses
import json
from datetime import date
from pathlib import Path

from maxine import backtest, clv

OUT = Path("reports/scorecard.json")


def build(con) -> dict:
    bt = backtest.run(con)
    bt["tuned_params"] = dataclasses.asdict(bt["tuned_params"])
    bt["by_season"] = {str(k): v for k, v in bt["by_season"].items()}
    bt["shot_model_coefs"] = {str(k): v for k, v in bt["shot_model_coefs"].items()}
    return {"generated": date.today().isoformat(), "backtest": bt, "clv": clv.run(con)}


def write(con) -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(con), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return OUT
