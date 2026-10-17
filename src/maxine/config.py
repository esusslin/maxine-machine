import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("MAXINE_DATA_DIR", "./data")).resolve()
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "maxine.duckdb"
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

NHL_BASE = "https://api-web.nhle.com/v1"
ODDS_BASE = "https://api.the-odds-api.com/v4"
BETNHL_BASE = "https://www.betnhl.ca"

USER_AGENT = "maxine-machine/0.1 (personal research)"
