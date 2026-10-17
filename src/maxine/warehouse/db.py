from importlib.resources import files

import duckdb

from maxine.config import DB_PATH


def connect() -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute(files("maxine.warehouse").joinpath("schema.sql").read_text())
    return con
