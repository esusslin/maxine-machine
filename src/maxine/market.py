"""Market prices per game: opening and closing no-vig home win probability (moneyline, includes OT/SO).

Closing line = the last snapshot taken at or before puck drop. Each book's two prices are de-vigged
(proportional method), then the median across books is the consensus probability.
"""
import pyarrow as pa

from maxine.teams import ODDS_NAME_TO_NHL

TS = "COALESCE(try_strptime(taken_at, '%Y-%m-%dT%H:%M:%SZ'), try_strptime(taken_at, '%Y-%m-%dT%H-%M-%SZ'))"


def build_closing_lines(con) -> dict:
    names = list(ODDS_NAME_TO_NHL)
    con.register("_team_map", pa.table({"name": names, "abbrev": [ODDS_NAME_TO_NHL[n] for n in names]}))
    con.execute(f"""
        CREATE OR REPLACE TABLE market_lines AS
        WITH o AS (
            SELECT {TS} AS ts, s.commence_time, hm.abbrev AS home, aw.abbrev AS away,
                   s.home AS home_name, s.away AS away_name, s.book, s.outcome, s.price
            FROM odds_snapshots s
            JOIN _team_map hm ON hm.name = s.home
            JOIN _team_map aw ON aw.name = s.away
            WHERE s.league = 'nhl' AND s.market = 'h2h' AND s.price > 1
        ),
        g AS (SELECT game_id, start_utc, home, away FROM games WHERE league = 'NHL' AND game_type = 2),
        j AS (
            SELECT g.game_id, o.ts, o.book,
                   max(o.price) FILTER (WHERE o.outcome = o.home_name) AS home_price,
                   max(o.price) FILTER (WHERE o.outcome = o.away_name) AS away_price
            FROM g JOIN o ON o.home = g.home AND o.away = g.away
                 AND abs(epoch(o.commence_time) - epoch(g.start_utc)) < 12 * 3600
            WHERE o.ts <= g.start_utc
            GROUP BY ALL
        ),
        bk AS (
            SELECT *, (1 / home_price) / (1 / home_price + 1 / away_price) AS p_home_book,
                   1 / home_price + 1 / away_price - 1 AS overround
            FROM j WHERE home_price IS NOT NULL AND away_price IS NOT NULL
        ),
        snap AS (
            SELECT game_id, ts, median(p_home_book) AS p_home, count(*) AS n_books,
                   max(home_price) AS best_home_price, max(away_price) AS best_away_price,
                   median(overround) AS median_overround
            FROM bk GROUP BY game_id, ts
        )
        SELECT c.game_id,
               c.ts AS close_ts, c.p_home AS p_home_close, c.n_books AS n_books_close,
               c.best_home_price AS best_home_price_close, c.best_away_price AS best_away_price_close,
               c.median_overround AS overround_close,
               op.ts AS open_ts, op.p_home AS p_home_open
        FROM (SELECT * FROM snap QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY ts DESC) = 1) c
        JOIN (SELECT * FROM snap QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY ts ASC) = 1) op
          USING (game_id)
    """)
    con.unregister("_team_map")
    total = con.execute("SELECT count(*) FROM games WHERE league = 'NHL' AND game_type = 2").fetchone()[0]
    matched, mins = con.execute("""
        SELECT count(*), median(epoch(g.start_utc) - epoch(m.close_ts)) / 60
        FROM market_lines m JOIN games g USING (game_id)""").fetchone()
    return {"games": total, "with_closing_line": matched,
            "median_minutes_before_puck": round(mins or 0, 1)}
