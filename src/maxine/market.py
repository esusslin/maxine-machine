"""Market prices per game: opening, morning and closing no-vig home win probability (moneyline, OT/SO incl.).

Opening line = the earliest snapshot we have (often the day before).
Morning line = the first snapshot at or after 14:00 UTC on game day (about 7-8 a.m. PT), after the
previous night's games are final. This is the fair "bet time" for a model whose features include
those games.
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
        g AS (SELECT game_id, start_utc, game_date, home, away FROM games WHERE league = 'NHL' AND game_type = 2),
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
            SELECT game_id, ts, ts >= CAST(any_value(game_date) AS TIMESTAMP) + INTERVAL 14 HOUR AS game_day, median(p_home_book) AS p_home, count(*) AS n_books,
                   max(home_price) AS best_home_price, max(away_price) AS best_away_price,
                   median(home_price) AS med_home_price, median(away_price) AS med_away_price,
                   median(overround) AS median_overround
            FROM bk JOIN g USING (game_id) GROUP BY game_id, ts
        )
        SELECT c.game_id,
               c.ts AS close_ts, c.p_home AS p_home_close, c.n_books AS n_books_close,
               c.best_home_price AS best_home_price_close, c.best_away_price AS best_away_price_close,
               c.median_overround AS overround_close,
               c.med_home_price AS med_home_price_close, c.med_away_price AS med_away_price_close,
               op.ts AS open_ts, op.p_home AS p_home_open,
               op.med_home_price AS med_home_price_open, op.med_away_price AS med_away_price_open,
               op.best_home_price AS best_home_price_open, op.best_away_price AS best_away_price_open,
               mo.ts AS morning_ts, mo.p_home AS p_home_morning,
               mo.med_home_price AS med_home_price_morning, mo.med_away_price AS med_away_price_morning
        FROM (SELECT * FROM snap QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY ts DESC) = 1) c
        JOIN (SELECT * FROM snap QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY ts ASC) = 1) op
          USING (game_id)
        LEFT JOIN (SELECT * FROM snap WHERE game_day
                   QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY ts ASC) = 1) mo
          USING (game_id)
    """)
    con.unregister("_team_map")
    total = con.execute("SELECT count(*) FROM games WHERE league = 'NHL' AND game_type = 2").fetchone()[0]
    matched, mins = con.execute("""
        SELECT count(*), median(epoch(g.start_utc) - epoch(m.close_ts)) / 60
        FROM market_lines m JOIN games g USING (game_id)""").fetchone()
    morning, hrs = con.execute("""
        SELECT count(morning_ts), median(epoch(g.start_utc) - epoch(m.morning_ts)) / 3600
        FROM market_lines m JOIN games g USING (game_id)""").fetchone()
    return {"games": total, "with_closing_line": matched,
            "median_minutes_before_puck": round(mins or 0, 1),
            "with_morning_line": morning, "morning_median_hours_before_puck": round(hrs or 0, 1)}


def build_line_markets(con) -> dict:
    """Spreads and totals per game and snapshot, by line: median no-vig probability of side A
    (A = home covers for spreads, keyed by the HOME line; A = over for totals), book count, prices.
    Table: market_points(game_id, ts, market, line, p_a, n_books, med_a, med_b, best_a, best_b)."""
    names = list(ODDS_NAME_TO_NHL)
    con.register("_team_map", pa.table({"name": names, "abbrev": [ODDS_NAME_TO_NHL[n] for n in names]}))
    con.execute(f"""
        CREATE OR REPLACE TABLE market_points AS
        WITH o AS (
            SELECT {TS} AS ts, s.commence_time, hm.abbrev AS home, aw.abbrev AS away, s.home AS home_name,
                   s.away AS away_name, s.book, s.market, s.outcome, s.price, s.point
            FROM odds_snapshots s
            JOIN _team_map hm ON hm.name = s.home JOIN _team_map aw ON aw.name = s.away
            WHERE s.league = 'nhl' AND s.market IN ('spreads', 'totals') AND s.price > 1 AND s.point IS NOT NULL
        ),
        g AS (SELECT game_id, start_utc, home, away FROM games WHERE league = 'NHL' AND game_type = 2),
        j AS (
            SELECT g.game_id, o.ts, o.book, o.market,
                   CASE WHEN o.market = 'spreads' THEN max(o.point) FILTER (WHERE o.outcome = o.home_name)
                        ELSE max(o.point) FILTER (WHERE o.outcome = 'Over') END AS line,
                   CASE WHEN o.market = 'spreads' THEN max(o.price) FILTER (WHERE o.outcome = o.home_name)
                        ELSE max(o.price) FILTER (WHERE o.outcome = 'Over') END AS a_price,
                   CASE WHEN o.market = 'spreads' THEN max(o.price) FILTER (WHERE o.outcome = o.away_name)
                        ELSE max(o.price) FILTER (WHERE o.outcome = 'Under') END AS b_price
            FROM g JOIN o ON o.home = g.home AND o.away = g.away
                 AND abs(epoch(o.commence_time) - epoch(g.start_utc)) < 12 * 3600
            WHERE o.ts <= g.start_utc
            GROUP BY g.game_id, o.ts, o.book, o.market
        )
        SELECT game_id, ts, market, line,
               median((1 / a_price) / (1 / a_price + 1 / b_price)) AS p_a, count(*) AS n_books,
               median(a_price) AS med_a, median(b_price) AS med_b, max(a_price) AS best_a, max(b_price) AS best_b
        FROM j WHERE line IS NOT NULL AND a_price IS NOT NULL AND b_price IS NOT NULL
        GROUP BY game_id, ts, market, line
    """)
    con.unregister("_team_map")
    return dict(con.execute("SELECT market, count(*) FROM market_points GROUP BY 1").fetchall())


def build_reg3(con) -> int:
    """Regulation 3-way (home / draw / away after 60 minutes) from the EU feed (nhl_eu): books whose
    moneyline includes a Draw. Each book de-vigged proportionally across its three prices; median across
    books. Table: market_reg3(game_id, ts, p_home, p_draw, p_away, n_books, med_home, med_draw, med_away,
    best_home, best_draw, best_away)."""
    has = con.execute("SELECT count(*) FROM odds_snapshots WHERE league = 'nhl_eu'").fetchone()[0]
    if not has:
        con.execute("CREATE OR REPLACE TABLE market_reg3 (game_id BIGINT, ts TIMESTAMP, p_home DOUBLE, p_draw DOUBLE,"
                    " p_away DOUBLE, n_books BIGINT, med_home DOUBLE, med_draw DOUBLE, med_away DOUBLE,"
                    " best_home DOUBLE, best_draw DOUBLE, best_away DOUBLE)")
        return 0
    names = list(ODDS_NAME_TO_NHL)
    con.register("_team_map", pa.table({"name": names, "abbrev": [ODDS_NAME_TO_NHL[n] for n in names]}))
    con.execute(f"""
        CREATE OR REPLACE TABLE market_reg3 AS
        WITH o AS (
            SELECT {TS} AS ts, s.commence_time, hm.abbrev AS home, aw.abbrev AS away, s.home AS home_name,
                   s.away AS away_name, s.book, s.outcome, s.price
            FROM odds_snapshots s JOIN _team_map hm ON hm.name = s.home JOIN _team_map aw ON aw.name = s.away
            WHERE s.league = 'nhl_eu' AND s.market = 'h2h' AND s.price > 1
        ),
        g AS (SELECT game_id, start_utc, home, away FROM games WHERE league = 'NHL' AND game_type = 2),
        b AS (
            SELECT g.game_id, o.ts, o.book,
                   max(o.price) FILTER (WHERE o.outcome = o.home_name) AS hp,
                   max(o.price) FILTER (WHERE o.outcome = 'Draw') AS dp,
                   max(o.price) FILTER (WHERE o.outcome = o.away_name) AS ap
            FROM g JOIN o ON o.home = g.home AND o.away = g.away
                 AND abs(epoch(o.commence_time) - epoch(g.start_utc)) < 12 * 3600
            WHERE o.ts <= g.start_utc GROUP BY ALL
        ),
        v AS (SELECT *, 1/hp + 1/dp + 1/ap AS tot FROM b WHERE hp IS NOT NULL AND dp IS NOT NULL AND ap IS NOT NULL)
        SELECT game_id, ts, median((1/hp)/tot) AS p_home, median((1/dp)/tot) AS p_draw, median((1/ap)/tot) AS p_away,
               count(*) AS n_books, median(hp) AS med_home, median(dp) AS med_draw, median(ap) AS med_away,
               max(hp) AS best_home, max(dp) AS best_draw, max(ap) AS best_away
        FROM v GROUP BY game_id, ts
    """)
    con.unregister("_team_map")
    return con.execute("SELECT count(*) FROM market_reg3").fetchone()[0]
