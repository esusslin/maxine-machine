CREATE TABLE IF NOT EXISTS games (
    game_id          BIGINT PRIMARY KEY,
    league           VARCHAR NOT NULL,      -- 'NHL', later 'AHL', 'ECHL', 'NCAA'
    season           INTEGER NOT NULL,      -- e.g. 20242025
    game_type        INTEGER NOT NULL,      -- 2 regular season, 3 playoffs
    game_date        DATE NOT NULL,         -- local game date
    start_utc        TIMESTAMP,
    home             VARCHAR NOT NULL,      -- team abbrev
    away             VARCHAR NOT NULL,
    home_team_id     INTEGER,
    away_team_id     INTEGER,
    home_score       INTEGER,
    away_score       INTEGER,
    last_period_type VARCHAR,               -- REG, OT, SO
    venue            VARCHAR
);

CREATE TABLE IF NOT EXISTS team_games (
    game_id  BIGINT NOT NULL,
    team     VARCHAR NOT NULL,
    is_home  BOOLEAN NOT NULL,
    goals    INTEGER,
    sog      INTEGER,
    PRIMARY KEY (game_id, team)
);

CREATE TABLE IF NOT EXISTS goalie_games (
    game_id        BIGINT NOT NULL,
    team           VARCHAR NOT NULL,
    player_id      BIGINT NOT NULL,
    name           VARCHAR,
    starter        BOOLEAN,
    toi_seconds    INTEGER,
    shots_against  INTEGER,
    saves          INTEGER,
    goals_against  INTEGER,
    decision       VARCHAR,
    PRIMARY KEY (game_id, player_id)
);

CREATE TABLE IF NOT EXISTS shots (
    game_id         BIGINT NOT NULL,
    event_id        INTEGER NOT NULL,
    period          INTEGER,
    time_in_period  VARCHAR,
    situation_code  VARCHAR,
    event_type      VARCHAR,   -- shot-on-goal, goal, missed-shot, blocked-shot
    owner_team_id   INTEGER,   -- NOTE: for blocked-shot this is believed to be the blocking team; verify
    shooter_id      BIGINT,
    goalie_id       BIGINT,
    shot_type       VARCHAR,
    x               DOUBLE,
    y               DOUBLE,
    PRIMARY KEY (game_id, event_id)
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    taken_at       VARCHAR NOT NULL,
    league         VARCHAR NOT NULL,
    event_id       VARCHAR NOT NULL,
    commence_time  TIMESTAMP,
    home           VARCHAR,
    away           VARCHAR,
    book           VARCHAR NOT NULL,
    market         VARCHAR NOT NULL,   -- h2h, spreads, totals
    outcome        VARCHAR NOT NULL,
    price          DOUBLE,             -- decimal odds
    point          DOUBLE,             -- spread or total line
    PRIMARY KEY (taken_at, league, event_id, book, market, outcome)
);

CREATE TABLE IF NOT EXISTS betnhl_picks (
    pick_date            DATE NOT NULL,
    idx                  INTEGER NOT NULL,
    pick_text            VARCHAR,
    first_decimal_odds   DOUBLE,
    first_american_odds  INTEGER,
    extractor            VARCHAR,
    PRIMARY KEY (pick_date, idx)
);
