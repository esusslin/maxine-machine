"""maxine CLI. Run `maxine --help` for commands."""
import argparse
from datetime import date, timedelta

from maxine.ingest import betnhl, nhl, odds, odds_backfill
from maxine import backtest, clv, market, picks, scorecard
from maxine.models import xg
from maxine.validate import validate_season
from maxine.warehouse import load
from maxine.warehouse.db import connect

BACKFILL_SEASONS = [20212022, 20222023, 20232024, 20242025, 20252026]


def cmd_nhl_season(args):
    for season in args.seasons:
        n = nhl.ingest_season(season, with_pbp=not args.no_pbp)
        print(f"{season}: {n} games fetched (raw JSON)")


def cmd_nhl_range(args):
    n = nhl.ingest_range(date.fromisoformat(args.start), date.fromisoformat(args.end),
                         with_pbp=not args.no_pbp, refresh_schedule=True)
    print(f"{n} games fetched")


def cmd_odds(args):
    leagues = list(odds.SPORTS) if args.league == "all" else [args.league]
    for lg in leagues:
        print(lg, odds.snapshot(lg))


def cmd_odds_backfill(args):
    con = connect()
    print(odds_backfill.run(con, args.seasons, dry_run=args.dry_run, reserve=args.reserve))


def cmd_betnhl(args):
    day = date.fromisoformat(args.date) if args.date else date.today()
    print(betnhl.snapshot(day))


def cmd_betnhl_backfill(args):
    print(betnhl.backfill(date.fromisoformat(args.start), date.fromisoformat(args.end)))


def cmd_load(args):
    con = connect()
    if getattr(args, "full", False):
        load.reset_loaded(con)
    print("nhl", load.load_nhl(con))
    print("live odds", load.load_odds(con))
    print("historical odds", load.load_odds_hist(con))
    print("betnhl picks", load.load_betnhl(con))


def cmd_validate(args):
    con = connect()
    failed = False
    for season in args.seasons:
        problems = validate_season(con, season)
        if problems:
            failed = True
            print(f"{season}: FAIL ({len(problems)} teams differ)")
            for p in problems:
                print("   ", p)
        else:
            print(f"{season}: PASS")
    raise SystemExit(1 if failed else 0)


def cmd_market(_args):
    con = connect()
    print(market.build_closing_lines(con))


def cmd_xg(_args):
    con = connect()
    print(xg.score_shots(con))


def cmd_backtest(_args):
    con = connect()
    backtest.print_report(backtest.run(con))


def cmd_clv(_args):
    con = connect()
    clv.print_report(clv.run(con))


def cmd_picks(args):
    con = connect()
    picks.print_report(picks.run(con, hours=args.hours, min_edge=args.min_edge, game_day=args.game_day))


def cmd_scorecard(_args):
    con = connect()
    print("wrote", scorecard.write(con))


def cmd_daily(_args):
    """Yesterday's results + today's odds and BetNHL picks, then load. Run once each morning."""
    today = date.today()
    yesterday = today - timedelta(days=1)
    print("nhl games", nhl.ingest_range(yesterday, yesterday, refresh_schedule=True))
    for lg in odds.SPORTS:
        try:
            print("odds", lg, odds.snapshot(lg))
        except SystemExit as e:
            print("odds skipped:", e)
            break
    try:
        print("betnhl", betnhl.snapshot(today))
    except Exception as e:  # page may not be posted yet
        print("betnhl skipped:", e)
    cmd_load(None)


def main():
    p = argparse.ArgumentParser(prog="maxine")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("nhl-season", help="backfill whole NHL seasons, e.g. 20232024")
    s.add_argument("seasons", nargs="*", type=int, default=BACKFILL_SEASONS)
    s.add_argument("--no-pbp", action="store_true", help="skip play-by-play (faster)")
    s.set_defaults(func=cmd_nhl_season)

    s = sub.add_parser("nhl-range", help="fetch NHL games between two dates (YYYY-MM-DD)")
    s.add_argument("start")
    s.add_argument("end")
    s.add_argument("--no-pbp", action="store_true")
    s.set_defaults(func=cmd_nhl_range)

    s = sub.add_parser("odds", help="snapshot current odds (uses API credits)")
    s.add_argument("--league", choices=["nhl", "ahl", "all"], default="all")
    s.set_defaults(func=cmd_odds)

    s = sub.add_parser("odds-backfill", help="historical NHL odds for loaded seasons (paid plan; run `maxine load` first)")
    s.add_argument("seasons", nargs="*", type=int, default=list(reversed(BACKFILL_SEASONS)))
    s.add_argument("--dry-run", action="store_true", help="count snapshots and credits, spend nothing")
    s.add_argument("--reserve", type=int, default=3000, help="stop before credits remaining drop below this")
    s.set_defaults(func=cmd_odds_backfill)

    s = sub.add_parser("betnhl", help="snapshot a BetNHL tips page")
    s.add_argument("--date", help="YYYY-MM-DD, default today")
    s.set_defaults(func=cmd_betnhl)

    s = sub.add_parser("betnhl-backfill", help="save BetNHL tips pages for a date range (YYYY-MM-DD YYYY-MM-DD)")
    s.add_argument("start")
    s.add_argument("end")
    s.set_defaults(func=cmd_betnhl_backfill)

    s = sub.add_parser("load", help="parse new or changed raw files into the DuckDB warehouse")
    s.add_argument("--full", action="store_true", help="re-parse every raw file, not just new ones")
    s.set_defaults(func=cmd_load)

    s = sub.add_parser("validate", help="Phase 0 gate: rebuilt standings vs official")
    s.add_argument("seasons", nargs="*", type=int, default=BACKFILL_SEASONS)
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("market", help="build opening/closing no-vig lines per game from odds snapshots")
    s.set_defaults(func=cmd_market)

    s = sub.add_parser("xg", help="fit the expected-goals model walk-forward and score every unblocked shot")
    s.set_defaults(func=cmd_xg)

    s = sub.add_parser("backtest", help="walk-forward scorecard: Elo and baselines vs the closing line")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("clv", help="edge test: line movement toward the model and ROI at the morning line")
    s.set_defaults(func=cmd_clv)

    s = sub.add_parser("picks", help="evening job: price upcoming games vs current odds, log early-line picks")
    s.add_argument("--hours", type=int, default=36, help="price games starting within this many hours")
    s.add_argument("--min-edge", type=float, default=0.05, help="minimum edge for a pick")
    s.add_argument("--game-day", action="store_true", help="also pick today's games (no backtested edge)")
    s.set_defaults(func=cmd_picks)

    s = sub.add_parser("scorecard", help="write reports/scorecard.json (backtest + CLV) for the web Model tab")
    s.set_defaults(func=cmd_scorecard)

    s = sub.add_parser("daily", help="morning job: results, odds, BetNHL picks, load")
    s.set_defaults(func=cmd_daily)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
