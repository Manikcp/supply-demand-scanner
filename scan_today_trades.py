"""
3-Step A+ Supply & Demand Strategy — TODAY'S TRADES SCANNER (Historical/Backtest Mode)

View today's trade alerts even when market is closed.
Scans all FNO stocks + Indices and displays signals for a specific date.

Usage:
    python3 scan_today_trades.py
    python3 scan_today_trades.py --date 2025-03-10
    python3 scan_today_trades.py --save
"""

import argparse
import sys
from datetime import datetime

import pandas as pd

from strategies.signals import Config, run_signals
from scanner_base import (
    TICKERS_FILE,
    INDICES_FILE,
    now_ist,
    get_last_trading_date,
    load_tickers,
    load_indices,
    fetch_ohlcv,
    ticker_label,
    format_alert_with_targets,
    progress_print,
    clear_progress,
    make_config,
)


def scan_date(tickers: list, cfg: Config, target_date) -> list:
    """
    Scan all tickers and return signals for a specific date.
    """
    all_alerts = []
    total = len(tickers)

    print(f"\n{'#'*62}")
    print(f"  3-Step A+ S&D Strategy — TODAY'S TRADES SCANNER")
    print(f"  Scan Date      : {target_date.strftime('%Y-%m-%d (%A)')}")
    print(f"  Total Tickers  : {len(tickers)}")
    print(f"  Data Interval  : 15m")
    print(f"{'#'*62}\n")

    for pos, ticker in enumerate(tickers, 1):
        progress_print(pos, total, ticker)

        df = fetch_ohlcv(ticker, interval="15m", period="30d")
        if df is None:
            continue

        df_daily = fetch_ohlcv(ticker, interval="1d", period="1y")
        df_weekly = fetch_ohlcv(ticker, interval="1wk", period="2y")
        df_h1 = fetch_ohlcv(ticker, interval="1h", period="21d")

        tcfg = make_config(ticker, cfg)

        try:
            _, signals = run_signals(df, df_daily, tcfg, df_weekly, df_h1=df_h1)
        except Exception as exc:
            sys.stdout.write(f"\r  [ERROR] {ticker}: {exc}\n")
            continue

        if signals.empty:
            continue

        signals["index"] = ticker_label(ticker)

        signals["_date"] = signals["datetime"].apply(
            lambda x: x.astimezone(now_ist().tzinfo).date() if hasattr(x, "astimezone") else x.date()
        )
        date_sigs = signals[signals["_date"] == target_date].drop(columns=["_date"])

        for _, row in date_sigs.iterrows():
            rec = row.to_dict()
            rec["ticker"] = ticker_label(ticker)
            rec["ticker_raw"] = ticker
            all_alerts.append(rec)
            print(format_alert_with_targets(ticker, rec))

    clear_progress()
    return all_alerts


def main():
    parser = argparse.ArgumentParser(
        description="TWP Algo v6 -- Today's Trades Scanner (Historical Mode)"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Specific date to scan (YYYY-MM-DD format). Default: today",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV filename (default: auto-generated)",
    )
    parser.add_argument(
        "--journal",
        action="store_true",
        help="Track trades in journal & check previous trade outcomes",
    )
    args = parser.parse_args()

    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print("  [ERROR] Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        target_date = get_last_trading_date()

    if args.journal:
        from trade_journal import check_open_trades, add_trade, print_status
        check_open_trades()

    fno_tickers = load_tickers(TICKERS_FILE)
    indices = load_indices(INDICES_FILE)

    print(f"\n  Loaded {len(fno_tickers)} FNO stocks")
    print(f"  Loaded {len(indices)} Indices: {', '.join([ticker_label(i) for i in indices])}")

    all_tickers = indices + fno_tickers
    cfg = Config.from_json()

    all_alerts = scan_date(all_tickers, cfg, target_date)

    print(f"\n{'='*62}")
    print(f"  SCAN COMPLETE")
    print(f"  Date        : {target_date.strftime('%Y-%m-%d (%A)')}")
    print(f"  Total Alerts: {len(all_alerts)}")
    print(f"{'='*62}")

    if args.journal and all_alerts:
        for alert in all_alerts:
            add_trade(alert)
        print()
        print_status()

    if all_alerts:
        summary = pd.DataFrame(all_alerts)
        cols = ["ticker", "type", "strike", "entry", "target", "sl", "rr", "entry_grade", "datetime"]
        available_cols = [c for c in cols if c in summary.columns]
        print(f"\n  == ALL ALERTS FOR {target_date} ==")
        print(summary[available_cols].to_string(index=False))

        ce_count = len([a for a in all_alerts if "CE" in a["type"]])
        pe_count = len([a for a in all_alerts if "PE" in a["type"]])

        print(f"\n  == BREAKDOWN ==")
        print(f"  CE Signals : {ce_count}")
        print(f"  PE Signals : {pe_count}")

        if args.save:
            if args.output:
                filename = args.output
            else:
                filename = f"trades_{target_date.strftime('%Y%m%d')}.csv"

            df_export = pd.DataFrame(all_alerts)
            cols_order = [
                "ticker", "type", "index", "strike", "entry",
                "target", "sl", "rr", "entry_grade", "entry_quality",
                "pattern", "factors", "datetime", "ticker_raw"
            ]
            available = [c for c in cols_order if c in df_export.columns]
            df_export[available].to_csv(filename, index=False)
            print(f"\n  [SAVED] Results saved to: {filename}")

    else:
        print(f"\n  No trade alerts found for {target_date}")
        print("  Possible reasons:")
        print("  - Market was closed (holiday/weekend)")
        print("  - No signals generated based on strategy criteria")
        print("  - Data not available for the date")


if __name__ == "__main__":
    main()
