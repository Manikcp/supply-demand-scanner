"""
3-Step A+ Supply & Demand Strategy — LIVE FNO + Indices Scanner

Auto-runs during market hours (9:15 AM – 3:30 PM IST).
Scans all FNO stocks + Indices every 15 min and alerts on new signals.
Duplicate alerts are prevented — already seen signals are skipped.

Usage:
    python3 run_twp_scanner.py
    python3 run_twp_scanner.py --interval 15
"""

import argparse
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

from strategies.signals import Config, run_signals
from scanner_base import (
    TICKERS_FILE,
    INDEX_TICKERS,
    now_ist,
    is_market_open,
    load_tickers,
    fetch_ohlcv,
    ticker_label,
    signal_key,
    format_alert_with_targets,
    progress_print,
    clear_progress,
    make_config,
)


def scan_once(tickers: list, cfg: Config, seen: set, scan_start: datetime) -> list:
    """
    Scan all tickers and return new signals not in seen set.
    seen set is updated in-place.
    """
    today = scan_start.date()
    new_alerts = []
    total = len(tickers)

    for pos, ticker in enumerate(tickers, 1):
        progress_print(pos, total, ticker)

        df = fetch_ohlcv(ticker, interval="5m", period="7d")
        if df is None:
            continue

        df_daily = fetch_ohlcv(ticker, interval="1d", period="1y")
        df_weekly = fetch_ohlcv(ticker, interval="1wk", period="5y")
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
        today_sigs = signals[signals["_date"] == today].drop(columns=["_date"])

        for _, row in today_sigs.iterrows():
            key = signal_key(ticker, row["type"], row["datetime"])
            if key in seen:
                continue
            seen.add(key)

            rec = row.to_dict()
            rec["ticker"] = ticker_label(ticker)
            new_alerts.append(rec)
            print(format_alert_with_targets(ticker, rec))

    clear_progress()
    return new_alerts


def run_live(scan_interval_min: int = 15):
    tickers = load_tickers(TICKERS_FILE)
    cfg = Config.from_json()

    seen = set()
    all_day_alerts = []

    print(f"\n{'#'*62}")
    print(f"  3-Step A+ S&D Strategy — LIVE FNO + Indices Scanner")
    print(f"  Total tickers  : {len(tickers)}")
    print(f"  Scan interval  : {scan_interval_min} min")
    print(f"  Market hours   : 10:00 – 15:00 IST (active session)")
    print(f"  Min score      : {cfg.min_score_threshold} | Trend req: {cfg.require_trend_factor}")
    print(f"  Zone ATR width : {cfg.zone_atr_width} | SL buffer: {cfg.sl_atr_buffer} ATR")
    print(f"  Target RR      : 1:{cfg.target_rr}")
    print(f"  Started at     : {now_ist().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print(f"{'#'*62}\n")

    while True:
        now = now_ist()

        if not is_market_open():
            print(f"  Market closed  [{now.strftime('%H:%M:%S IST')}]  — "
                  f"next check in 1 min ...", end="\r")
            time.sleep(60)
            continue

        print(f"\n  -- Scan started : {now.strftime('%H:%M:%S IST')} --")
        new = scan_once(tickers, cfg, seen, scan_start=now)

        if new:
            all_day_alerts.extend(new)
            print(f"\n  [OK] {len(new)} new signal(s) this scan  "
                  f"| Total today: {len(all_day_alerts)}")
        else:
            print(f"  -- No new signals this scan  "
                  f"| Total today: {len(all_day_alerts)}")

        if all_day_alerts:
            print(f"\n  == TODAY'S ALERTS SO FAR ==")
            df_summary = pd.DataFrame(all_day_alerts)
            df_summary["risk"] = abs(df_summary["entry"] - df_summary["sl"]).round(2)
            cols = ["ticker", "type", "strike", "entry", "target", "sl", "risk", "rr", "entry_grade", "datetime"]
            available_cols = [c for c in cols if c in df_summary.columns]
            print(df_summary[available_cols].to_string(index=False))

        target = now_ist() + timedelta(minutes=scan_interval_min)
        print(f"\n  Next scan at: {target.strftime('%H:%M:%S IST')}  (Ctrl+C to stop)")

        while now_ist() < target:
            remaining = int((target - now_ist()).total_seconds())
            sys.stdout.write(f"\r  Waiting ... {remaining:>4}s remaining")
            sys.stdout.flush()
            time.sleep(1)

        sys.stdout.write("\r" + " " * 40 + "\r")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TWP Algo v6 -- Live FNO + Indices Scanner")
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Scan interval in minutes (default: 15)",
    )
    args = parser.parse_args()

    try:
        run_live(scan_interval_min=args.interval)
    except KeyboardInterrupt:
        print("\n\n  Scanner stopped. Good bye!\n")
