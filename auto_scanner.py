"""
Auto Scanner — runs every 15 min during market hours (9:15-15:30 IST).
Scans all FNO tickers, records signals, sends NEW active trades to Telegram.

Usage:
    python3 auto_scanner.py

Environment (set in GitHub Actions secrets):
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
"""

import os
import sys
import json
from datetime import datetime, timezone, timedelta

import pandas as pd

from scanner_base import fetch_ohlcv, ticker_label, INDEX_TICKERS, get_index_type
from strategies.supply_demand_v2 import Config, run_signals
from trade_journal import load_journal, record_scan_signals, check_open_trades
from telegram_notifier import send_message, format_trade_alert, already_sent, mark_sent

IST = timezone(timedelta(hours=5, minutes=30))

TICKERS_FILE = os.path.join(os.path.dirname(__file__), "fno_tickers.csv")


def load_tickers() -> list:
    tickers = list(INDEX_TICKERS)
    if os.path.exists(TICKERS_FILE):
        df = pd.read_csv(TICKERS_FILE)
        if "ticker" in df.columns:
            tickers.extend(df["ticker"].dropna().tolist())
    return tickers


def is_market_hours() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    t = now.hour * 60 + now.minute
    return 9 * 60 + 15 <= t <= 15 * 60 + 30


def main():
    if not is_market_hours():
        now = datetime.now(IST)
        print(f"[Auto] Outside market hours ({now.strftime('%H:%M')}). Skipping.")
        return

    print(f"[Auto] Starting scan at {datetime.now(IST).strftime('%H:%M:%S')} IST")

    tickers = load_tickers()
    print(f"[Auto] Tickers to scan: {len(tickers)}")

    all_signals = []
    for ticker in tickers:
        df_5m = fetch_ohlcv(ticker, interval="5m", period="14d")
        if df_5m is None or len(df_5m) < 50:
            continue

        df_daily = fetch_ohlcv(ticker, interval="1d", period="1y")
        df_weekly = fetch_ohlcv(ticker, interval="1wk", period="5y")
        df_h1 = fetch_ohlcv(ticker, interval="1h", period="21d")

        cfg = Config.from_json()
        if ticker in INDEX_TICKERS:
            cfg.use_vol_filt = False
        cfg.index_type = get_index_type(ticker)

        try:
            _, signals = run_signals(df_5m, df_daily, cfg, df_weekly, df_h1=df_h1)
        except Exception as e:
            print(f"[Auto] Error scanning {ticker}: {e}")
            continue

        if signals is not None and not signals.empty:
            signals["ticker"] = ticker_label(ticker)
            signals["symbol"] = ticker
            all_signals.append(signals)
            print(f"[Auto] Signal: {ticker_label(ticker)}")

    if not all_signals:
        print("[Auto] No signals found")
        return

    signals_df = pd.concat(all_signals, ignore_index=True)
    print(f"[Auto] Total signals: {len(signals_df)}")

    added = record_scan_signals(signals_df)
    print(f"[Auto] New records added to journal: {added}")

    check_open_trades()

    journal = load_journal()
    open_trades = [t for t in journal if t.get("status") == "open"]
    print(f"[Auto] Open trades: {len(open_trades)}")

    sent_count = 0
    for trade in open_trades:
        if already_sent(trade):
            continue
        msg = format_trade_alert(trade)
        ok = send_message(msg)
        if ok:
            mark_sent(trade)
            sent_count += 1
            print(f"[Auto] Sent to Telegram: {trade.get('ticker')} {trade.get('trade_type')}")
        else:
            print(f"[Auto] Failed to send: {trade.get('ticker')}")

    print(f"[Auto] Done. Sent {sent_count} new trade(s) to Telegram.")


if __name__ == "__main__":
    main()
