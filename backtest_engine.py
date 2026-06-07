"""
3-Step A+ Supply & Demand Strategy — BACKTEST ENGINE

For each historical signal, checks if target/SL was hit first in subsequent bars.
Tracks P&L per trade, generates summary statistics.

Features:
  - Per-trade outcome: target hit, SL hit, or EOD forced exit
  - P&L tracking with risk-based position sizing
  - Summary: win rate, avg profit, max drawdown, profit factor, avg hold time
  - Multi-ticker support
  - Respects max positions per day and max daily loss from config.json

Usage:
    python3 backtest_engine.py --ticker RELIANCE.NS
    python3 backtest_engine.py --ticker RELIANCE.NS --timeframes 15m
    python3 backtest_engine.py --all --timeframes 15m
    python3 backtest_engine.py --ticker ^NSEI --date 2026-04-07
    python3 backtest_engine.py --all --save
"""

import argparse
import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from strategies.supply_demand_v2 import Config, run_signals
from scanner_base import (
    TICKERS_FILE,
    INDICES_FILE,
    now_ist,
    get_last_trading_date,
    load_tickers,
    load_indices,
    fetch_ohlcv,
    ticker_label,
    progress_print,
    clear_progress,
    make_config,
    INDEX_TICKERS,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

MIN_ROWS = 50
IST = now_ist().tzinfo if hasattr(now_ist(), 'tzinfo') else None

ALL_TIMEFRAMES = ["3m", "5m", "15m", "2h"]

BASE_INTERVALS = {
    "1m": {"yf_period": "7d", "children": {"3m": "3min"}},
    "5m": {"yf_period": "60d", "children": {"5m": None}},
    "15m": {"yf_period": "60d", "children": {"15m": None}},
    "1h": {"yf_period": "730d", "children": {"2h": "2h"}},
}

TF_TO_BASE = {}
for _base, _cfg in BASE_INTERVALS.items():
    for _child_tf in _cfg["children"]:
        TF_TO_BASE[_child_tf] = _base

EOD_EXIT_MINUTES = 15 * 60 + 15


def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    return df.resample(target_tf).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["open", "high", "low", "close"])


def download_base(ticker: str, yf_interval: str, yf_period: str, timeout: int = 60):
    return fetch_ohlcv(ticker, interval=yf_interval, period=yf_period, min_rows=MIN_ROWS)


def fetch_all_timeframes(ticker: str, timeframes: list) -> dict:
    needed_bases = {}
    for tf in timeframes:
        base = TF_TO_BASE[tf]
        needed_bases.setdefault(base, []).append(tf)

    base_data = {}
    for base_interval, child_tfs in needed_bases.items():
        base_cfg = BASE_INTERVALS[base_interval]
        base_data[base_interval] = download_base(ticker, base_interval, base_cfg["yf_period"])

    results = {}
    for tf in timeframes:
        base = TF_TO_BASE[tf]
        base_df = base_data.get(base)
        if base_df is None or base_df.empty:
            results[tf] = None
            continue
        resample_rule = BASE_INTERVALS[base]["children"][tf]
        df = base_df.copy() if resample_rule is None else resample_ohlcv(base_df, resample_rule)
        results[tf] = df if len(df) >= MIN_ROWS else None

    return results


def resolve_signal_outcome(signal_row: dict, ohlcv_df: pd.DataFrame, cfg: Config) -> dict | None:
    """
    Given a signal and the full OHLCV DataFrame, walk forward bar-by-bar
    starting from the bar AFTER the signal bar. Determine what happens first:
      - Target hit
      - SL hit (stop loss)
      - EOD forced exit at 15:15 IST
      - Max bars exhausted
    """
    sig_dt = signal_row["datetime"]
    is_ce = "CE" in signal_row["type"]
    entry = signal_row["entry"]
    sl = signal_row["sl"]
    target = signal_row["target"]
    risk = abs(entry - sl)

    if risk <= 0:
        return None

    try:
        sig_idx = ohlcv_df.index.get_loc(sig_dt)
    except KeyError:
        candidates = ohlcv_df.index[ohlcv_df.index >= sig_dt]
        if len(candidates) == 0:
            return None
        sig_idx = ohlcv_df.index.get_loc(candidates[0])

    max_bars_ahead = 100

    for offset in range(1, min(max_bars_ahead + 1, len(ohlcv_df) - sig_idx)):
        bar_idx = sig_idx + offset
        bar = ohlcv_df.iloc[bar_idx]
        bar_dt = ohlcv_df.index[bar_idx]

        bar_ist = bar_dt.tz_convert("Asia/Kolkata") if bar_dt.tzinfo else bar_dt
        bar_minutes = bar_ist.hour * 60 + bar_ist.minute

        bar_high = bar["high"]
        bar_low = bar["low"]
        bar_close = bar["close"]

        target_hit = False
        sl_hit = False

        if is_ce:
            if bar_low <= sl:
                sl_hit = True
            if bar_high >= target:
                target_hit = True
        else:
            if bar_high >= sl:
                sl_hit = True
            if bar_low <= target:
                target_hit = True

        if sl_hit and not target_hit:
            exit_price = sl
            pnl_points = (exit_price - entry) if is_ce else (entry - exit_price)
            outcome = "SL"
            break

        if sl_hit and target_hit:
            bar_open = bar.get("open", bar_close)
            if is_ce:
                exit_price = sl if (abs(bar_open - sl) < abs(bar_open - target)) else target
            else:
                exit_price = sl if (abs(bar_open - sl) < abs(bar_open - target)) else target
            pnl_points = (exit_price - entry) if is_ce else (entry - exit_price)
            outcome = "SL" if exit_price == sl else "TARGET"
            break

        if target_hit:
            exit_price = target
            pnl_points = (exit_price - entry) if is_ce else (entry - exit_price)
            outcome = "TARGET"
            break

        if bar_minutes >= EOD_EXIT_MINUTES:
            exit_price = bar_close
            pnl_points = (exit_price - entry) if is_ce else (entry - exit_price)
            outcome = "EOD"
            break
    else:
        bar_idx = min(sig_idx + max_bars_ahead, len(ohlcv_df) - 1)
        bar_dt = ohlcv_df.index[bar_idx]
        exit_price = ohlcv_df.iloc[bar_idx]["close"]
        pnl_points = (exit_price - entry) if is_ce else (entry - exit_price)
        outcome = "TIMEOUT"

    pnl_pct = (pnl_points / entry) * 100
    rr_achieved = pnl_points / risk if risk > 0 else 0
    hold_bars = bar_idx - sig_idx

    capital_per_trade = cfg.trading_capital * (cfg.risk_per_trade_percent / 100)
    lot_multiplier = capital_per_trade / risk if risk > 0 else 1
    pnl_rupees = pnl_points * lot_multiplier

    return {
        "ticker": signal_row.get("ticker", ""),
        "timeframe": signal_row.get("timeframe", ""),
        "type": signal_row["type"],
        "entry": entry,
        "sl": sl,
        "target": target,
        "risk": round(risk, 2),
        "rr": signal_row["rr"],
        "signal_dt": sig_dt,
        "exit_dt": bar_dt if 'bar_dt' in dir() else ohlcv_df.index[min(sig_idx + 1, len(ohlcv_df) - 1)],
        "outcome": outcome,
        "exit_price": round(exit_price, 2),
        "pnl_points": round(pnl_points, 2),
        "pnl_pct": round(pnl_pct, 2),
        "pnl_rupees": round(pnl_rupees, 2),
        "rr_achieved": round(rr_achieved, 2),
        "hold_bars": hold_bars,
    }


def backtest_ticker(ticker: str, timeframes: list, cfg: Config = None) -> list:
    """
    Download data, generate signals, resolve outcomes for one ticker.
    Returns list of trade result dicts.
    """
    if cfg is None:
        cfg = make_config(ticker)

    tf_data = fetch_all_timeframes(ticker, timeframes)
    all_results = []

    for tf in timeframes:
        df = tf_data.get(tf)
        if df is None:
            continue

        tcfg = make_config(ticker)
        df_daily = tf_data.get("1d") if "1d" in tf_data else fetch_ohlcv(ticker, interval="1d", period="1y", min_rows=50)
        try:
            result_df, signals = run_signals(df, df_daily, tcfg)
        except Exception:
            continue

        if signals.empty:
            continue

        for _, sig_row in signals.iterrows():
            sig_dict = sig_row.to_dict()
            sig_dict["ticker"] = ticker_label(ticker)
            sig_dict["timeframe"] = tf

            outcome = resolve_signal_outcome(sig_dict, df, tcfg)
            if outcome is not None:
                all_results.append(outcome)

    return all_results


def apply_daily_limits(results: list, cfg: Config) -> list:
    """
    Apply max positions per day and max daily loss limits.
    Returns filtered results.
    """
    if not results:
        return results

    df = pd.DataFrame(results)
    df["date"] = df["signal_dt"].apply(
        lambda x: x.date() if hasattr(x, 'date') else x.date()
    )

    filtered = []
    daily_pnl = {}

    for _, row in df.iterrows():
        date = row["date"]
        daily_pnl.setdefault(date, 0.0)

        max_daily_loss = cfg.trading_capital * (cfg.max_loss_per_day_pct / 100)
        if daily_pnl[date] <= -max_daily_loss:
            continue

        day_trades = [r for r in filtered if r.get("_date") == date]
        if len(day_trades) >= cfg.max_positions_per_day:
            continue

        row_dict = row.to_dict()
        row_dict["_date"] = date
        filtered.append(row_dict)
        daily_pnl[date] += row_dict["pnl_rupees"]

    for r in filtered:
        r.pop("_date", None)

    return filtered


def compute_stats(results: list, cfg: Config) -> dict:
    """
    Compute summary statistics from a list of trade result dicts.
    """
    if not results:
        return {
            "total_trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "avg_pnl": 0.0, "total_pnl": 0.0,
            "max_drawdown": 0.0, "profit_factor": 0.0,
            "avg_hold_bars": 0.0, "target_rate": 0.0,
            "sl_rate": 0.0, "avg_rr_achieved": 0.0,
            "max_win": 0.0, "max_loss": 0.0,
            "consecutive_wins": 0, "consecutive_losses": 0,
        }

    df = pd.DataFrame(results)

    total_trades = len(df)
    wins = len(df[df["pnl_points"] > 0])
    losses = len(df[df["pnl_points"] <= 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    target_trades = len(df[df["outcome"] == "TARGET"])
    sl_trades = len(df[df["outcome"] == "SL"])

    target_rate = (target_trades / total_trades * 100) if total_trades > 0 else 0.0
    sl_rate = (sl_trades / total_trades * 100) if total_trades > 0 else 0.0

    avg_pnl = df["pnl_rupees"].mean() if "pnl_rupees" in df.columns else 0.0
    total_pnl = df["pnl_rupees"].sum() if "pnl_rupees" in df.columns else 0.0

    cumulative_pnl = df["pnl_rupees"].cumsum() if "pnl_rupees" in df.columns else pd.Series([0])
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0

    gross_profit = df.loc[df["pnl_rupees"] > 0, "pnl_rupees"].sum() if "pnl_rupees" in df.columns else 0.0
    gross_loss = abs(df.loc[df["pnl_rupees"] < 0, "pnl_rupees"].sum()) if "pnl_rupees" in df.columns else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    avg_hold_bars = df["hold_bars"].mean() if "hold_bars" in df.columns else 0.0
    avg_rr_achieved = df["rr_achieved"].mean() if "rr_achieved" in df.columns else 0.0

    max_win = df["pnl_rupees"].max() if "pnl_rupees" in df.columns and len(df) > 0 else 0.0
    max_loss = df["pnl_rupees"].min() if "pnl_rupees" in df.columns and len(df) > 0 else 0.0

    pnl_sign = (df["pnl_points"] > 0).values if "pnl_points" in df.columns else np.array([])
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0
    for s in pnl_sign:
        if s:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 2),
        "target_rate": round(target_rate, 2),
        "sl_rate": round(sl_rate, 2),
        "avg_pnl": round(avg_pnl, 2),
        "total_pnl": round(total_pnl, 2),
        "max_drawdown": round(max_drawdown, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "INF",
        "avg_hold_bars": round(avg_hold_bars, 1),
        "avg_rr_achieved": round(avg_rr_achieved, 2),
        "max_win": round(max_win, 2),
        "max_loss": round(max_loss, 2),
        "consecutive_wins": max_consec_wins,
        "consecutive_losses": max_consec_losses,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
    }


def print_stats(stats: dict, label: str = ""):
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  BACKTEST RESULTS  {label}")
    print(f"{sep}")
    print(f"  Total Trades      : {stats['total_trades']}")
    print(f"  Wins / Losses     : {stats['wins']} / {stats['losses']}")
    print(f"  Win Rate          : {stats['win_rate']}%")
    print(f"  Target Hit Rate   : {stats['target_rate']}%")
    print(f"  SL Hit Rate       : {stats['sl_rate']}%")
    print(f"  ─────────────────────────────────")
    print(f"  Avg P&L / Trade   : Rs {stats['avg_pnl']}")
    print(f"  Total P&L         : Rs {stats['total_pnl']}")
    print(f"  Max Win           : Rs {stats['max_win']}")
    print(f"  Max Loss          : Rs {stats['max_loss']}")
    print(f"  Max Drawdown      : Rs {stats['max_drawdown']}")
    print(f"  Gross Profit      : Rs {stats['gross_profit']}")
    print(f"  Gross Loss        : Rs {stats['gross_loss']}")
    print(f"  Profit Factor     : {stats['profit_factor']}")
    print(f"  ─────────────────────────────────")
    print(f"  Avg RR Achieved   : {stats['avg_rr_achieved']}")
    print(f"  Avg Hold (bars)   : {stats['avg_hold_bars']}")
    print(f"  Max Consec Wins   : {stats['consecutive_wins']}")
    print(f"  Max Consec Losses : {stats['consecutive_losses']}")
    print(f"{sep}")


def print_trade_log(results: list, limit: int = 50):
    if not results:
        print("  No trades to display.")
        return

    df = pd.DataFrame(results)
    display_cols = ["ticker", "timeframe", "type", "entry", "sl", "outcome",
                    "exit_price", "pnl_points", "pnl_rupees", "rr_achieved",
                    "hold_bars", "signal_dt"]
    available = [c for c in display_cols if c in df.columns]
    display_df = df[available].head(limit)

    print(f"\n  TRADE LOG (showing {min(limit, len(df))} of {len(df)} trades)")
    print(f"  {'─' * 58}")
    print(display_df.to_string(index=False))


def print_outcome_breakdown(results: list):
    if not results:
        return

    df = pd.DataFrame(results)
    print(f"\n  OUTCOME BREAKDOWN:")
    print(f"  {'─' * 40}")

    outcome_counts = df["outcome"].value_counts()
    total = len(df)
    for outcome, count in outcome_counts.items():
        avg_pnl = df.loc[df["outcome"] == outcome, "pnl_rupees"].mean()
        pct = count / total * 100
        print(f"    {outcome:<10} : {count:>4} trades ({pct:>5.1f}%)  avg P&L: Rs {avg_pnl:>8.2f}")

    if "timeframe" in df.columns:
        print(f"\n  BY TIMEFRAME:")
        print(f"  {'─' * 40}")
        for tf in df["timeframe"].unique():
            tf_df = df[df["timeframe"] == tf]
            wr = len(tf_df[tf_df["pnl_points"] > 0]) / len(tf_df) * 100 if len(tf_df) > 0 else 0
            tpnl = tf_df["pnl_rupees"].sum()
            print(f"    {tf:<6} : {len(tf_df):>3} trades  WR: {wr:>5.1f}%  P&L: Rs {tpnl:>8.2f}")


def backtest_single(ticker: str, timeframes: list, cfg: Config = None) -> list:
    label = ticker_label(ticker)
    print(f"\n  {'#'*62}")
    print(f"  S&D BACKTEST — {label}")
    print(f"  Ticker     : {ticker}")
    print(f"  Timeframes : {', '.join(timeframes)}")
    print(f"  {'#'*62}")

    t0 = time.time()
    results = backtest_ticker(ticker, timeframes, cfg)
    elapsed = time.time() - t0

    print(f"\n  Completed in {elapsed:.1f}s — {len(results)} trades resolved")

    if results:
        results = apply_daily_limits(results, cfg or make_config(ticker))
        stats = compute_stats(results, cfg or make_config(ticker))
        print_stats(stats, f"— {label}")
        print_outcome_breakdown(results)
        print_trade_log(results)

    return results


def backtest_all(timeframes: list, cfg: Config = None) -> list:
    fno_tickers = load_tickers(TICKERS_FILE)
    indices = load_indices(INDICES_FILE)
    all_tickers = indices + fno_tickers
    total = len(all_tickers)

    print(f"\n  {'#'*62}")
    print(f"  BACKTEST — ALL TICKERS")
    print(f"  Tickers    : {total}")
    print(f"  Timeframes : {', '.join(timeframes)}")
    print(f"  {'#'*62}\n")

    t0 = time.time()
    all_results = []
    completed = 0
    errors = 0

    for ticker in all_tickers:
        completed += 1
        label = ticker_label(ticker)
        elapsed = time.time() - t0

        try:
            results = backtest_ticker(ticker, timeframes)
            sig_count = len(results)
            status = f"{sig_count} trades"
        except Exception as exc:
            errors += 1
            status = f"ERROR: {exc}"
            results = []

        pct = completed / total * 100
        print(f"  [{completed:>3}/{total}] ({pct:>5.1f}%) {label:<15} {status:>12}  [{elapsed:>4.0f}s]", flush=True)

        all_results.extend(results)

    elapsed = time.time() - t0

    print(f"\n  {'='*62}")
    print(f"  SCAN COMPLETE — {elapsed:.1f}s")
    print(f"  Tickers scanned : {total}")
    print(f"  Total trades    : {len(all_results)}")
    print(f"  Errors          : {errors}")
    print(f"  {'='*62}")

    if all_results:
        default_cfg = Config.from_json()
        all_results = apply_daily_limits(all_results, default_cfg)
        stats = compute_stats(all_results, default_cfg)
        print_stats(stats, "— ALL TICKERS")
        print_outcome_breakdown(all_results)
        print_trade_log(all_results, limit=30)

    return all_results


def save_results(results: list, filename: str):
    if not results:
        print("  No results to save.")
        return

    df = pd.DataFrame(results)
    cols_order = [
        "ticker", "timeframe", "type", "entry", "sl", "target",
        "risk", "rr", "outcome", "exit_price", "pnl_points",
        "pnl_pct", "pnl_rupees", "rr_achieved", "hold_bars",
        "signal_dt", "exit_dt",
    ]
    available = [c for c in cols_order if c in df.columns]
    df[available].to_csv(filename, index=False)
    print(f"\n  [SAVED] Results saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="TWP Algo v6 — Backtest Engine"
    )
    parser.add_argument("--ticker", type=str, default=None,
                        help="Single ticker to backtest (e.g., RELIANCE.NS, ^NSEI)")
    parser.add_argument("--all", action="store_true", dest="scan_all",
                        help="Backtest ALL FNO stocks + Indices")
    parser.add_argument("--timeframes", nargs="+", default=None,
                        help=f"Timeframes (default: all). Choices: {', '.join(ALL_TIMEFRAMES)}")
    parser.add_argument("--save", action="store_true",
                        help="Save results to CSV")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV filename (default: auto-generated)")
    args = parser.parse_args()

    if not args.ticker and not args.scan_all:
        parser.error("--ticker TICKER.NS or --all required")

    timeframes = args.timeframes if args.timeframes else ALL_TIMEFRAMES
    for tf in timeframes:
        if tf not in ALL_TIMEFRAMES:
            print(f"  [ERROR] Invalid timeframe: {tf}")
            print(f"  Valid: {', '.join(ALL_TIMEFRAMES)}")
            sys.exit(1)

    cfg = Config.from_json()

    if args.ticker:
        results = backtest_single(args.ticker, timeframes, cfg)
    else:
        results = backtest_all(timeframes, cfg)

    if args.save and results:
        if args.output:
            filename = args.output
        else:
            ticker_part = ticker_label(args.ticker) if args.ticker else "ALL"
            filename = f"backtest_{ticker_part}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        save_results(results, filename)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Backtest stopped.\n")
