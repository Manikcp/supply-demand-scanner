"""
Scanner Base Module

Shared code for CLI scanners:
- Ticker loading
- OHLCV data fetching with caching
- Signal formatting
- Market hours utilities
"""

import contextlib
import io
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Set

import pandas as pd

from strategies.signals import Config

IST = timezone(timedelta(hours=5, minutes=30))

INDEX_TICKERS: Set[str] = {
    "^NSEI", "^NSEBANK", "^BSESN", "^NSENXT50",
    "NIFTY_FIN_SERVICE.NS", "NIFTY_MID_SELECT.NS", "BANKEX.BO",
    "GIFTNIFTY.NS", "^INDIAVIX",
}

TICKER_LABELS: Dict[str, str] = {
    "^NSEI": "NIFTY50",
    "^NSEBANK": "BANKNIFTY",
    "^BSESN": "SENSEX",
    "NIFTY_FIN_SERVICE.NS": "FINNIFTY",
    "NIFTY_MID_SELECT.NS": "MIDCAPSELECT",
    "BANKEX.BO": "BANKEX",
    "^NSENXT50": "NIFTYJR",
    "GIFTNIFTY.NS": "GIFTNIFTY",
    "^INDIAVIX": "INDIAVIX",
}

TICKERS_FILE = "fno_tickers.csv"
INDICES_FILE = "indices_tikcers.csv"

MARKET_OPEN_H, MARKET_OPEN_M = 9, 15
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 30


def now_ist() -> datetime:
    return datetime.now(IST)


def is_market_open() -> bool:
    t = now_ist()
    if t.weekday() >= 5:
        return False
    mins = t.hour * 60 + t.minute
    open_mins = MARKET_OPEN_H * 60 + MARKET_OPEN_M
    close_mins = MARKET_CLOSE_H * 60 + MARKET_CLOSE_M
    return open_mins <= mins <= close_mins


def get_last_trading_date(target_date: Optional[datetime.date] = None) -> datetime.date:
    if target_date is None:
        target_date = now_ist().date()
    weekday = target_date.weekday()
    if weekday == 5:
        return target_date - timedelta(days=1)
    elif weekday == 6:
        return target_date - timedelta(days=2)
    return target_date


def load_tickers(path: str) -> List[str]:
    df = pd.read_csv(path)
    col = df.columns[0]
    tickers = df[col].dropna().str.strip().tolist()
    tickers = [t for t in tickers if t.lower() != "ticker"]
    return tickers


def load_indices(path: str) -> List[str]:
    try:
        df = pd.read_csv(path)
        return df["ticker"].dropna().str.strip().tolist()
    except Exception:
        return []


def ticker_label(ticker: str) -> str:
    return TICKER_LABELS.get(ticker, ticker.split(".")[0])


def get_index_type(ticker: str) -> str:
    label = ticker_label(ticker).upper()
    if "BANK" in label or ticker in {"^NSEBANK", "BANKEX.BO"}:
        return "BANKNIFTY"
    elif ticker == "^BSESN":
        return "SENSEX"
    return "NIFTY"


def make_config(ticker: str, base_cfg: Optional[Config] = None) -> Config:
    if base_cfg is None:
        cfg = Config.from_json()
    else:
        cfg = Config(
            index_type=base_cfg.index_type,
            use_vol_filt=base_cfg.use_vol_filt,
            lookback_bars=base_cfg.lookback_bars,
            min_impulse_candles=base_cfg.min_impulse_candles,
            max_consolidation_candles=base_cfg.max_consolidation_candles,
            impulse_body_mult=base_cfg.impulse_body_mult,
            zone_atr_width=base_cfg.zone_atr_width,
            fvg_min_gap_atr=base_cfg.fvg_min_gap_atr,
            slow_momentum_bars=base_cfg.slow_momentum_bars,
            slow_momentum_max_body_pct=base_cfg.slow_momentum_max_body_pct,
            min_close_in_zone_pct=base_cfg.min_close_in_zone_pct,
            allow_wick_entry=base_cfg.allow_wick_entry,
            trend_swing_lookback=base_cfg.trend_swing_lookback,
            bos_confirm_bars=base_cfg.bos_confirm_bars,
            ema_len=base_cfg.ema_len,
            volume_avg_period=base_cfg.volume_avg_period,
            volume_spike_mult=base_cfg.volume_spike_mult,
            atr_len=base_cfg.atr_len,
            min_rr_ratio=base_cfg.min_rr_ratio,
            target_rr=base_cfg.target_rr,
            sl_atr_buffer=base_cfg.sl_atr_buffer,
            trading_capital=base_cfg.trading_capital,
            risk_per_trade_percent=base_cfg.risk_per_trade_percent,
            max_positions_per_day=base_cfg.max_positions_per_day,
            max_loss_per_day_pct=base_cfg.max_loss_per_day_pct,
            min_score_threshold=base_cfg.min_score_threshold,
            require_trend_factor=base_cfg.require_trend_factor,
            block_opening_session=base_cfg.block_opening_session,
            dedup_bar_cooldown=base_cfg.dedup_bar_cooldown,
            use_h1_zones=base_cfg.use_h1_zones,
            body_based_zones=base_cfg.body_based_zones,
            next_bar_entry=base_cfg.next_bar_entry,
        )

    if ticker in INDEX_TICKERS:
        cfg.use_vol_filt = False

    label = ticker_label(ticker).upper()
    if "BANK" in label or ticker in {"^NSEBANK", "BANKEX.BO"}:
        cfg.index_type = "BANKNIFTY"
    elif ticker == "^BSESN":
        cfg.index_type = "SENSEX"
    else:
        cfg.index_type = "NIFTY"

    return cfg


def fetch_ohlcv(
    ticker: str,
    interval: str = "15m",
    period: str = "7d",
    min_rows: int = 24,
    use_cache: bool = True,
    cache_dir: str = "data_cache"
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data from yfinance with optional caching.

    Args:
        ticker: Stock ticker symbol
        interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk)
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y)
        min_rows: Minimum rows required
        use_cache: Whether to use caching
        cache_dir: Cache directory path

    Returns:
        DataFrame with OHLCV data or None if not available
    """
    import yfinance as yf

    cache_key = f"{ticker}_{interval}_{period}"
    cache_file = Path(cache_dir) / f"{cache_key}.parquet"

    if use_cache and cache_file.exists():
        cache_age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds()
        max_cache_age = 300 if interval in ["1m", "3m", "5m", "15m"] else 3600

        if cache_age < max_cache_age:
            try:
                df = pd.read_parquet(cache_file)
                if len(df) >= min_rows:
                    return df
            except Exception:
                pass

    max_retries = 3
    raw = None
    for attempt in range(max_retries):
        try:
            _sink = io.StringIO()
            with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
                raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if raw is not None and not raw.empty:
                break
        except Exception:
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.columns = [c.lower() for c in raw.columns]
    needed = {"open", "high", "low", "close", "volume"}
    if not needed.issubset(raw.columns):
        return None

    raw = raw[["open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])
    raw["volume"] = raw["volume"].fillna(0)

    if len(raw) < min_rows:
        return None

    if raw.index.tzinfo is None:
        raw.index = raw.index.tz_localize("UTC")
    raw.index = raw.index.tz_convert("Asia/Kolkata")

    MARKET_OPEN = 9 * 60 + 15
    MARKET_CLOSE = 15 * 60 + 30

    if interval == "15m" or interval.endswith("m"):
        raw = raw[
            (raw.index.hour * 60 + raw.index.minute >= MARKET_OPEN) &
            (raw.index.hour * 60 + raw.index.minute <= MARKET_CLOSE)
        ]
        raw = raw[raw.index.dayofweek < 5]

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            raw.to_parquet(cache_file)
        except Exception:
            pass

    return raw


def signal_key(ticker: str, sig_type: str, bar_time) -> str:
    return f"{ticker}|{sig_type}|{bar_time}"


def format_alert(ticker: str, row: dict) -> str:
    arrow = "▲" if "CE" in row["type"] else "▼"
    sep = "=" * 62
    risk = round(abs(float(row["entry"]) - float(row["sl"])), 2)
    strong_flag = " ★" if row.get("strong_signal") else ""
    weak_flag = " ⚠WEAK" if row.get("weak_exit") else ""
    return (
        f"\n{sep}\n"
        f"  {arrow}  {row['type']}{strong_flag}{weak_flag}  —  {ticker_label(ticker)}\n"
        f"{sep}\n"
        f"  Strike  : {row.get('index', 'NIFTY')}{row['strike']}\n"
        f"  Entry   : {row['entry']}\n"
        f"  Target  : {row['target']}\n"
        f"  SL      : {row['sl']}\n"
        f"  Risk    : ₹{risk}\n"
        f"  RR      : 1:{row['rr']}\n"
        f"  Grade   : {row.get('entry_grade', 'N/A')} ({row.get('entry_quality', 0)}%)\n"
        f"  ST      : {row.get('supertrend', 'N/A')} | RSI: {row.get('rsi_zone', 'N/A')}\n"
        f"  Time    : {row['datetime']}\n"
        f"{sep}"
    )


def format_alert_with_targets(ticker: str, row: dict) -> str:
    arrow = "🟢" if "CE" in row["type"] else "🔴"
    sep = "=" * 60
    risk = round(abs(float(row["entry"]) - float(row["sl"])), 2)
    strong_flag = " ★" if row.get("strong_signal") else ""
    fvg_flag = " ★ FVG" if row.get("fvg_present") else ""
    trend_icon = "🟢" if row.get("market_structure") == "bullish" else "🔴"
    lines = [
        f"\n{sep}",
        f"  {arrow} {row['type']}{fvg_flag}{strong_flag} — {ticker_label(ticker)} @ ₹{row['entry']} | Grade {row.get('entry_grade', 'N/A')} | ⏳",
        f"{sep}",
        f"  Pattern: {row.get('pattern', 'N/A')}",
        f"",
        f"  Entry   : ₹{row['entry']}",
        f"  Target  : ₹{row['target']}",
        f"  SL      : ₹{row['sl']}",
    ]
    if "support" in row:
        lines.append(f"  Support : ₹{row['support']}")
    if "resistance" in row:
        lines.append(f"  Resis   : ₹{row['resistance']}")
    lines += [
        f"  RR      : 1:{row['rr']}",
        f"",
        f"  Trend   : {trend_icon} {row.get('market_structure', 'N/A')} | Entry: {row.get('zone_entry_reason', 'N/A')}",
        f"  Strong  : {'Yes ★' if row.get('strong_signal') else 'No'} | Slow Mom: {'Yes' if row.get('slow_momentum') else 'No'} | FVG: {'Yes' if row.get('fvg_present') else 'No'}",
        f"  EMA     : {'Aligned' if row.get('ema_aligned') else 'Not Aligned'}",
        f"",
        f"  Position Size:",
        f"  Qty: {row.get('qty', '—')} shares | Risk: ₹{risk}",
        f"",
        f"  Option:",
        f"  Strike: {row.get('index', 'NIFTY')}{row['strike']} ATM",
        f"  Time: {row['datetime']}",
        f"",
        f"  Factors: {row.get('factors', 'N/A')}",
        f"{sep}",
    ]
    return "\n".join(lines)


def progress_print(pos: int, total: int, ticker: str) -> None:
    sys.stdout.write(f"\r  [{pos:>3}/{total}] {ticker:<28}")
    sys.stdout.flush()


def clear_progress() -> None:
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()
