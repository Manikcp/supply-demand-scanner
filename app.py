"""
3-Step A+ Supply & Demand Strategy — Multi-Timeframe Web UI

3-Step Signal Flow:
  1. Identify Institutional Demand/Supply Zones
     - Explosive impulsive moves (3+ big candles)
     - Zone drawn from consolidation BEFORE the impulse
     - Fair Value Gap (FVG) for momentum confirmation
  2. Trade with the Trend (Break of Structure)
     - Swing high/low identification
     - BOS confirmation for trend direction
     - Only demand in uptrend, supply in downtrend
  3. Entry Conditions
     - Slow momentum approaching zone
     - Candle closes IN or wicks INTO zone
     - Confirmation candle triggers entry
     - SL tight to zone, TP at 1:1 to 1.5 R:R

Usage:
    streamlit run app.py
"""

import json
import os
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import contextlib
import io
import time
import warnings
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple

from strategies.supply_demand_v2 import Config, run_signals
from strategies.sr_levels import get_all_sr_levels, detect_market_structure
from trade_journal import record_scan_signals, check_open_trades

warnings.filterwarnings("ignore")

TICKERS_FILE = "fno_tickers.csv"
INDICES_FILE = "indices_tikcers.csv"
DATA_INTERVAL = "5m"
DATA_PERIOD = "14d"
DAILY_PERIOD = "1y"
WEEKLY_PERIOD = "5y"
INTRADAY_LOOKBACK_BARS = 50
MIN_ROWS = 50

IST = timezone(timedelta(hours=5, minutes=30))

INDEX_TICKERS = {
    "^NSEI", "^NSEBANK", "^BSESN", "^NSENXT50",
    "NIFTY_FIN_SERVICE.NS", "NIFTY_MID_SELECT.NS", "BANKEX.BO",
    "GIFTNIFTY.NS", "^INDIAVIX",
}

TICKER_LABELS = {
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

STRIKE_STEPS = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
    "SENSEX": 100,
    "FINNIFTY": 50,
    "MIDCAPSELECT": 50,
    "STOCK": 1,
}

def _refresh_journal_state():
    import os as _os, json as _json
    st.session_state.trade_journal = []
    _jf = _os.path.join(_os.path.dirname(__file__), "trade_journal.json")
    if _os.path.exists(_jf):
        try:
            with open(_jf) as _f:
                data = _json.load(_f)
            for t in data:
                st.session_state.trade_journal.append({
                    "date": t.get("entry_date", ""),
                    "ticker": t.get("ticker", ""),
                    "type": "CE BUY" if t.get("trade_type") == "CE" else "PE BUY",
                    "entry": t.get("entry_price", 0),
                    "target": t.get("target", 0),
                    "sl": t.get("stop_loss", 0),
                    "qty": 0,
                    "notes": t.get("factors", ""),
                    "result": "Target Hit" if t.get("status") == "hit_target" else "SL Hit" if t.get("status") == "hit_sl" else "Manual Exit" if t.get("status") == "expired" else "Pending",
                    "pnl": t.get("pnl_pct"),
                    "pattern": t.get("pattern"),
                    "confidence": t.get("confidence"),
                    "stage": t.get("stage"),
                    "grade": t.get("entry_grade", ""),
                    "score": t.get("score", 0),
                    "rr": t.get("rr", 0),
                    "factors": t.get("factors", ""),
                    "strike": t.get("strike", 0),
                    "entry_time": t.get("entry_time", ""),
                })
        except Exception:
            pass


def cross_ref_outcome(signals_df):
    import os as _os, json as _json
    _jf = _os.path.join(_os.path.dirname(__file__), "trade_journal.json")
    if not _os.path.exists(_jf):
        return
    try:
        with open(_jf) as _f:
            journal = _json.load(_f)
    except Exception:
        return
    status_map = {}
    for t in journal:
        ticker = t.get("ticker", "")
        tt = t.get("trade_type", "")
        status = t.get("status", "")
        if ticker and tt and status in ("hit_target", "hit_sl"):
            status_map[(ticker, tt)] = status
    if "outcome" not in signals_df.columns:
        signals_df["outcome"] = "pending"
    for i, row in signals_df.iterrows():
        ticker = row.get("symbol", row.get("ticker", ""))
        sig_type = row.get("type", "")
        tt = "CE" if "CE" in str(sig_type) else "PE"
        key = (ticker, tt)
        if key in status_map:
            signals_df.at[i, "outcome"] = status_map[key]


if "signals_df" not in st.session_state:
    st.session_state.signals_df = None
if "ohlcv_data" not in st.session_state:
    st.session_state.ohlcv_data = {}
if "h1_data" not in st.session_state:
    st.session_state.h1_data = {}
if "daily_data" not in st.session_state:
    st.session_state.daily_data = {}
if "weekly_data" not in st.session_state:
    st.session_state.weekly_data = {}
if "scan_done" not in st.session_state:
    st.session_state.scan_done = False
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "trade_journal" not in st.session_state:
    st.session_state.trade_journal = []
    _refresh_journal_state()
if "journal_checked" not in st.session_state:
    st.session_state.journal_checked = False
if "selected_trade" not in st.session_state:
    st.session_state.selected_trade = None
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
if "scan_cancelled" not in st.session_state:
    st.session_state.scan_cancelled = False
if "scanning" not in st.session_state:
    st.session_state.scanning = False
if "current_batch" not in st.session_state:
    st.session_state.current_batch = 0
if "batch_signals" not in st.session_state:
    st.session_state.batch_signals = []
if "batch_ohlcv" not in st.session_state:
    st.session_state.batch_ohlcv = {}
if "batch_h1" not in st.session_state:
    st.session_state.batch_h1 = {}
if "batch_daily" not in st.session_state:
    st.session_state.batch_daily = {}
if "batch_weekly" not in st.session_state:
    st.session_state.batch_weekly = {}


def load_config() -> dict:
    try:
        with open("config.json") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(cfg: dict):
    try:
        with open("config.json", "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save config: {e}")


def load_tickers(path: str) -> list:
    df = pd.read_csv(path)
    col = df.columns[0]
    tickers = df[col].dropna().str.strip().tolist()
    tickers = [t for t in tickers if t.lower() != "ticker"]
    return tickers


def load_indices(path: str) -> list:
    try:
        df = pd.read_csv(path)
        if "ticker" in df.columns:
            return df["ticker"].dropna().str.strip().tolist()
        col = df.columns[0]
        return df[col].dropna().str.strip().tolist()
    except Exception:
        return list(INDEX_TICKERS)


def fetch_ohlcv(ticker: str, interval: str = "5m", period: str = "14d") -> Optional[pd.DataFrame]:
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

    if len(raw) < MIN_ROWS:
        return None

    if raw.index.tzinfo is None:
        raw.index = raw.index.tz_localize("UTC")
    raw.index = raw.index.tz_convert("Asia/Kolkata")

    MARKET_OPEN = 9 * 60 + 15
    MARKET_CLOSE = 15 * 60 + 30

    # Filter market hours for intraday intervals
    if interval in ["1m", "2m", "3m", "5m"]:
        raw = raw[
            (raw.index.hour * 60 + raw.index.minute >= MARKET_OPEN) &
            (raw.index.hour * 60 + raw.index.minute <= MARKET_CLOSE)
        ]
        raw = raw[raw.index.dayofweek < 5]

    return raw


def ticker_label(ticker: str) -> str:
    return TICKER_LABELS.get(ticker, ticker.split(".")[0])


def get_index_type(ticker: str) -> str:
    label = ticker_label(ticker).upper()
    if "BANK" in label or ticker in {"^NSEBANK", "BANKEX.BO"}:
        return "BANKNIFTY"
    elif ticker == "^BSESN":
        return "SENSEX"
    elif "FIN" in label:
        return "FINNIFTY"
    elif "MID" in label:
        return "MIDCAPSELECT"
    return "NIFTY"


def get_strike_step(ticker: str) -> float:
    idx_type = get_index_type(ticker)
    if idx_type in STRIKE_STEPS:
        return STRIKE_STEPS[idx_type]
    return 1.0


def calculate_position_size(entry: float, sl: float, capital: float, risk_pct: float, 
                            lot_size: int = 1) -> dict:
    risk_per_trade = capital * (risk_pct / 100)
    risk_per_share = abs(entry - sl)
    if risk_per_share <= 0:
        return {"quantity": 0, "lots": 0, "risk_amount": 0}
    
    quantity = int(risk_per_trade / risk_per_share)
    lots = quantity // lot_size if lot_size > 0 else quantity
    
    return {
        "quantity": quantity,
        "lots": max(lots, 1),
        "risk_amount": round(risk_per_trade, 2),
        "risk_per_share": round(risk_per_share, 2)
    }


def get_option_strike(spot: float, strike_step: float, direction: str, 
                      preference: str = "atm") -> Tuple[float, str]:
    atm = round(spot / strike_step) * strike_step
    
    if preference == "atm":
        return atm, "ATM"
    elif preference == "otm_1":
        if direction == "CE":
            return atm + strike_step, "OTM-1"
        else:
            return atm - strike_step, "OTM-1"
    elif preference == "otm_2":
        if direction == "CE":
            return atm + 2 * strike_step, "OTM-2"
        else:
            return atm - 2 * strike_step, "OTM-2"
    elif preference == "itm_1":
        if direction == "CE":
            return atm - strike_step, "ITM-1"
        else:
            return atm + strike_step, "ITM-1"
    return atm, "ATM"


def get_expiry_recommendation() -> dict:
    today = datetime.now()
    days_to_thursday = (3 - today.weekday()) % 7
    if days_to_thursday == 0:
        days_to_thursday = 7
    
    weekly_expiry = today + timedelta(days=days_to_thursday)
    monthly_expiry = today + timedelta(days=days_to_thursday + 21)
    
    return {
        "weekly": weekly_expiry.strftime("%d-%b-%Y"),
        "monthly": monthly_expiry.strftime("%d-%b-%Y"),
        "days_to_weekly": days_to_thursday,
        "recommendation": "weekly" if days_to_thursday >= 2 else "next_weekly"
    }


def scan_tickers(tickers: list, progress_bar, status_text=None) -> tuple:
    all_signals = []
    ohlcv_data = {}
    h1_data = {}
    daily_data = {}
    weekly_data = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if st.session_state.get("scan_cancelled", False):
            if status_text:
                status_text.warning("Scan cancelled by user")
            break

        progress_bar.progress((i + 1) / total, text=f"Scanning {ticker}... ({i + 1}/{total})")

        df_5m = fetch_ohlcv(ticker, interval="5m", period=DATA_PERIOD)
        if df_5m is None:
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
        except Exception:
            continue

        if signals.empty:
            continue

        signals["ticker"] = ticker_label(ticker)
        signals["symbol"] = ticker
        all_signals.append(signals)
        ohlcv_data[ticker] = df_5m
        if df_h1 is not None:
            h1_data[ticker] = df_h1
        if df_daily is not None:
            daily_data[ticker] = df_daily
        if df_weekly is not None:
            weekly_data[ticker] = df_weekly

    if all_signals:
        return pd.concat(all_signals, ignore_index=True), ohlcv_data, h1_data, daily_data, weekly_data
    return pd.DataFrame(), ohlcv_data, h1_data, daily_data, weekly_data


def _scan_one_ticker(ticker: str) -> Optional[pd.DataFrame]:
    from scanner_base import get_index_type
    try:
        df_5m = fetch_ohlcv(ticker, interval="5m", period=DATA_PERIOD)
        if df_5m is None or len(df_5m) < MIN_ROWS:
            return None
        df_daily = fetch_ohlcv(ticker, interval="1d", period="1y")
        df_weekly = fetch_ohlcv(ticker, interval="1wk", period="5y")
        df_h1 = fetch_ohlcv(ticker, interval="1h", period="21d")
        cfg = Config.from_json()
        if ticker in INDEX_TICKERS:
            cfg.use_vol_filt = False
        cfg.index_type = get_index_type(ticker)
        _, signals = run_signals(df_5m, df_daily, cfg, df_weekly, df_h1=df_h1)
        if signals is not None and not signals.empty:
            signals["ticker"] = ticker_label(ticker)
            signals["symbol"] = ticker
            return signals
    except Exception:
        return None
    return None


def run_auto_scan():
    """Run scan silently and send new active trades to Telegram."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from scanner_base import get_index_type
    from trade_journal import load_journal, check_open_trades
    from telegram_notifier import send_message, format_trade_alert, already_sent, mark_sent

    all_tickers = list(INDEX_TICKERS)
    fno_file = os.path.join(os.path.dirname(__file__), "fno_tickers.csv")
    if os.path.exists(fno_file):
        try:
            df = pd.read_csv(fno_file)
            if "ticker" in df.columns:
                all_tickers.extend(df["ticker"].dropna().tolist())
        except Exception:
            pass

    n = len(all_tickers)
    status = st.sidebar.empty()
    bar = st.sidebar.progress(0, text=f"Auto-scan: 0/{n} tickers")

    all_signals = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_scan_one_ticker, t): t for t in all_tickers}
        for i, f in enumerate(as_completed(futures), 1):
            sig = f.result()
            if sig is not None:
                all_signals.append(sig)
            bar.progress(i / n, text=f"Auto-scan: {i}/{n} tickers")

    bar.empty()
    status.success(f"✅ Auto-scan done: {len(all_signals)} signals")

    if not all_signals:
        return

    signals_df = pd.concat(all_signals, ignore_index=True)
    record_scan_signals(signals_df)
    check_open_trades()

    journal = load_journal()
    sent = 0
    for t in journal:
        if t.get("status") != "open":
            continue
        if already_sent(t):
            continue
        msg = format_trade_alert(t)
        if send_message(msg):
            mark_sent(t)
            sent += 1

    if sent:
        status.success(f"✅ Auto-scan done: {len(all_signals)} signals, {sent} sent to Telegram")


def _find_pivots(series: pd.Series, window: int = 5):
    highs, lows = [], []
    arr = series.values
    idx = series.index
    n = len(arr)
    for i in range(window, n - window):
        if all(arr[i] >= arr[i - j] for j in range(1, window + 1)) and \
           all(arr[i] >= arr[i + j] for j in range(1, window + 1)):
            highs.append((idx[i], arr[i]))
        if all(arr[i] <= arr[i - j] for j in range(1, window + 1)) and \
           all(arr[i] <= arr[i + j] for j in range(1, window + 1)):
            lows.append((idx[i], arr[i]))
    return highs, lows


def _draw_trendlines(fig, df_chart, color_high="#ef5350", color_low="#26a69a"):
    if len(df_chart) < 20:
        return

    highs, lows = _find_pivots(df_chart["high"], window=4)
    lows_c, _ = _find_pivots(df_chart["low"], window=4)

    if len(highs) >= 2:
        h1, h2 = highs[-2], highs[-1]
        fig.add_shape(
            type="line",
            x0=h1[0], y0=h1[1], x1=h2[0], y1=h2[1],
            line=dict(color=color_high, width=1.5, dash="dot"),
            xref="x", yref="y",
        )
        fig.add_annotation(
            x=h2[0], y=h2[1], text="Resist", showarrow=False,
            font=dict(color=color_high, size=9),
            xref="x", yref="y", yshift=6,
        )

    if len(lows_c) >= 2:
        l1, l2 = lows_c[-2], lows_c[-1]
        fig.add_shape(
            type="line",
            x0=l1[0], y0=l1[1], x1=l2[0], y1=l2[1],
            line=dict(color=color_low, width=1.5, dash="dot"),
            xref="x", yref="y",
        )
        fig.add_annotation(
            x=l2[0], y=l2[1], text="Supp", showarrow=False,
            font=dict(color=color_low, size=9),
            xref="x", yref="y", yshift=-10,
        )


def _draw_pattern(fig, df_chart, pattern_name: str, is_ce: bool):
    if not pattern_name or len(df_chart) < 15:
        return

    name = pattern_name.lower()
    closes = df_chart["close"]
    highs = df_chart["high"]
    lows = df_chart["low"]
    xs = df_chart.index

    col_bull = "rgba(38,166,154,0.25)"
    col_bear = "rgba(239,83,80,0.25)"
    col_line = "#a78bfa"
    col_fill = col_bull if is_ce else col_bear

    def mark_zone(x0, x1, y_lo, y_hi, fill=col_fill, label=""):
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=y_lo, y1=y_hi,
            fillcolor=fill, line_color="rgba(167,139,250,0.4)",
            line_width=1, xref="x", yref="y", layer="below",
        )
        if label:
            fig.add_annotation(
                x=x0, y=y_hi, text=label, showarrow=False,
                font=dict(color=col_line, size=9),
                xref="x", yref="y", yanchor="bottom",
            )

    def connect(pts, color=col_line, width=1.5, dash="solid"):
        if len(pts) < 2:
            return
        for i in range(len(pts) - 1):
            fig.add_shape(
                type="line",
                x0=pts[i][0], y0=pts[i][1],
                x1=pts[i + 1][0], y1=pts[i + 1][1],
                line=dict(color=color, width=width, dash=dash),
                xref="x", yref="y",
            )

    n = len(df_chart)
    seg = max(n // 5, 3)

    try:
        if "head" in name and "shoulder" in name:
            inv = "inverse" in name or is_ce
            pts_x = [xs[seg], xs[2 * seg], xs[int(2.5 * seg)], xs[3 * seg], xs[4 * seg]]
            if inv:
                base = lows.iloc[:4 * seg]
                pts_y = [
                    base.iloc[seg],
                    base.iloc[2 * seg - 1],
                    base.min(),
                    base.iloc[3 * seg - 1],
                    base.iloc[min(4 * seg - 1, len(base) - 1)],
                ]
            else:
                base = highs.iloc[:4 * seg]
                pts_y = [
                    base.iloc[seg],
                    base.iloc[2 * seg - 1],
                    base.max(),
                    base.iloc[3 * seg - 1],
                    base.iloc[min(4 * seg - 1, len(base) - 1)],
                ]
            connect(list(zip(pts_x, pts_y)), color=col_line, width=2)
            neck_y = (pts_y[0] + pts_y[4]) / 2
            fig.add_hline(
                y=neck_y, line_dash="dash", line_color=col_line, line_width=1.2,
                annotation_text="Neckline", annotation_position="left",
                annotation_font_color=col_line, annotation_font_size=9,
            )
            mark_zone(pts_x[0], pts_x[-1], min(pts_y) * 0.998, max(pts_y) * 1.002, label=pattern_name)

        elif "double top" in name or "double bottom" in name:
            is_top = "top" in name
            mid = n // 2
            q = n // 4
            if is_top:
                p1 = (xs[q], highs.iloc[q])
                p2 = (xs[mid], closes.iloc[mid])
                p3 = (xs[3 * q], highs.iloc[3 * q])
                connect([p1, p2, p3], color=col_line, width=2)
                neck_y = closes.iloc[mid]
            else:
                p1 = (xs[q], lows.iloc[q])
                p2 = (xs[mid], closes.iloc[mid])
                p3 = (xs[3 * q], lows.iloc[3 * q])
                connect([p1, p2, p3], color=col_line, width=2)
                neck_y = closes.iloc[mid]
            fig.add_hline(
                y=neck_y, line_dash="dash", line_color=col_line, line_width=1.2,
                annotation_text="Neckline", annotation_position="left",
                annotation_font_color=col_line, annotation_font_size=9,
            )
            mark_zone(xs[q - 2], xs[min(3 * q + 2, n - 1)],
                      min(lows.iloc[q - 2:3 * q + 2]) * 0.998,
                      max(highs.iloc[q - 2:3 * q + 2]) * 1.002,
                      label=pattern_name)

        elif "triangle" in name:
            high_pts = _find_pivots(highs, window=3)[0][-3:]
            low_pts = _find_pivots(lows, window=3)[1][-3:]
            if len(high_pts) >= 2:
                connect(high_pts[:2], color="#ef5350", width=1.5, dash="dot")
            if len(low_pts) >= 2:
                connect(low_pts[:2], color="#26a69a", width=1.5, dash="dot")
            mark_zone(xs[n // 4], xs[n - 1],
                      lows.iloc[n // 4:].min() * 0.999,
                      highs.iloc[n // 4:].max() * 1.001,
                      fill="rgba(167,139,250,0.08)", label=pattern_name)

        elif "flag" in name or "wedge" in name:
            pole_end = n // 3
            pole_lo = lows.iloc[:pole_end].min()
            pole_hi = highs.iloc[:pole_end].max()
            mark_zone(xs[0], xs[pole_end], pole_lo, pole_hi,
                      fill="rgba(167,139,250,0.12)", label="Pole")
            flag_hi = highs.iloc[pole_end:].max()
            flag_lo = lows.iloc[pole_end:].min()
            mark_zone(xs[pole_end], xs[-1], flag_lo, flag_hi, fill=col_fill, label=pattern_name)
            h_pts = _find_pivots(highs.iloc[pole_end:], window=3)[0]
            l_pts = _find_pivots(lows.iloc[pole_end:], window=3)[1]
            if len(h_pts) >= 2:
                connect(h_pts[-2:], color="#ef5350", width=1.2, dash="dot")
            if len(l_pts) >= 2:
                connect(l_pts[-2:], color="#26a69a", width=1.2, dash="dot")

        elif "cup" in name:
            cup_start = n // 4
            cup_end = 3 * n // 4
            cup_lo = lows.iloc[cup_start:cup_end].min()
            cup_hi = highs.iloc[:cup_start].max()
            cup_points = []
            for i in range(cup_start, cup_end, max(1, (cup_end - cup_start) // 10)):
                cup_points.append((xs[i], lows.iloc[i]))
            if len(cup_points) >= 2:
                connect(cup_points, color=col_line, width=2)
            mark_zone(xs[cup_start], xs[cup_end], cup_lo * 0.998, cup_hi * 1.002, fill=col_fill, label="Cup")
            mark_zone(xs[cup_end], xs[-1],
                      lows.iloc[cup_end:].min() * 0.998,
                      highs.iloc[cup_end:].max() * 1.002,
                      fill="rgba(167,139,250,0.15)", label="Handle")

        elif "rounding" in name:
            points = []
            for i in range(0, n, max(1, n // 15)):
                points.append((xs[i], lows.iloc[i]))
            if len(points) >= 2:
                connect(points, color=col_line, width=2)
            mark_zone(xs[0], xs[-1], lows.min() * 0.998, highs.max() * 1.002, fill=col_fill, label=pattern_name)
    except Exception:
        pass


def create_weekly_chart(df: pd.DataFrame, signal: dict) -> go.Figure:
    is_ce = "CE" in signal["type"]
    entry = signal["entry"]
    target = signal["target"]
    sl = signal["sl"]
    level_price = signal.get("level", signal.get("support", signal.get("resistance", entry)))

    df_chart = df.tail(260)
    if df_chart.empty:
        df_chart = df.tail(52)

    last_bar_time = df_chart.index[-1] if len(df_chart) > 0 else None

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df_chart.index,
            open=df_chart["open"],
            high=df_chart["high"],
            low=df_chart["low"],
            close=df_chart["close"],
            name="Weekly",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )

    fig.add_hline(y=entry, line_dash="solid", line_color="#ffeb3b", line_width=2,
                  annotation_text=f"Entry: ₹{entry}", annotation_position="right",
                  annotation_font_color="#ffeb3b", annotation_font_size=11)
    fig.add_hline(y=target, line_dash="solid", line_color="#4caf50" if is_ce else "#f44336", line_width=2,
                  annotation_text=f"Target: ₹{target}", annotation_position="right",
                  annotation_font_color="#4caf50" if is_ce else "#f44336", annotation_font_size=11)
    fig.add_hline(y=sl, line_dash="solid", line_color="#f44336" if is_ce else "#4caf50", line_width=2,
                  annotation_text=f"SL: ₹{sl}", annotation_position="right",
                  annotation_font_color="#f44336" if is_ce else "#4caf50", annotation_font_size=11)
    fig.add_hline(y=level_price, line_dash="dash", line_color="#2196f3" if is_ce else "#ff9800", line_width=1.5,
                  annotation_text=f"{'Support' if is_ce else 'Resistance'}", annotation_position="left",
                  annotation_font_color="#2196f3" if is_ce else "#ff9800", annotation_font_size=10)

    _draw_trendlines(fig, df_chart)

    weekly_pattern_name = signal.get("weekly_pattern") or signal.get("pattern")
    if weekly_pattern_name:
        _draw_pattern(fig, df_chart, weekly_pattern_name, is_ce)

    last_bar_str = last_bar_time.strftime("%d-%b-%Y") if last_bar_time else "N/A"
    pattern_title = f" | {weekly_pattern_name}" if weekly_pattern_name else ""
    fig.update_layout(
        title=dict(text=f"Weekly Chart — {signal['ticker']}{pattern_title} | Last: {last_bar_str}", font=dict(size=14, color="#fff"), x=0.5),
        template="plotly_dark", height=400, xaxis_rangeslider_visible=False,
        showlegend=False, plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        margin=dict(l=10, r=60, t=50, b=10), hovermode="x unified",
        dragmode="drawline",
        modebar_add=["drawline", "drawopenpath", "drawclosedpath", "drawcircle", "drawrect", "eraseshape"],
    )
    fig.update_xaxes(showticklabels=True, tickformat="%d-%b\n%Y")

    return fig


def create_daily_chart(df: pd.DataFrame, signal: dict) -> go.Figure:
    is_ce = "CE" in signal["type"]
    entry = signal["entry"]
    target = signal["target"]
    sl = signal["sl"]
    level_price = signal.get("level", signal.get("support", signal.get("resistance", entry)))

    df_trading = df[df.index.dayofweek < 5]
    df_chart = df_trading.tail(250)
    if df_chart.empty:
        df_chart = df.tail(250)

    last_bar_time = df_chart.index[-1] if len(df_chart) > 0 else None

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df_chart.index,
            open=df_chart["open"],
            high=df_chart["high"],
            low=df_chart["low"],
            close=df_chart["close"],
            name="Daily",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )

    if "volume" in df_chart.columns:
        fig.add_trace(
            go.Bar(
                x=df_chart.index,
                y=df_chart["volume"],
                name="Volume",
                marker_color="#546e7a",
                opacity=0.4,
                yaxis="y2",
            )
        )

    fig.add_hline(y=entry, line_dash="solid", line_color="#ffeb3b", line_width=2,
                  annotation_text=f"Entry: ₹{entry}", annotation_position="right",
                  annotation_font_color="#ffeb3b", annotation_font_size=11)
    fig.add_hline(y=target, line_dash="solid", line_color="#4caf50" if is_ce else "#f44336", line_width=2,
                  annotation_text=f"Target: ₹{target}", annotation_position="right",
                  annotation_font_color="#4caf50" if is_ce else "#f44336", annotation_font_size=11)
    fig.add_hline(y=sl, line_dash="solid", line_color="#f44336" if is_ce else "#4caf50", line_width=2,
                  annotation_text=f"SL: ₹{sl}", annotation_position="right",
                  annotation_font_color="#f44336" if is_ce else "#4caf50", annotation_font_size=11)
    fig.add_hline(y=level_price, line_dash="dash", line_color="#2196f3" if is_ce else "#ff9800", line_width=1.5,
                  annotation_text=f"{'Support' if is_ce else 'Resistance'}", annotation_position="left",
                  annotation_font_color="#2196f3" if is_ce else "#ff9800", annotation_font_size=10)

    _draw_trendlines(fig, df_chart)

    pattern_name = signal.get("pattern")
    if pattern_name:
        _draw_pattern(fig, df_chart, pattern_name, is_ce)

    last_bar_str = last_bar_time.strftime("%d-%b-%Y") if last_bar_time else "N/A"
    pattern_title = f" | {pattern_name}" if pattern_name else ""
    fig.update_layout(
        title=dict(text=f"Daily Chart — {signal['ticker']}{pattern_title} | Last: {last_bar_str}", font=dict(size=14, color="#fff"), x=0.5),
        template="plotly_dark", height=400, xaxis_rangeslider_visible=False,
        yaxis=dict(side="right", gridcolor="#333", showgrid=True, domain=[0.2, 1]),
        yaxis2=dict(side="right", gridcolor="#333", showgrid=False, domain=[0, 0.15]),
        showlegend=False, plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        margin=dict(l=10, r=60, t=50, b=10), hovermode="x unified",
        dragmode="drawline",
        modebar_add=["drawline", "drawopenpath", "drawclosedpath", "drawcircle", "drawrect", "eraseshape"],
    )
    fig.update_xaxes(showticklabels=True, tickformat="%d-%b")

    return fig


def create_signal_chart(df: pd.DataFrame, signal: dict, num_bars: int = 100) -> go.Figure:
    is_ce = "CE" in signal["type"]
    entry = signal["entry"]
    target = signal["target"]
    sl = signal["sl"]
    level_price = signal.get("support", signal.get("resistance", entry))

    signal_time = signal["datetime"]
    if hasattr(signal_time, "to_pydatetime"):
        signal_time = signal_time.to_pydatetime()

    MARKET_OPEN = 9 * 60 + 15
    MARKET_CLOSE = 15 * 60 + 30

    df_trading = df.copy()
    df_trading = df_trading[
        (df_trading.index.hour * 60 + df_trading.index.minute >= MARKET_OPEN) &
        (df_trading.index.hour * 60 + df_trading.index.minute <= MARKET_CLOSE)
    ]
    df_trading = df_trading[df_trading.index.dayofweek < 5]

    df_chart = df_trading.tail(num_bars)
    if df_chart.empty:
        df_chart = df.tail(num_bars)

    last_bar_time = df_chart.index[-1] if len(df_chart) > 0 else None

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df_chart.index,
            open=df_chart["open"],
            high=df_chart["high"],
            low=df_chart["low"],
            close=df_chart["close"],
            name="5m",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )

    if "ema" in df_chart.columns:
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["ema"], mode="lines",
                                 name="EMA", line=dict(color="#ff9800", width=1.5, dash="dash"), opacity=0.7))

    if "volume" in df_chart.columns:
        fig.add_trace(
            go.Bar(
                x=df_chart.index,
                y=df_chart["volume"],
                name="Volume",
                marker_color="#546e7a",
                opacity=0.4,
                yaxis="y2",
            )
        )

    fvg_points = detect_fvg_points(df_chart)
    add_fvg_shapes(fig, fvg_points, is_bullish=is_ce)

    fig.add_hline(y=entry, line_dash="solid", line_color="#ffeb3b", line_width=2,
                  annotation_text=f"Entry: ₹{entry}", annotation_position="right",
                  annotation_font_color="#ffeb3b", annotation_font_size=11)

    fig.add_hline(y=sl, line_dash="solid", line_color="#f44336", line_width=2,
                  annotation_text=f"SL: ₹{sl}", annotation_position="right",
                  annotation_font_color="#f44336", annotation_font_size=11)
    fig.add_hline(y=target, line_dash="dot", line_color="#4caf50", line_width=2,
                  annotation_text=f"Target: ₹{target}", annotation_position="right",
                  annotation_font_color="#4caf50", annotation_font_size=11)

    fig.add_hline(y=level_price, line_dash="dash", line_color="#2196f3" if is_ce else "#ff9800", line_width=1.5,
                  annotation_text=f"{'Support' if is_ce else 'Resistance'}", annotation_position="left",
                  annotation_font_color="#2196f3" if is_ce else "#ff9800", annotation_font_size=10)

    _draw_trendlines(fig, df_chart)

    pattern_name = signal.get("pattern")
    if pattern_name:
        _draw_pattern(fig, df_chart, pattern_name, is_ce)

    last_bar_str = last_bar_time.strftime("%d-%b-%Y %H:%M") if last_bar_time else "N/A"
    pattern_title = f" | {pattern_name}" if pattern_name else ""
    fig.update_layout(
        title=dict(text=f"5 Min — {signal['ticker']}{pattern_title} | {last_bar_str}", font=dict(size=14, color="#fff"), x=0.5),
        template="plotly_dark", height=400, xaxis_rangeslider_visible=False,
        yaxis=dict(side="right", gridcolor="#333", showgrid=True, domain=[0.2, 1]),
        yaxis2=dict(side="right", gridcolor="#333", showgrid=False, domain=[0, 0.15]),
        showlegend=True, plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9, color="#8892b0")),
        margin=dict(l=10, r=60, t=50, b=10), hovermode="x unified",
        xaxis=dict(type="date", rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[15.5, 9.25], pattern="hour")]),
        dragmode="drawline",
        modebar_add=["drawline", "drawopenpath", "drawclosedpath", "drawcircle", "drawrect", "eraseshape"],
    )
    fig.update_xaxes(showticklabels=True, tickformat="%d-%b %H:%M")

    return fig


def detect_fvg_points(df: pd.DataFrame, min_gap_atr: float = 0.3) -> list:
    fvg_list = []
    if len(df) < 3:
        return fvg_list
    atr_series = (df["high"] - df["low"]).rolling(14).mean()
    for i in range(2, len(df)):
        atr_i = atr_series.iloc[i]
        if pd.isna(atr_i) or atr_i == 0:
            continue
        prev_high = df["high"].iloc[i - 1]
        prev_low = df["low"].iloc[i - 1]
        curr_low = df["low"].iloc[i]
        curr_high = df["high"].iloc[i]

        gap_up = curr_low > prev_high
        gap_down = curr_high < prev_low

        if gap_up:
            gap_size = curr_low - prev_high
            if gap_size >= min_gap_atr * atr_i:
                fvg_list.append((df.index[i], df.index[i - 1], prev_high, curr_low))
        elif gap_down:
            gap_size = prev_low - curr_high
            if gap_size >= min_gap_atr * atr_i:
                fvg_list.append((df.index[i], df.index[i - 1], curr_high, prev_low))
    return fvg_list


def add_fvg_shapes(fig, fvg_points: list, is_bullish: bool = True):
    for fvg_dt, prev_dt, lo, hi in fvg_points:
        color = "rgba(38, 166, 154, 0.25)" if is_bullish else "rgba(239, 83, 80, 0.25)"
        fig.add_shape(
            type="rect",
            x0=prev_dt, x1=fvg_dt,
            y0=lo, y1=hi,
            fillcolor=color,
            line=dict(width=0),
            layer="below",
            xref="x", yref="y",
        )


def create_h1_chart(df: pd.DataFrame, signal: dict, num_bars: int = 50) -> go.Figure:
    is_ce = "CE" in signal["type"]
    entry = signal["entry"]
    target = signal["target"]
    sl = signal["sl"]
    level_price = signal.get("support", signal.get("resistance", entry))

    df_chart = df.tail(num_bars)
    last_bar_time = df_chart.index[-1] if len(df_chart) > 0 else None

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df_chart.index,
            open=df_chart["open"],
            high=df_chart["high"],
            low=df_chart["low"],
            close=df_chart["close"],
            name="1H",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )

    ema = df_chart["close"].rolling(20).mean()
    if not ema.isna().all():
        fig.add_trace(go.Scatter(x=df_chart.index, y=ema, mode="lines",
                                 name="EMA", line=dict(color="#ff9800", width=1.5, dash="dash"), opacity=0.7))

    if "volume" in df_chart.columns:
        fig.add_trace(
            go.Bar(
                x=df_chart.index,
                y=df_chart["volume"],
                name="Volume",
                marker_color="#546e7a",
                opacity=0.4,
                yaxis="y2",
            )
        )

    fvg_points = detect_fvg_points(df_chart)
    add_fvg_shapes(fig, fvg_points, is_bullish=is_ce)

    fig.add_hline(y=entry, line_dash="solid", line_color="#ffeb3b", line_width=2,
                  annotation_text=f"Entry: ₹{entry}", annotation_position="right",
                  annotation_font_color="#ffeb3b", annotation_font_size=11)
    fig.add_hline(y=sl, line_dash="solid", line_color="#f44336", line_width=2,
                  annotation_text=f"SL: ₹{sl}", annotation_position="right",
                  annotation_font_color="#f44336", annotation_font_size=11)
    fig.add_hline(y=target, line_dash="dot", line_color="#4caf50", line_width=2,
                  annotation_text=f"Target: ₹{target}", annotation_position="right",
                  annotation_font_color="#4caf50", annotation_font_size=11)
    fig.add_hline(y=level_price, line_dash="dash", line_color="#2196f3" if is_ce else "#ff9800", line_width=1.5,
                  annotation_text=f"{'Support' if is_ce else 'Resistance'}", annotation_position="left",
                  annotation_font_color="#2196f3" if is_ce else "#ff9800", annotation_font_size=10)

    last_bar_str = last_bar_time.strftime("%d-%b-%Y %H:%M") if last_bar_time else "N/A"
    fig.update_layout(
        title=dict(text=f"1 Hour — {signal.get('ticker', '')} | {last_bar_str}", font=dict(size=14, color="#fff"), x=0.5),
        template="plotly_dark", height=400, xaxis_rangeslider_visible=False,
        yaxis=dict(side="right", gridcolor="#333", showgrid=True, domain=[0.2, 1]),
        yaxis2=dict(side="right", gridcolor="#333", showgrid=False, domain=[0, 0.15]),
        showlegend=True, plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9, color="#8892b0")),
        margin=dict(l=10, r=60, t=50, b=10), hovermode="x unified",
        xaxis=dict(type="date", rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[15.5, 9.25], pattern="hour")]),
        dragmode="drawline",
        modebar_add=["drawline", "drawopenpath", "drawclosedpath", "drawcircle", "drawrect", "eraseshape"],
    )
    fig.update_xaxes(showticklabels=True, tickformat="%d-%b %H:%M")

    return fig


def run_quick_backtest(signals_df: pd.DataFrame, ohlcv_data: dict) -> dict:
    """Quick backtest on scanned signals."""
    if signals_df.empty:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl": 0, "equity_curve": [100000]}
    
    wins = 0
    losses = 0
    total_pnl = 0
    equity_curve = [100000]
    
    for _, signal in signals_df.iterrows():
        pnl = 0
        outcome = signal.get("outcome", "pending")
        if outcome == "target_hit":
            wins += 1
            pnl = abs(signal["target"] - signal["entry"])
            total_pnl += pnl
        elif outcome == "sl_hit":
            losses += 1
            pnl = -abs(signal["sl"] - signal["entry"])
            total_pnl += pnl
        elif outcome in ("partial_target", "eod_exit"):
            exit_price = signal.get("exit_price", signal.get("entry", 0))
            entry = signal.get("entry", 0)
            is_ce = "CE" in str(signal.get("type", ""))
            pnl = (exit_price - entry) if is_ce else (entry - exit_price)
            if pnl > 0:
                wins += 1
            else:
                losses += 1
            total_pnl += pnl
        
        equity_curve.append(equity_curve[-1] + pnl * 10)
    
    total = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0
    
    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "equity_curve": equity_curve
    }


st.set_page_config(page_title="Combined Strategy", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# ─── Auth Gate ────────────────────────────────────────────────
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.markdown("""
        <style>
        html, body, .stApp { background: #0a0b16 !important; }
        header, #stDecoration, .stDeployButton, .stAppDeployButton,
        [data-testid="stHeader"], [data-testid="stToolbar"],
        [data-testid="stDecoration"], [data-testid="stStatusWidget"],
        [data-testid="stSidebar"] { display: none !important; }
        .main, .block-container, .stApp { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
        section[data-testid="stSidebar"] { display: none !important; }
        .stTextInput input { background:#0e0f20 !important;border:1px solid #2a2b55 !important;border-radius:10px !important;color:#e0e0ff !important;padding:11px 14px !important; }
        .stTextInput input:focus { border-color:#5a5aff !important; }
        .stButton button { width:100% !important;padding:12px !important;border-radius:10px !important;background:linear-gradient(135deg,#4a4aff,#6a3aff) !important;color:white !important;border:none !important;font-size:15px !important;font-weight:600 !important; }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<h2 style='color:#f0f0ff;font-size:22px;text-align:center;margin:60px 0 4px;font-weight:700'>🔒 Supply & Demand Scanner</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#7878aa;font-size:13px;text-align:center;margin:0 0 24px'>Sign in to access the dashboard</p>", unsafe_allow_html=True)
        _err = st.empty()
        user = st.text_input("Username", placeholder="Username", key="au", label_visibility="collapsed")
        pwd = st.text_input("Password", type="password", placeholder="Password", key="ap", label_visibility="collapsed")
    try:
        auth_user = st.secrets.get("auth", {}).get("username", "")
        auth_pass = st.secrets.get("auth", {}).get("password", "")
    except Exception:
        auth_user = os.environ.get("AUTH_USERNAME", "")
        auth_pass = os.environ.get("AUTH_PASSWORD", "")
    if st.button("Sign In", key="ab", use_container_width=True):
        if user == auth_user and pwd == auth_pass:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            _err.error("✗ Invalid credentials")
    st.stop()

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background: #0d0f1a !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f1123 0%, #141729 100%) !important; border-right: 1px solid #1e2340 !important; }
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
.hero-header { background: linear-gradient(135deg, #0f3460 0%, #16213e 40%, #1a1a2e 100%); border: 1px solid #1e3a5f; border-radius: 16px; padding: 28px 36px; margin-bottom: 20px; position: relative; overflow: hidden; }
.hero-title { font-size: 2.1rem; font-weight: 800; background: linear-gradient(90deg, #00d4ff, #7b2ff7, #00d4ff); background-size: 200%; -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 6px 0; letter-spacing: -0.5px; }
.hero-sub { color: #8892b0; font-size: 0.95rem; margin: 0; }
.hero-badges { display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }
.badge { background: rgba(0,212,255,0.08); border: 1px solid rgba(0,212,255,0.25); color: #00d4ff; border-radius: 20px; padding: 4px 14px; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.5px; }
.badge.purple { background: rgba(123,47,247,0.1); border-color: rgba(123,47,247,0.35); color: #a78bfa; }
.badge.green { background: rgba(16,185,129,0.1); border-color: rgba(16,185,129,0.3); color: #10b981; }
.badge.red { background: rgba(239,68,68,0.1); border-color: rgba(239,68,68,0.3); color: #ef4444; }
.stat-card { background: linear-gradient(135deg, #141729 0%, #1a1e35 100%); border: 1px solid #1e2a45; border-radius: 14px; padding: 18px 20px; text-align: center; transition: border-color 0.2s; }
.stat-card:hover { border-color: #00d4ff55; }
.stat-label { color: #64748b; font-size: 0.75rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
.stat-value { color: #e2e8f0; font-size: 2rem; font-weight: 800; line-height: 1; }
.stat-value.cyan { color: #00d4ff; }
.stat-value.green { color: #10b981; }
.stat-value.red { color: #ef4444; }
.stat-value.amber { color: #f59e0b; }
[data-testid="stExpander"] { background: linear-gradient(135deg, #141729 0%, #1a1e35 100%) !important; border: 1px solid #1e2a45 !important; border-radius: 14px !important; margin-bottom: 10px !important; }
[data-testid="stExpander"]:hover { border-color: #00d4ff44 !important; }
[data-testid="stTabs"] [role="tablist"] { background: #0f1123; border-radius: 10px; padding: 4px; gap: 4px; border: 1px solid #1e2340; }
[data-testid="stTabs"] [role="tab"] { background: transparent; color: #64748b !important; border-radius: 8px; font-size: 0.82rem; font-weight: 600; padding: 6px 16px; border: none !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: linear-gradient(135deg, #00d4ff22, #7b2ff722) !important; color: #00d4ff !important; border: 1px solid #00d4ff33 !important; }
[data-testid="baseButton-primary"] { background: linear-gradient(135deg, #00d4ff, #7b2ff7) !important; border: none !important; border-radius: 10px !important; font-weight: 700 !important; color: #fff !important; }
.section-heading { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; display: flex; align-items: center; gap: 8px; margin-bottom: 14px; }
.section-heading .dot { width: 8px; height: 8px; background: linear-gradient(135deg, #00d4ff, #7b2ff7); border-radius: 50%; display: inline-block; }
hr { border-color: #1e2340 !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0f1a; }
::-webkit-scrollbar-thumb { background: #1e2a45; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

if not st.session_state.journal_checked:
    try:
        check_open_trades()
        _refresh_journal_state()
    except Exception:
        pass
    st.session_state.journal_checked = True

config = load_config()
general_config = config.get("general", {})
premium_config = config.get("premium_strategy", {})
alerts_config = config.get("alerts", {})

now_ist = datetime.now(IST)
st.markdown(f"""
<div class="hero-header">
    <div class="hero-title">3-Step A+ S&D Strategy</div>
    <div class="hero-sub">Supply & Demand Zones + BOS Trend | Trade with Pat Method</div>
    <div class="hero-badges">
        <span class="badge">Institutional Zones</span>
        <span class="badge">FVG Confirmation</span>
        <span class="badge purple">Break of Structure</span>
        <span class="badge green">Slow Momentum Entry</span>
        <span class="badge red">1:1.5 R:R Target</span>
    </div>
</div>
""", unsafe_allow_html=True)

MAIN_TABS = ["Scanner", "S/R Levels", "Backtest", "Dashboard", "Journal"]
main_tab = st.sidebar.radio("Navigate", MAIN_TABS, index=0)

st.sidebar.divider()

if main_tab == "Scanner":
    st.sidebar.header("Scanner Settings")
    
    ticker_source = st.sidebar.radio("Ticker Source", ["All FNO + Indices", "Indices Only", "Custom"])
    
    custom_tickers = []
    if ticker_source == "Custom":
        try:
            all_available_tickers = load_tickers(TICKERS_FILE)
            ticker_options = []
            ticker_map = {}
            for t in all_available_tickers:
                label = TICKER_LABELS.get(t, t.replace(".NS", "").replace(".BO", ""))
                formatted = f"{label} ({t})"
                ticker_options.append(formatted)
                ticker_map[formatted] = t
            
            selected = st.sidebar.multiselect("Search tickers...", options=ticker_options, default=[], max_selections=20)
            custom_tickers = [ticker_map[s] for s in selected if s in ticker_map]
            
            if custom_tickers:
                st.sidebar.caption(f"Selected: {len(custom_tickers)} ticker(s)")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    
    st.sidebar.divider()
    st.sidebar.header("Filters")
    
    signal_type_filter = st.sidebar.multiselect("Signal Type", ["CE BUY", "PE BUY"], default=["CE BUY", "PE BUY"])
    grade_filter = st.sidebar.multiselect("Grade", ["A", "B", "C", "D"], default=["A", "B", "C"])
    min_rr = st.sidebar.slider("Min RR", 1.0, 5.0, float(config.get("general", {}).get("min_reward_risk_ratio", 2.0)), 0.5)
    min_quality = st.sidebar.slider("Min Quality %", 0, 100, 55, 5)
    fvg_only = st.sidebar.checkbox("FVG confirmed only", False)
    strong_only = st.sidebar.checkbox("Strong signals only", False)
    slow_mom_only = st.sidebar.checkbox("Slow momentum only", False)
    
    st.sidebar.divider()
    show_chart_bars = st.sidebar.slider("Chart Bars", 50, 200, 100, 10)
    
    batch_size = st.sidebar.select_slider("Batch Size", options=[10, 25, 50, 100, 250], value=50, help="Smaller batches = more responsive UI")
    
    col_scan, col_cancel, col_batch = st.sidebar.columns(3)
    scan_button = col_scan.button("Run Scan", type="primary")
    cancel_button = col_cancel.button("Cancel")
    batch_button = col_batch.button("Next Batch")

    st.sidebar.divider()
    auto_on = st.sidebar.checkbox("🔄 Auto-Scan (every 15 min)", key="auto_scan_on",
                                  help="Automatically scans every 15 min during market hours (9:15-15:30 IST)")
    if auto_on:
        now = datetime.now(IST)
        in_market = 9 * 60 + 15 <= now.hour * 60 + now.minute <= 15 * 60 + 30 and now.weekday() < 5
        already_scanning = st.session_state.get("scanning", False)

        last = st.session_state.get("auto_last_scan", None)
        count = st.session_state.get("auto_scan_count", 0)
        if last:
            elapsed = (now - datetime.fromisoformat(last)).total_seconds()
            next_in = max(0, 900 - int(elapsed))
            st.sidebar.caption(f"Last scan: {last[-8:]} | Next: ~{next_in//60}m{next_in%60}s")
        st.sidebar.caption(f"Today's scans: {count}" + (" 🟢 Market open" if in_market else " 🔴 Market closed"))
        if already_scanning:
            st.sidebar.warning("⏳ Scan in progress...", icon=False)

        if not already_scanning:
            last_scan = st.session_state.get("auto_last_scan_dt", None)
            should_scan = False
            if last_scan is None:
                should_scan = True
            else:
                elapsed = (datetime.now() - last_scan).total_seconds()
                if elapsed >= 900:
                    should_scan = True

            if should_scan and in_market:
                st.session_state.auto_last_scan_dt = datetime.now()
                st.session_state.auto_last_scan = datetime.now(IST).isoformat()
                st.session_state.auto_scan_count = st.session_state.get("auto_scan_count", 0) + 1
                run_auto_scan()

        st_autorefresh(interval=1800000, key="auto_refresh")
    
    if cancel_button:
        st.session_state.scan_cancelled = True
        st.session_state.scanning = False
        st.session_state.current_batch = 0
        st.rerun()
    
    if scan_button:
        st.session_state.scan_cancelled = False
        st.session_state.scanning = True
        st.session_state.current_batch = 0
        st.session_state.batch_signals = []
        st.session_state.batch_ohlcv = {}
        st.session_state.batch_daily = {}
        st.session_state.batch_weekly = {}
    
    if scan_button or batch_button:
        if batch_button and not st.session_state.get("scanning", False):
            st.warning("Click 'Run Scan' first to start scanning")
        else:
            try:
                if ticker_source == "All FNO + Indices":
                    fno_tickers = load_tickers(TICKERS_FILE)
                    indices = load_indices(INDICES_FILE)
                    all_tickers = indices + fno_tickers
                elif ticker_source == "Indices Only":
                    all_tickers = load_indices(INDICES_FILE)
                else:
                    all_tickers = custom_tickers
                    if custom_tickers:
                        st.sidebar.caption(f"Custom scan: {len(custom_tickers)} ticker(s)")
                    else:
                        st.sidebar.warning("Select tickers from the list above")

                if not all_tickers:
                    st.error("No tickers to scan!")
                else:
                    current_batch = st.session_state.get("current_batch", 0)
                    start_idx = current_batch * batch_size
                    end_idx = min(start_idx + batch_size, len(all_tickers))
                    tickers = all_tickers[start_idx:end_idx]
                    
                    if start_idx >= len(all_tickers):
                        st.success("All tickers scanned!")
                        st.session_state.scanning = False
                    else:
                        status_placeholder = st.empty()
                        status_placeholder.info(f"Batch {current_batch + 1}: Scanning {start_idx + 1}-{end_idx} of {len(all_tickers)} tickers")
                        progress_bar = st.progress(0, text="Starting batch...")
                        start_time = datetime.now()

                        result = scan_tickers(tickers, progress_bar, status_placeholder)
                        signals_df, ohlcv_data, h1_data, daily_data, weekly_data = result
                        elapsed = (datetime.now() - start_time).total_seconds()

                        batch_signals = st.session_state.get("batch_signals", [])
                        batch_ohlcv = st.session_state.get("batch_ohlcv", {})
                        batch_h1 = st.session_state.get("batch_h1", {})
                        batch_daily = st.session_state.get("batch_daily", {})
                        batch_weekly = st.session_state.get("batch_weekly", {})

                        if not signals_df.empty:
                            batch_signals.append(signals_df)
                            batch_ohlcv.update(ohlcv_data)
                            batch_h1.update(h1_data)
                            batch_daily.update(daily_data)
                            batch_weekly.update(weekly_data)
                            st.session_state.batch_signals = batch_signals
                            st.session_state.batch_ohlcv = batch_ohlcv
                            st.session_state.batch_h1 = batch_h1
                            st.session_state.batch_daily = batch_daily
                            st.session_state.batch_weekly = batch_weekly

                        total_signals = sum(len(s) for s in batch_signals)
                        status_placeholder.success(f"Batch {current_batch + 1} done in {elapsed:.1f}s | Total signals: {total_signals}")

                        st.session_state.current_batch = current_batch + 1

                        if end_idx >= len(all_tickers):
                            st.session_state.scanning = False
                            st.balloons()
                            if batch_signals:
                                st.session_state.signals_df = pd.concat(batch_signals, ignore_index=True)
                                st.session_state.ohlcv_data = batch_ohlcv
                                st.session_state.h1_data = batch_h1
                                st.session_state.daily_data = batch_daily
                                st.session_state.weekly_data = batch_weekly
                                st.session_state.scan_done = True
                                # Auto-record signals to journal & check open trades
                                signals_concat = pd.concat(batch_signals, ignore_index=True)
                                try:
                                    added = record_scan_signals(signals_concat)
                                    if added:
                                        check_open_trades()
                                        _refresh_journal_state()
                                except Exception:
                                    pass
                                st.success(f"Scan complete! Found {total_signals} signal(s)")
                            else:
                                st.session_state.signals_df = pd.DataFrame()
                                st.session_state.scan_done = True
                                st.info(f"Scan complete! No signals found for {len(tickers)} ticker(s)")
                        else:
                            remaining = len(all_tickers) - end_idx
                            st.info(f"Click 'Next Batch' to continue ({remaining} tickers remaining)")
                            
            except Exception as e:
                st.session_state.scanning = False
                st.error(f"Error: {e}")

    if st.session_state.signals_df is not None:
        if st.session_state.signals_df.empty:
            st.info("No signals found. Try adjusting filters or selecting different tickers.")
        else:
            signals_df = st.session_state.signals_df.copy()
            ohlcv_data = st.session_state.ohlcv_data
            h1_data = st.session_state.h1_data
            daily_data = st.session_state.daily_data
            weekly_data = st.session_state.weekly_data
        
            st.caption(f"Raw signals: {len(signals_df)} | Columns: {list(signals_df.columns)[:10]}...")
        
            if "type" in signals_df.columns:
                signals_df = signals_df[signals_df["type"].isin(signal_type_filter)]
            if "entry_grade" in signals_df.columns:
                signals_df = signals_df[signals_df["entry_grade"].isin(grade_filter)]
            if "entry_quality" in signals_df.columns:
                signals_df = signals_df[signals_df["entry_quality"] >= min_quality]
            if "rr" in signals_df.columns:
                signals_df = signals_df[signals_df["rr"] >= min_rr]

            if fvg_only and "fvg_present" in signals_df.columns:
                signals_df = signals_df[signals_df["fvg_present"] == True]
            if strong_only and "strong_signal" in signals_df.columns:
                signals_df = signals_df[signals_df["strong_signal"] == True]
            if slow_mom_only and "slow_momentum" in signals_df.columns:
                signals_df = signals_df[signals_df["slow_momentum"] == True]
        
            st.caption(f"After filtering: {len(signals_df)} signals")

            cross_ref_outcome(signals_df)

            if not signals_df.empty:
                st.divider()
                
                target_hits = len(signals_df[signals_df["outcome"] == "target_hit"]) if "outcome" in signals_df.columns else 0
                sl_hits = len(signals_df[signals_df["outcome"] == "sl_hit"]) if "outcome" in signals_df.columns else 0
                win_rate = round(target_hits / (target_hits + sl_hits) * 100, 1) if (target_hits + sl_hits) > 0 else 0
                avg_rr = round(signals_df["rr"].mean(), 2) if "rr" in signals_df.columns else 0

                strong_count = len(signals_df[signals_df["strong_signal"] == True]) if "strong_signal" in signals_df.columns else 0
                fvg_count = len(signals_df[signals_df["fvg_present"] == True]) if "fvg_present" in signals_df.columns else 0
                trend_aligned = len(signals_df[signals_df["factors"].str.contains("trend_aligned", na=False)]) if "factors" in signals_df.columns else 0

                st.markdown(f"""
                <div style="display:grid;grid-template-columns:repeat(7,1fr);gap:10px;margin:8px 0 20px 0;">
                    <div class="stat-card"><div class="stat-label">Signals</div><div class="stat-value cyan">{len(signals_df)}</div></div>
                    <div class="stat-card"><div class="stat-label">Strong ★</div><div class="stat-value green">{strong_count}</div></div>
                    <div class="stat-card"><div class="stat-label">FVG Confirmed</div><div class="stat-value green">{fvg_count}</div></div>
                    <div class="stat-card"><div class="stat-label">Trend Aligned</div><div class="stat-value cyan">{trend_aligned}</div></div>
                    <div class="stat-card"><div class="stat-label">Target Hit</div><div class="stat-value green">{target_hits}</div></div>
                    <div class="stat-card"><div class="stat-label">SL Hit</div><div class="stat-value red">{sl_hits}</div></div>
                    <div class="stat-card"><div class="stat-label">Avg RR</div><div class="stat-value cyan">1:{avg_rr}</div></div>
                </div>
                """, unsafe_allow_html=True)

                st.divider()
                
                capital = general_config.get("trading_capital", 100000)
                risk_pct = general_config.get("risk_per_trade_percent", 1.5)

                col_top, col_download = st.columns([3, 1])
                with col_top:
                    st.markdown('<div class="section-heading"><span class="dot"></span> Top 10 Recommended</div>', unsafe_allow_html=True)
                with col_download:
                    csv = signals_df.to_csv(index=False).encode()
                    st.download_button("Download CSV", csv, f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")

                sort_cols = []
                if "entry_quality" in signals_df.columns:
                    sort_cols.append("entry_quality")
                if "rr" in signals_df.columns:
                    sort_cols.append("rr")
                
                if sort_cols:
                    top_10 = signals_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(10)
                else:
                    top_10 = signals_df.head(10)

                pending_signals = top_10[top_10["outcome"] != "target_hit"]
                pending_signals = pending_signals[pending_signals["outcome"] != "sl_hit"]

                for sig_idx, (idx, row) in enumerate(pending_signals.iterrows()):
                    try:
                        signal_dict = row.to_dict()
                        ticker = signal_dict.get("symbol", signal_dict.get("ticker", "UNKNOWN"))
                        is_ce = "CE" in str(signal_dict.get("type", ""))
                        grade = signal_dict.get("entry_grade", "N/A")
                        signal_time_str = str(signal_dict.get("datetime", sig_idx))

                        grade_color = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴"}.get(grade, "⚪")
                        outcome = signal_dict.get("outcome", "pending")
                        outcome_emoji = {"target_hit": "✅", "sl_hit": "❌", "in_progress": "⏳", "pending": "⏳"}.get(outcome, "⏳")
                        strong_tag = " ★" if signal_dict.get("strong_signal") else ""
                        fvg_tag = " FVG" if signal_dict.get("fvg_present") else ""
                        st_dir = signal_dict.get("market_structure", "")
                        st_emoji = "🟢" if (is_ce and st_dir == "bullish") or (not is_ce and st_dir == "bearish") else "🔴" if st_dir else ""

                        with st.expander(f"{grade_color} **{signal_dict.get('type', 'N/A')}**{strong_tag}{fvg_tag} — {signal_dict.get('ticker', 'N/A')} @ ₹{signal_dict.get('entry', 'N/A')} | Grade {grade} | {outcome_emoji}", expanded=False):
                            col_left, col_right = st.columns([1, 2])

                            with col_left:
                                if signal_dict.get("pattern"):
                                    reliability = signal_dict.get("pattern_reliability", "")
                                    rel_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(reliability, "")
                                    st.info(f"Pattern: {signal_dict.get('pattern', 'N/A')} {rel_emoji}")
                                
                                level_type = "Support" if is_ce else "Resistance"
                                level_value = signal_dict.get("support" if is_ce else "resistance", "N/A")

                                c1, c2 = st.columns(2)
                                with c1:
                                    st.metric("Entry", f"₹{signal_dict.get('entry', 0)}")
                                with c2:
                                    st.metric("Target", f"₹{signal_dict.get('target', 0)}")

                                c3, c4 = st.columns(2)
                                with c3:
                                    st.metric("SL", f"₹{signal_dict.get('sl', 0)}")
                                with c4:
                                    st.metric(f"{level_type}", f"₹{level_value}")

                                c5, c6 = st.columns(2)
                                with c5:
                                    st.metric("RR", f"1:{signal_dict.get('rr', 0)}")
                                
                                st.divider()
                                
                                st.markdown(f"**Trend:** {st_emoji} {st_dir} | **Entry:** {signal_dict.get('zone_entry_reason', 'N/A')}")
                                st.caption(f"Strong Signal: {'Yes ★' if signal_dict.get('strong_signal') else 'No'} | Slow Mom: {'Yes' if signal_dict.get('slow_momentum') else 'No'} | FVG: {'Yes' if signal_dict.get('fvg_present') else 'No'}")
                                st.caption(f"EMA Aligned: {'Yes' if signal_dict.get('ema_aligned') else 'No'}")
                                
                                st.divider()
                                
                                entry_val = signal_dict.get('entry', 100)
                                sl_val = signal_dict.get('sl', 95)
                                pos_size = calculate_position_size(entry_val, sl_val, capital, risk_pct)
                                st.markdown(f"**Position Size:**")
                                st.caption(f"Qty: {pos_size['quantity']} shares | Risk: ₹{pos_size['risk_amount']}")
                                
                                st.divider()
                                
                                strike_step = get_strike_step(ticker)
                                option_strike, strike_type = get_option_strike(
                                    entry_val, strike_step, 
                                    "CE" if is_ce else "PE", "atm"
                                )
                                expiry = get_expiry_recommendation()
                                
                                st.markdown(f"**Option:**")
                                st.caption(f"Strike: {signal_dict.get('index', 'NIFTY')}{int(option_strike)} {strike_type}")
                                st.caption(f"Expiry: {expiry['weekly']} ({expiry['days_to_weekly']}d)")
                                
                                signal_dt = signal_dict.get("datetime")
                                if signal_dt is not None and hasattr(signal_dt, 'strftime'):
                                    st.caption(f"Time: {signal_dt.strftime('%d-%b-%Y %H:%M')}")
                                
                                st.caption(f"**Factors:** {signal_dict.get('factors', 'N/A')}")
                                
                                ticker_name = signal_dict.get('ticker', 'UNKNOWN')
                                if st.button("Add to Watchlist", key=f"watch_{sig_idx}_{signal_time_str}"):
                                    if ticker_name not in st.session_state.watchlist:
                                        st.session_state.watchlist.append(ticker_name)
                                        st.success(f"Added {ticker_name} to watchlist")

                            with col_right:
                                chart_data_5m = ohlcv_data.get(ticker)
                                if chart_data_5m is None:
                                    chart_data_5m = ohlcv_data.get(signal_dict.get("ticker"))
                                
                                daily_data_sess = st.session_state.daily_data
                                weekly_data_sess = st.session_state.weekly_data
                                h1_data_sess = st.session_state.h1_data

                                chart_data_daily = daily_data_sess.get(ticker)
                                if chart_data_daily is None:
                                    chart_data_daily = daily_data_sess.get(signal_dict.get("ticker"))

                                chart_data_weekly = weekly_data_sess.get(ticker)
                                if chart_data_weekly is None:
                                    chart_data_weekly = weekly_data_sess.get(signal_dict.get("ticker"))

                                chart_data_h1 = h1_data_sess.get(ticker)
                                if chart_data_h1 is None:
                                    chart_data_h1 = h1_data_sess.get(signal_dict.get("ticker"))

                                chart_tab_w, chart_tab_d, chart_tab_h1, chart_tab_5 = st.tabs(["1 Week", "1 Day", "1 Hour", "5 Min"])
                                
                                with chart_tab_w:
                                    if chart_data_weekly is not None:
                                        fig_w = create_weekly_chart(chart_data_weekly, signal_dict)
                                        st.plotly_chart(fig_w, use_container_width=True, key=f"weekly_{sig_idx}_{signal_time_str}")
                                    else:
                                        st.warning("Weekly chart data not available")
                                
                                with chart_tab_d:
                                    if chart_data_daily is not None:
                                        fig_d = create_daily_chart(chart_data_daily, signal_dict)
                                        st.plotly_chart(fig_d, use_container_width=True, key=f"daily_{sig_idx}_{signal_time_str}")
                                    else:
                                        st.warning("Daily chart data not available")

                                with chart_tab_h1:
                                    if chart_data_h1 is not None:
                                        fig_h1 = create_h1_chart(chart_data_h1, signal_dict)
                                        st.plotly_chart(fig_h1, use_container_width=True, key=f"h1_{sig_idx}_{signal_time_str}")
                                    else:
                                        st.warning("1 Hour chart data not available")

                                with chart_tab_5:
                                    if chart_data_5m is not None:
                                        fig_5 = create_signal_chart(chart_data_5m, signal_dict, show_chart_bars)
                                        st.plotly_chart(fig_5, use_container_width=True, key=f"5m_{sig_idx}_{signal_time_str}")
                                        
                                        with st.expander("Price Measure Tool"):
                                            m_col1, m_col2 = st.columns(2)
                                            with m_col1:
                                                price1 = st.number_input("Price 1 (₹)", value=float(signal_dict.get('entry', 100)), key=f"p1_{sig_idx}_{signal_time_str}")
                                            with m_col2:
                                                price2 = st.number_input("Price 2 (₹)", value=float(signal_dict.get('target', 110)), key=f"p2_{sig_idx}_{signal_time_str}")
                                            
                                            diff = abs(price2 - price1)
                                            pct = (diff / price1 * 100) if price1 > 0 else 0
                                            st.metric("Difference", f"₹{diff:.2f}", f"{pct:.2f}%")
                                            
                                            st.markdown("**Fibonacci Levels:**")
                                            high_p = max(price1, price2)
                                            low_p = min(price1, price2)
                                            rng = high_p - low_p
                                            fib_cols = st.columns(7)
                                            for i, (lvl, col) in enumerate(zip([0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0], fib_cols)):
                                                fib_price = high_p - (rng * lvl)
                                                col.metric(f"{lvl*100:.1f}%", f"₹{fib_price:.0f}")
                                    else:
                                        st.warning("15-min chart data not available")
                    except Exception as e:
                        st.error(f"Error displaying signal: {e}")
                        continue

elif main_tab == "S/R Levels":
    st.header("Support & Resistance Levels")
    
    sr_ticker_source = st.radio("Select Tickers", ["Indices", "FNO Stocks", "Custom"], horizontal=True)
    
    sr_tickers = []
    if sr_ticker_source == "Indices":
        index_options = {
            "NIFTY 50": "^NSEI",
            "BANK NIFTY": "^NSEBANK",
            "FIN NIFTY": "NIFTY_FIN_SERVICE.NS",
            "MIDCAP SELECT": "NIFTY_MID_SELECT.NS",
            "SENSEX": "^BSESN",
            "GIFT NIFTY": "GIFTNIFTY.NS",
            "INDIA VIX": "^INDIAVIX",
        }
        selected_indices = st.multiselect("Select Indices", list(index_options.keys()), default=["NIFTY 50", "BANK NIFTY"])
        sr_tickers = [index_options[i] for i in selected_indices]
    elif sr_ticker_source == "FNO Stocks":
        try:
            all_fno = load_tickers(TICKERS_FILE)
            select_all_fno = st.checkbox("Select All FNO Stocks", value=False)
            if select_all_fno:
                sr_tickers = all_fno
                st.caption(f"Selected all {len(all_fno)} FNO stocks")
            else:
                popular = ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "TATAMOTORS.NS", "SBIN.NS", "BHARTIARTL.NS"]
                default_popular = [t for t in popular if t in all_fno][:5]
                sr_tickers = st.multiselect("Select Stocks", all_fno, default=default_popular)
                if sr_tickers:
                    st.caption(f"Selected: {len(sr_tickers)} stock(s)")
        except Exception as e:
            st.error(f"Could not load tickers: {e}")
    else:
        custom_sr = st.text_input("Enter tickers (comma separated)", value="RELIANCE.NS, TCS.NS")
        sr_tickers = [t.strip() for t in custom_sr.split(",") if t.strip()]
    
    min_quality_sr = st.slider("Min Level Quality", 0, 100, 50, 5)
    
    refresh_sr = st.button("Load S/R Levels", type="primary")
    
    if refresh_sr:
        st.session_state.sr_data = None
    
    if sr_tickers and (refresh_sr or st.session_state.get("sr_data") is None):
        sr_data = {}
        progress_bar = st.progress(0, text="Loading S/R levels...")
        
        for i, ticker in enumerate(sr_tickers):
            progress_bar.progress((i + 1) / len(sr_tickers), text=f"Loading {ticker}...")
            
            try:
                df_5m = fetch_ohlcv(ticker, interval="5m", period="14d")
                df_daily = fetch_ohlcv(ticker, interval="1d", period="1y")
                df_weekly = fetch_ohlcv(ticker, interval="1wk", period="2y")
                
                if df_5m is not None and len(df_5m) >= MIN_ROWS:
                    from strategies.ta_helpers import atr as _atr
                    
                    current_price = float(df_5m["close"].iloc[-1])
                    prev_close = float(df_5m["close"].iloc[-2]) if len(df_5m) > 1 else current_price
                    change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
                    
                    atr_series = _atr(df_5m["high"], df_5m["low"], df_5m["close"], 14)
                    avg_atr = float(atr_series.iloc[-1]) if atr_series is not None and len(atr_series) > 0 else current_price * 0.01
                    
                    market_structure = detect_market_structure(df_5m)
                    
                    all_resistances, all_supports = get_all_sr_levels(
                        df_5m, df_daily, df_weekly, avg_atr,
                        lookback_weeks=26, min_level_quality=min_quality_sr
                    )
                    
                    resistances = [r for r in all_resistances if float(r["price"]) > current_price][:5]
                    supports = [s for s in all_supports if float(s["price"]) < current_price][:5]
                    
                    sr_data[ticker] = {
                        "ticker": TICKER_LABELS.get(ticker, ticker.replace(".NS", "").replace(".BO", "")),
                        "current_price": round(current_price, 2),
                        "change_pct": round(change_pct, 2),
                        "atr": round(avg_atr, 2),
                        "market_structure": market_structure.value,
                        "resistances": resistances,
                        "supports": supports,
                    }
                else:
                    sr_data[ticker] = {"error": f"Insufficient 5m data ({len(df_5m) if df_5m is not None else 0} bars)"}
            except Exception as e:
                import traceback
                sr_data[ticker] = {"error": f"{str(e)}"}
        
        st.session_state.sr_data = sr_data
        progress_bar.empty()
    
    sr_data = st.session_state.get("sr_data")
    
    if sr_data:
        st.divider()
        
        for ticker, data in sr_data.items():
            if "error" in data:
                st.warning(f"**{ticker}**: {data['error']}")
                continue
            
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                col1.markdown(f"### {data['ticker']}")
                
                change = data['change_pct']
                col2.metric("Price", f"₹{data['current_price']}", f"{change:+.2f}%")
                col3.metric("ATR", f"₹{data['atr']}")
                col4.metric("Structure", data['market_structure'].upper())
                
                r_data = data.get("resistances", [])
                s_data = data.get("supports", [])
                
                col_r, col_s = st.columns(2)
                
                with col_r:
                    st.markdown("**🔴 Resistances (Above)**")
                    if r_data:
                        for r in r_data:
                            dist_pct = (r["price"] - data['current_price']) / data['current_price'] * 100
                            quality = r.get("quality_score", 0)
                            source = r.get("source", "")
                            confirmed = " ✓" if r.get("daily_confirmed") else ""
                            
                            st.markdown(f"₹{r['price']:.2f} `+{dist_pct:.2f}%` | Q:{quality:.0f} | {source}{confirmed}")
                    else:
                        st.info("No resistance levels found")
                
                with col_s:
                    st.markdown("**🟢 Supports (Below)**")
                    if s_data:
                        for s in s_data:
                            dist_pct = (data['current_price'] - s["price"]) / data['current_price'] * 100
                            quality = s.get("quality_score", 0)
                            source = s.get("source", "")
                            confirmed = " ✓" if s.get("daily_confirmed") else ""
                            
                            st.markdown(f"₹{s['price']:.2f} `-{dist_pct:.2f}%` | Q:{quality:.0f} | {source}{confirmed}")
                    else:
                        st.info("No support levels found")
        
        all_levels = []
        for ticker, data in sr_data.items():
            if "error" not in data:
                for r in data.get("resistances", []):
                    all_levels.append({
                        "Ticker": data['ticker'],
                        "Type": "Resistance",
                        "Price": r['price'],
                        "Distance %": round((r['price'] - data['current_price']) / data['current_price'] * 100, 2),
                        "Quality": round(r.get('quality_score', 0), 1),
                        "Source": r.get('source', ''),
                        "Confirmed": r.get('daily_confirmed', False)
                    })
                for s in data.get("supports", []):
                    all_levels.append({
                        "Ticker": data['ticker'],
                        "Type": "Support",
                        "Price": s['price'],
                        "Distance %": round((data['current_price'] - s['price']) / data['current_price'] * 100, 2),
                        "Quality": round(s.get('quality_score', 0), 1),
                        "Source": s.get('source', ''),
                        "Confirmed": s.get('daily_confirmed', False)
                    })
        
        if all_levels:
            levels_df = pd.DataFrame(all_levels)
            levels_df = levels_df.sort_values(["Ticker", "Type", "Quality"], ascending=[True, True, False])
            
            st.divider()
            with st.expander("📋 View All Levels as Table"):
                st.dataframe(levels_df, use_container_width=True, hide_index=True)
                
                csv = levels_df.to_csv(index=False).encode()
                st.download_button("Download S/R Levels CSV", csv, f"sr_levels_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
    
    elif not sr_tickers:
        st.info("Select tickers above and click 'Load S/R Levels'")

elif main_tab == "Backtest":
    st.header("Backtest Results")
    
    if st.session_state.signals_df is not None and not st.session_state.signals_df.empty:
        signals_df = st.session_state.signals_df.copy()
        
        capital = st.sidebar.number_input("Capital", value=float(general_config.get("trading_capital", 100000)))
        risk_pct = st.sidebar.slider("Risk % per Trade", 0.5, 5.0, float(general_config.get("risk_per_trade_percent", 1.5)), 0.5)
        max_positions = st.sidebar.slider("Max Positions/Day", 1, 10, int(general_config.get("max_positions_per_day", 3)))
        
        if st.button("Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                results = run_quick_backtest(signals_df, st.session_state.ohlcv_data)
                st.session_state.backtest_results = results
        
        if st.session_state.backtest_results:
            results = st.session_state.backtest_results
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Trades", results["trades"])
            col2.metric("Wins", results["wins"])
            col3.metric("Losses", results["losses"])
            col4.metric("Win Rate", f"{results['win_rate']}%")
            col5.metric("Total P&L", f"₹{results['total_pnl']}")
            
            st.divider()
            
            if len(results.get("equity_curve", [])) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=results["equity_curve"],
                    mode='lines',
                    name='Equity',
                    line=dict(color='#00d4ff', width=2)
                ))
                fig.update_layout(
                    title="Equity Curve",
                    template="plotly_dark",
                    height=400,
                    showlegend=False,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.subheader("All Trades")
        
        display_cols = ["datetime", "ticker", "type", "entry_grade", "entry", "target", "sl", "rr", "market_structure", "strong_signal", "fvg_present", "slow_momentum", "outcome"]
        display_cols = [c for c in display_cols if c in signals_df.columns]
        st.dataframe(signals_df[display_cols].sort_values("datetime", ascending=False), use_container_width=True, hide_index=True)
    
    else:
        st.info("Run a scan first to see backtest results")

elif main_tab == "Dashboard":
    if "selected_trade" not in st.session_state:
        st.session_state.selected_trade = None

    j = st.session_state.trade_journal
    open_trades = [t for t in j if t["result"] == "Pending"]

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#141729 0%,#1a1e35 100%);border:1px solid #1e2a45;border-radius:14px;padding:20px 24px;margin-bottom:20px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <h3 style="margin:0;font-size:1.1rem;font-weight:800;color:#e2e8f0;">🔵 ACTIVE TRADES ({len(open_trades)})</h3>
        <span style="color:#64748b;font-size:0.78rem;">{datetime.now(IST).strftime("%b %d, %Y %H:%M")} IST</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if open_trades:
        for i, t in enumerate(open_trades):
            c1, c2, c3, c4, c5, c6 = st.columns([2, 0.7, 1.3, 1.3, 1.3, 0.8])
            c1.markdown(f'<span style="color:#00d4ff;font-weight:700;font-size:0.85rem;">{t["ticker"]}</span>', unsafe_allow_html=True)
            cls = "ce" if "CE" in t["type"] else "pe"
            bg = "rgba(16,185,129,0.15)" if cls == "ce" else "rgba(239,68,68,0.15)"
            clr = "#10b981" if cls == "ce" else "#ef4444"
            c2.markdown(f'<span style="font-size:0.7rem;font-weight:700;padding:2px 8px;border-radius:4px;background:{bg};color:{clr};">{t["type"][:2]}</span>', unsafe_allow_html=True)
            c3.markdown(f'<span style="color:#94a3b8;font-size:0.82rem;">₹{t["entry"]:.1f}</span>', unsafe_allow_html=True)
            c4.markdown(f'<span style="color:#10b981;font-size:0.82rem;">₹{t["target"]:.1f}</span>', unsafe_allow_html=True)
            c5.markdown(f'<span style="color:#ef4444;font-size:0.82rem;">₹{t["sl"]:.1f}</span>', unsafe_allow_html=True)
            if c6.button("📊", key=f"dash_view_{i}"):
                st.session_state.selected_trade = t
    else:
        st.info("No active trades. Scan results will appear here automatically.")

    capital = general_config.get("trading_capital", 100000)
    risk_pct = general_config.get("risk_per_trade_percent", 1.5)

    sel = st.session_state.selected_trade
    if sel:
        st.divider()
        col_h, col_c = st.columns([3, 1])
        with col_h:
            st.subheader(f"📈 {sel['ticker']} — {sel['type']} @ ₹{sel['entry']}")
        with col_c:
            if st.button("✕ Close", use_container_width=True):
                st.session_state.selected_trade = None
                st.rerun()

        is_ce = "CE" in sel["type"]
        level_type = "Support" if is_ce else "Resistance"

        col_details, col_charts = st.columns([1, 2])
        with col_details:
            st.markdown(f'<div style="background:linear-gradient(135deg,#141729 0%,#1a1e35 100%);border:1px solid #1e2a45;border-radius:14px;padding:16px;">', unsafe_allow_html=True)

            pattern = sel.get("pattern", "")
            if pattern:
                st.info(f"Pattern: {pattern}")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Entry", f"₹{sel['entry']:.1f}")
            with c2:
                st.metric("Target", f"₹{sel['target']:.1f}")
            c3, c4 = st.columns(2)
            with c3:
                st.metric("SL", f"₹{sel['sl']:.1f}")
            with c4:
                st.metric(level_type, f"₹{sel.get('support' if is_ce else 'resistance', sel['entry']):.1f}")
            c5, c6 = st.columns(2)
            with c5:
                rr = sel.get("rr", 0)
                st.metric("RR", f"1:{rr}" if rr else "—")
            with c6:
                grade = sel.get("grade", "")
                st.metric("Grade", grade if grade else "—")

            st.divider()

            factors = sel.get("factors", "")
            st.markdown(f"**Factors:** {factors}" if factors else "")
            entry_time = sel.get("entry_time", "")
            if entry_time:
                st.caption(f"Time: {entry_time[:19]}")
            stage = sel.get("stage", "")
            if stage:
                st.caption(f"Stage: {stage}")
            score = sel.get("score", 0)
            if score:
                st.caption(f"Score: {score}")

            st.divider()

            entry_val = sel["entry"]
            sl_val = sel["sl"]
            pos_size = calculate_position_size(entry_val, sl_val, capital, risk_pct)
            st.markdown(f"**Position Size:**")
            st.caption(f"Qty: {pos_size['quantity']} shares | Risk: ₹{pos_size['risk_amount']}")

            strike = sel.get("strike", 0)
            if strike:
                st.markdown(f"**Option:**")
                st.caption(f"Strike: {int(strike)}")
                expiry = get_expiry_recommendation()
                st.caption(f"Expiry: {expiry['weekly']} ({expiry['days_to_weekly']}d)")

            st.markdown('</div>', unsafe_allow_html=True)

        with col_charts:
            ticker_symbol = sel["ticker"]
            if not ticker_symbol.endswith((".NS", ".BO")) and not ticker_symbol.startswith("^"):
                ticker_symbol = ticker_symbol + ".NS"

            with st.spinner("Loading chart data..."):
                df_5m = fetch_ohlcv(ticker_symbol, interval="5m", period="14d")
                df_h1 = fetch_ohlcv(ticker_symbol, interval="1h", period="21d")
                df_daily = fetch_ohlcv(ticker_symbol, interval="1d", period="1y")
                df_weekly = fetch_ohlcv(ticker_symbol, interval="1wk", period="5y")

            signal_dict = {
                "ticker": sel["ticker"],
                "type": sel["type"],
                "entry": sel["entry"],
                "target": sel["target"],
                "sl": sel["sl"],
                "support": sel.get("support", sel["entry"]),
                "resistance": sel.get("resistance", sel["entry"]),
                "datetime": sel.get("date", sel.get("entry_time", "")),
            }

            chart_tab_w, chart_tab_d, chart_tab_h1, chart_tab_5 = st.tabs(["1 Week", "1 Day", "1 Hour", "5 Min"])
            with chart_tab_w:
                if df_weekly is not None and len(df_weekly) > 5:
                    fig_w = create_weekly_chart(df_weekly, signal_dict)
                    st.plotly_chart(fig_w, use_container_width=True, key="dash_weekly")
                else:
                    st.warning("Weekly data not available")
            with chart_tab_d:
                if df_daily is not None and len(df_daily) > 5:
                    fig_d = create_daily_chart(df_daily, signal_dict)
                    st.plotly_chart(fig_d, use_container_width=True, key="dash_daily")
                else:
                    st.warning("Daily data not available")
            with chart_tab_h1:
                if df_h1 is not None and len(df_h1) > 5:
                    fig_h1 = create_h1_chart(df_h1, signal_dict)
                    st.plotly_chart(fig_h1, use_container_width=True, key="dash_h1")
                else:
                    st.warning("1 Hour data not available")
            with chart_tab_5:
                if df_5m is not None and len(df_5m) > 20:
                    fig_5 = create_signal_chart(df_5m, signal_dict, 100)
                    st.plotly_chart(fig_5, use_container_width=True, key="dash_5m")
                else:
                    st.warning("5 Min data not available")

elif main_tab == "Journal":
    j = st.session_state.trade_journal
    completed_list = [t for t in j if t["result"] != "Pending"]
    wins = [t for t in completed_list if t["result"] == "Target Hit"]
    losses = [t for t in completed_list if t["result"] == "SL Hit"]
    winrate = len(wins) / len(completed_list) * 100 if completed_list else 0
    avg_pnl = sum(t.get("pnl") or 0 for t in completed_list) / len(completed_list) if completed_list else 0

    st.markdown("""
    <style>
    .journal-card { background: linear-gradient(135deg, #141729 0%, #1a1e35 100%); border: 1px solid #1e2a45; border-radius: 14px; padding: 24px; margin-bottom: 20px; }
    .journal-card h3 { color: #e2e8f0; font-size: 1rem; font-weight: 700; margin: 0 0 16px 0; display: flex; align-items: center; gap: 8px; }
    .jt-row { display: flex; align-items: center; gap: 12px; padding: 10px 14px; border: 1px solid #1e2a45; border-radius: 10px; margin-bottom: 6px; transition: border-color 0.2s; background: rgba(13,15,26,0.4); }
    .jt-row:hover { border-color: #00d4ff44; }
    .jt-ticker { color: #00d4ff; font-weight: 700; font-size: 0.85rem; min-width: 100px; }
    .jt-type { font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 4px; min-width: 50px; text-align: center; }
    .jt-type.ce { background: rgba(16,185,129,0.15); color: #10b981; }
    .jt-type.pe { background: rgba(239,68,68,0.15); color: #ef4444; }
    .jt-result { font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 12px; text-align: center; }
    .jt-result.win { background: rgba(16,185,129,0.15); color: #10b981; }
    .jt-result.loss { background: rgba(239,68,68,0.12); color: #ef4444; }
    .jt-result.pending { background: rgba(245,158,11,0.12); color: #f59e0b; }
    .jt-result.manual { background: rgba(99,102,241,0.12); color: #818cf8; }
    .jt-pnl { font-size: 0.82rem; font-weight: 700; text-align: right; }
    .jt-pnl.pos { color: #10b981; }
    .jt-pnl.neg { color: #ef4444; }
    </style>
    """, unsafe_allow_html=True)

    # ── Completed Trades Summary ──
    wr_color = "#10b981" if winrate >= 50 else "#ef4444"
    pnl_color = "#10b981" if avg_pnl >= 0 else "#ef4444"
    st.markdown(f"""
    <div class="journal-card" style="padding:20px 24px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
        <h3 style="margin:0;font-size:1.1rem;font-weight:800;color:#e2e8f0;">📊 COMPLETED TRADES SUMMARY</h3>
        <span style="color:#64748b;font-size:0.78rem;">{datetime.now(IST).strftime("%b %d, %Y %H:%M")} IST</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px 24px;">
        <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1e2a45;">
          <span style="color:#8892b0;">Target Hit ✅</span>
          <span style="color:#10b981;font-weight:700;">{len(wins)} ({winrate:.0f}%)</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1e2a45;">
          <span style="color:#8892b0;">SL Hit ❌</span>
          <span style="color:#ef4444;font-weight:700;">{len(losses)}</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1e2a45;">
          <span style="color:#8892b0;">Avg PnL</span>
          <span style="color:{pnl_color};font-weight:700;">{avg_pnl:+.1f}%</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Completed Trades List ──
    if completed_list:
        st.markdown(f'<div class="journal-card"><h3>✅ COMPLETED TRADES ({len(completed_list)})</h3>', unsafe_allow_html=True)
        sorted_c = sorted(completed_list, key=lambda x: x.get("date", ""), reverse=True)
        for t in sorted_c[:50]:
            cls_type = "ce" if "CE" in t["type"] else "pe"
            cls_res = "win" if t["result"] == "Target Hit" else "loss"
            pnl = t.get("pnl")
            pnl_str = f'{pnl:+.1f}%' if pnl is not None else "—"
            pnl_cls = "pos" if pnl and pnl > 0 else "neg" if pnl and pnl < 0 else ""
            exit_price = t["target"] if t["result"] == "Target Hit" else t["sl"]
            st.markdown(
                f'<div class="jt-row">'
                f'<span class="jt-ticker">{t["ticker"]}</span>'
                f'<span class="jt-type {cls_type}">{t["type"][:2]}</span>'
                f'<span style="color:#94a3b8;font-size:0.82rem;">₹{t["entry"]:.1f}→₹{exit_price:.1f}</span>'
                f'<span style="color:#64748b;font-size:0.72rem;">{t.get("date","")[:10]}</span>'
                f'<span class="jt-result {cls_res}">{t["result"]}</span>'
                f'<span class="jt-pnl {pnl_cls}">{pnl_str}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Log New Trade ──
    st.markdown('<div class="journal-card"><h3>📝 LOG NEW TRADE</h3>', unsafe_allow_html=True)
    with st.form("journal_form", clear_on_submit=True):
        _all_tickers = sorted(set(
            list(pd.read_csv("indices_tikcers.csv")["ticker"]) +
            list(pd.read_csv("fno_tickers.csv")["ticker"])
        ))
        r1a, r1b = st.columns([1, 1])
        with r1a:
            jt_ticker = st.selectbox("Ticker", _all_tickers, index=None, placeholder="Select ticker...")
        with r1b:
            jt_type = st.selectbox("Type", ["CE BUY", "PE BUY"])
        r2a, r2b, r2c = st.columns([1, 1, 1])
        with r2a:
            jt_entry = st.number_input("Entry", value=None, format="%.2f", placeholder="Entry")
        with r2b:
            jt_target = st.number_input("Target", value=None, format="%.2f", placeholder="Target")
        with r2c:
            jt_sl = st.number_input("Stop Loss", value=None, format="%.2f", placeholder="SL")
        r3a, r3b = st.columns([1, 2])
        with r3a:
            jt_result = st.selectbox("Result", ["Pending", "Target Hit", "SL Hit"])
        with r3b:
            jt_notes = st.text_area("Notes", placeholder="e.g. pattern, trade rationale...", height=60)
        if st.form_submit_button("Save Trade", use_container_width=True, type="primary"):
            ticker_name = jt_ticker.upper()
            is_ce = "CE" in jt_type
            entry_val = jt_entry or 0
            target_val = jt_target or 0
            sl_val = jt_sl or 0
            pnl = None
            if jt_result == "Target Hit" and entry_val:
                pnl = round((target_val - entry_val) / entry_val * 100, 2)
            elif jt_result == "SL Hit" and entry_val:
                pnl = round((entry_val - sl_val) / entry_val * -100, 2)

            trade = {
                "date": datetime.now(IST).strftime("%Y-%m-%d %H:%M"),
                "ticker": ticker_name,
                "type": jt_type,
                "entry": entry_val,
                "target": target_val,
                "sl": sl_val,
                "qty": 0,
                "notes": jt_notes or "",
                "result": jt_result,
                "pnl": pnl,
            }
            j.append(trade)
            import json as _json, os as _os
            _jf = _os.path.join(_os.path.dirname(__file__), "trade_journal.json")
            try:
                existing = []
                if _os.path.exists(_jf):
                    with open(_jf) as _f:
                        existing = _json.load(_f)
                existing.append({
                    "ticker": ticker_name, "trade_type": "CE" if is_ce else "PE",
                    "entry_date": datetime.now(IST).strftime("%Y-%m-%d"), "entry_price": entry_val,
                    "target": target_val, "stop_loss": sl_val, "pnl_pct": pnl,
                    "status": "hit_target" if jt_result == "Target Hit" else "hit_sl" if jt_result == "SL Hit" else "open",
                })
                with open(_jf, "w") as _f:
                    _json.dump(existing, _f, indent=2)
            except Exception:
                pass
            st.success(f"Trade saved as {selected_result}!")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Trade History ──
    st.markdown(f'<div class="journal-card"><h3>📋 TRADE HISTORY ({len(j)} total)</h3>', unsafe_allow_html=True)
    if j:
        sorted_j = sorted(j, key=lambda x: x.get("date", ""), reverse=True)
        for t in sorted_j[:50]:
            cls_type = "ce" if "CE" in t["type"] else "pe"
            cls_res = "win" if t["result"] == "Target Hit" else "loss" if t["result"] == "SL Hit" else "manual" if t["result"] == "Manual Exit" else "pending"
            pnl = t.get("pnl")
            pnl_str = f'{pnl:+.1f}%' if pnl is not None else "—"
            pnl_cls = "pos" if pnl and pnl > 0 else "neg" if pnl and pnl < 0 else ""
            pat = t.get("pattern", "")
            pat_tag = f'<span style="color:#a78bfa;font-size:0.7rem;margin-left:6px;">{pat}</span>' if pat else ""
            st.markdown(
                f'<div class="jt-row">'
                f'<span class="jt-ticker">{t["ticker"]}{pat_tag}</span>'
                f'<span class="jt-type {cls_type}">{t["type"][:2]}</span>'
                f'<span style="color:#94a3b8;font-size:0.82rem;">{t["entry"]:.1f}</span>'
                f'<span style="color:#64748b;font-size:0.72rem;">{t.get("date","")[:10]}</span>'
                f'<span class="jt-result {cls_res}">{t["result"]}</span>'
                f'<span class="jt-pnl {pnl_cls}">{pnl_str}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        if len(j) > 50:
            st.caption(f"Showing last 50 of {len(j)} trades")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Import from JSON ──
    with st.expander("Reload from JSON File"):
        col_imp1, col_imp2 = st.columns([2, 1])
        with col_imp1:
            st.markdown("Trades are auto-recorded on scan. Reload here if you edited `trade_journal.json` externally.")
        with col_imp2:
            if st.button("Reload from JSON", use_container_width=True):
                _refresh_journal_state()
                st.success("Journal reloaded from JSON!")
                st.rerun()


st.divider()

with st.expander("Strategy Rules — 3-Step A+ Supply & Demand"):
    st.markdown("""
    ### 3-STEP SIGNAL FLOW (Trade with Pat)

    | Step | Name | What It Does |
    |------|------|-------------|
    | 1 | Institutional Zones | Detect explosive impulsive moves (3+ big candles), draw zone from consolidation BEFORE impulse, check FVG for momentum confirmation |
    | 2 | Break of Structure | Mark swing highs/lows, identify BOS for trend direction — only buy demand in uptrend, sell supply in downtrend |
    | 3 | Entry Conditions | Wait for slow momentum approaching zone, candle must close IN or wick INTO zone, enter on confirmation candle, SL tight to zone, TP at 1:1 to 1.5 R:R |

    ### 6 KEYS TO VALID ZONES
    1. **Fresh zone** — not already used/retested
    2. **Close in or wick into zone** — close through = invalid
    3. **Confluence stack** — EMA as support/resistance
    4. **Slow momentum** — small candle bodies approaching zone
    5. **Higher timeframe alignment** — weekly/daily trend
    6. **Institutional footprint** — big volume on impulse

    ### GRADING
    - **Grade A (75%+):** Strong conviction — trend + entry + slow mom + FVG + strong zone
    - **Grade B (55-74%):** Good setup — 3+ factors aligned
    - **Grade C/D (<55%):** Filtered out — not traded

    ### SCORING BREAKDOWN
    | Factor | Points |
    |--------|--------|
    | Trend aligned (BOS) | +25 |
    | Entry confirmed (close/wick in zone) | +20 |
    | Slow momentum approaching zone | +15 |
    | FVG (Fair Value Gap) present | +15 |
    | Strong zone (strength >= 70) | +10 |
    | Volume spike on impulse | +10 |
    | Regime aligned (trending market) | +5 |
    """)

st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.78rem;margin-top:20px;padding:12px;border-top:1px solid #1e2340;">
    For educational purposes only | Not financial advice | 3-Step A+ Supply & Demand Strategy (Trade with Pat)
</div>
""", unsafe_allow_html=True)
