"""
Show Support & Resistance Levels for Today
Run: python show_sr_levels.py
"""

import sys
from datetime import datetime

import pandas as pd

from scanner_base import fetch_ohlcv, ticker_label, INDEX_TICKERS
from strategies.sr_levels import get_all_sr_levels, detect_market_structure
from strategies.supply_demand_v2 import Config
from strategies.ta_helpers import atr as _atr


def get_sr_for_ticker(ticker: str) -> dict:
    """Get S/R levels for a single ticker"""
    
    df_15m = fetch_ohlcv(ticker, interval="15m", period="30d", min_rows=50)
    df_daily = fetch_ohlcv(ticker, interval="1d", period="1y", min_rows=100)
    df_weekly = fetch_ohlcv(ticker, interval="1wk", period="2y", min_rows=50)
    
    if df_15m is None or len(df_15m) < 50:
        return {"error": f"Insufficient 15m data ({len(df_15m) if df_15m is not None else 0} bars)"}
    
    current_price = df_15m["close"].iloc[-1]
    prev_close = df_15m["close"].iloc[-2] if len(df_15m) > 1 else current_price
    change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
    
    atr = _atr(df_15m["high"], df_15m["low"], df_15m["close"], 14)
    avg_atr = atr.iloc[-1] if atr is not None and not atr.empty else current_price * 0.01
    
    market_structure = detect_market_structure(df_15m)
    
    all_resistances, all_supports = get_all_sr_levels(
        df_15m, df_daily, df_weekly, avg_atr,
        lookback_weeks=26, min_level_quality=50
    )
    
    resistances = [r for r in all_resistances if r["price"] > current_price][:5]
    supports = [s for s in all_supports if s["price"] < current_price][:5]
    
    return {
        "ticker": ticker_label(ticker),
        "current_price": round(current_price, 2),
        "change_pct": round(change_pct, 2),
        "atr": round(avg_atr, 2),
        "market_structure": market_structure.value,
        "resistances": resistances,
        "supports": supports,
    }


def print_sr_table(data: dict):
    """Print S/R levels in table format"""
    
    if "error" in data:
        print(f"\n  {data['error']}")
        return
    
    ticker = data["ticker"]
    price = data["current_price"]
    change = data["change_pct"]
    structure = data["market_structure"]
    atr = data["atr"]
    
    change_color = "\033[92m" if change >= 0 else "\033[91m"
    reset = "\033[0m"
    
    print(f"\n{'='*70}")
    print(f"  {ticker}  |  Price: ₹{price}  |  {change_color}{change:+.2f}%{reset}  |  ATR: {atr}  |  {structure.upper()}")
    print(f"{'='*70}")
    
    print(f"\n  {'RESISTANCES':^32}  |  {'SUPPORTS':^32}")
    print(f"  {'-'*32}  |  {'-'*32}")
    
    resistances = data.get("resistances", [])
    supports = data.get("supports", [])
    
    max_rows = max(len(resistances), len(supports))
    
    for i in range(max_rows):
        r_str = ""
        s_str = ""
        
        if i < len(resistances):
            r = resistances[i]
            dist_pct = ((r["price"] - price) / price * 100)
            r_str = f"  ₹{r['price']:>8.2f}  (+{dist_pct:.2f}%)  Q:{r['quality_score']:>3.0f}  [{r.get('source','')}]"
            if r.get("daily_confirmed"):
                r_str += " ✓"
        else:
            r_str = "  " + " " * 30
        
        if i < len(supports):
            s = supports[i]
            dist_pct = ((price - s["price"]) / price * 100)
            s_str = f"  ₹{s['price']:>8.2f}  (-{dist_pct:.2f}%)  Q:{s['quality_score']:>3.0f}  [{s.get('source','')}]"
            if s.get("daily_confirmed"):
                s_str += " ✓"
        else:
            s_str = "  " + " " * 30
        
        print(f"{r_str}  |{s_str}")
    
    print(f"\n  Legend: Q = Quality Score (0-100)  |  ✓ = Daily Confirmed  |  Source: weekly/15m")
    print(f"{'='*70}")


def main():
    tickers = []
    
    if len(sys.argv) > 1:
        tickers = sys.argv[1:]
    else:
        tickers = ["^NSEI", "^NSEBANK", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS"]
    
    print(f"\n{'#'*70}")
    print(f"  SUPPORT & RESISTANCE LEVELS  |  {datetime.now().strftime('%d-%b-%Y %H:%M')}")
    print(f"{'#'*70}")
    
    for ticker in tickers:
        data = get_sr_for_ticker(ticker)
        print_sr_table(data)
    
    print(f"\n  Usage: python show_sr_levels.py [TICKER1] [TICKER2] ...")
    print(f"  Example: python show_sr_levels.py ^NSEI RELIANCE.NS TATAMOTORS.NS\n")


if __name__ == "__main__":
    main()
