"""
Trade Journal — Tracks trade alert outcomes automatically.

Usage:
    python3 trade_journal.py status          # Show open/pending trades & winrate
    python3 trade_journal.py history          # Show all completed trades
    python3 trade_journal.py reset            # Clear journal (start fresh)

Integration with scan:
    python3 scan_today_trades.py --journal   # Auto-record AND check previous trades
"""

import json
import os
import sys
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict

JOURNAL_FILE = os.path.join(os.path.dirname(__file__), "trade_journal.json")


@dataclass
class TradeRecord:
    ticker: str
    trade_type: str       # "CE" or "PE"
    entry_date: str       # YYYY-MM-DD
    entry_time: str       # HH:MM
    entry_price: float
    target: float
    stop_loss: float
    strike: int
    rr: float
    entry_grade: str
    score: int
    factors: str
    pattern: Optional[str]
    stage: Optional[str]
    confidence: Optional[float]
    vwap: Optional[float]
    status: str = "open"  # open, hit_target, hit_sl, expired
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None


def load_journal() -> List[dict]:
    if not os.path.exists(JOURNAL_FILE):
        return []
    with open(JOURNAL_FILE) as f:
        return json.load(f)


def save_journal(records: List[dict]):
    os.makedirs(os.path.dirname(JOURNAL_FILE) or ".", exist_ok=True)
    with open(JOURNAL_FILE, "w") as f:
        json.dump(records, f, indent=2)


def add_trade(alert: dict):
    records = load_journal()
    # dedup by ticker + type + entry_time
    key = (alert["ticker"], alert["type"], str(alert["datetime"]))
    for r in records:
        if (r.get("ticker", ""), r.get("trade_type", ""), r.get("entry_time", "")) == key:
            return  # already recorded
    records.append({
        "ticker": alert["ticker"],
        "trade_type": "CE" if "CE" in alert.get("type", "") else "PE",
        "entry_date": str(alert.get("_date", date.today())),
        "entry_time": str(alert.get("datetime", "")),
        "entry_price": alert.get("entry", 0),
        "target": alert.get("target", 0),
        "stop_loss": alert.get("sl", 0),
        "strike": alert.get("strike", 0),
        "rr": alert.get("rr", 0),
        "entry_grade": alert.get("entry_grade", ""),
        "score": alert.get("entry_quality", 0),
        "factors": alert.get("factors", ""),
        "pattern": alert.get("pattern"),
        "stage": alert.get("pattern_stage"),
        "confidence": alert.get("pattern_confidence"),
        "vwap": alert.get("vwap"),
        "status": "open",
        "exit_date": None,
        "exit_price": None,
        "pnl_pct": None,
    })
    save_journal(records)


def check_open_trades():
    """Check all open trades against current price to see if target/SL hit."""
    from scanner_base import fetch_ohlcv, now_ist, ticker_label

    records = load_journal()
    if not records:
        return records

    open_trades = [r for r in records if r["status"] == "open"]
    if not open_trades:
        print("  ✓ No open trades to check.\n")
        return records

    updated = 0
    for trade in open_trades:
        raw_ticker = trade["ticker"]
        try:
            df = fetch_ohlcv(raw_ticker, interval="15m", period="5d")
        except Exception:
            # Try without .NS suffix if needed
            raw_ticker = raw_ticker.replace(".NS", "") + ".NS"
            try:
                df = fetch_ohlcv(raw_ticker, interval="15m", period="5d")
            except Exception:
                continue

        if df is None or df.empty:
            continue

        entry_date = trade["entry_date"]
        is_ce = trade["trade_type"] == "CE"
        target = trade["target"]
        sl = trade["stop_loss"]
        entry_price = trade["entry_price"]

        # Check if price reached target or SL on or after entry date
        entry_dt = datetime.strptime(entry_date, "%Y-%m-%d").date()
        bars_after_entry = df[df.index.date >= entry_dt]

        for idx in bars_after_entry.index:
            bar = bars_after_entry.loc[idx]
            high = bar["high"]
            low = bar["low"]

            if is_ce:
                if high >= target:
                    trade["status"] = "hit_target"
                    trade["exit_date"] = str(idx.date())
                    trade["exit_price"] = round(target, 2)
                    trade["pnl_pct"] = round((target - entry_price) / entry_price * 100, 2)
                    updated += 1
                    break
                if low <= sl:
                    trade["status"] = "hit_sl"
                    trade["exit_date"] = str(idx.date())
                    trade["exit_price"] = round(sl, 2)
                    trade["pnl_pct"] = round((sl - entry_price) / entry_price * 100, 2)
                    updated += 1
                    break
            else:
                if low <= target:
                    trade["status"] = "hit_target"
                    trade["exit_date"] = str(idx.date())
                    trade["exit_price"] = round(target, 2)
                    trade["pnl_pct"] = round((entry_price - target) / entry_price * 100, 2)
                    updated += 1
                    break
                if high >= sl:
                    trade["status"] = "hit_sl"
                    trade["exit_date"] = str(idx.date())
                    trade["exit_price"] = round(sl, 2)
                    trade["pnl_pct"] = round((entry_price - sl) / entry_price * 100, 2)
                    updated += 1
                    break

        # If entry was 3+ days ago and still open, mark expired
        entry_dt_obj = datetime.strptime(entry_date, "%Y-%m-%d")
        if trade["status"] == "open" and (datetime.now() - entry_dt_obj).days >= 3:
            trade["status"] = "expired"
            trade["exit_date"] = str(date.today())
            trade["exit_price"] = 0
            trade["pnl_pct"] = 0
            updated += 1

    if updated:
        save_journal(records)
        print(f"  ✓ {updated} trade(s) resolved.\n")
    else:
        print("  No new outcomes yet — trades still open.\n")

    return records


def print_status():
    records = load_journal()
    if not records:
        print("\n  No trades recorded yet. Run scan with --journal to start tracking.\n")
        return

    total = len(records)
    open_trades = [r for r in records if r["status"] == "open"]
    completed = [r for r in records if r["status"] != "open"]
    wins = [r for r in completed if r["status"] == "hit_target"]
    losses = [r for r in completed if r["status"] == "hit_sl"]
    expired = [r for r in completed if r["status"] == "expired"]

    winrate = len(wins) / len(completed) * 100 if completed else 0
    avg_pnl = sum(r["pnl_pct"] or 0 for r in completed) / len(completed) if completed else 0

    print(f"\n{'='*60}")
    print(f"  TRADE JOURNAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total Trades   : {total}")
    print(f"  Open           : {len(open_trades)}")
    print(f"  Completed      : {len(completed)}")
    print(f"  ✅ Wins        : {len(wins)}  ({winrate:.0f}%)")
    print(f"  ❌ Losses      : {len(losses)}")
    print(f"  ⏳ Expired     : {len(expired)}")
    print(f"  Avg PnL        : {avg_pnl:+.2f}%")
    print(f"{'='*60}\n")

    if open_trades:
        print(f"  OPEN TRADES:")
        for t in open_trades:
            td = t["trade_type"]
            print(f"    {t['ticker']:20s} {td:3s} entry={t['entry_price']:.1f} "
                  f"target={t['target']:.1f} SL={t['stop_loss']:.1f} "
                  f"grade={t['entry_grade']} score={t['score']}")

    if completed:
        print(f"\n  LAST 10 COMPLETED:")
        for t in completed[-10:]:
            icon = "✅" if t["status"] == "hit_target" else "❌" if t["status"] == "hit_sl" else "⏳"
            print(f"    {icon} {t['ticker']:20s} {t['trade_type']:3s} "
                  f"entry={t['entry_price']:.1f} exit={t['exit_price'] or 0:.1f} "
                  f"pnl={t['pnl_pct']:+.1f}%")


def print_history():
    records = load_journal()
    if not records:
        print("  No trades recorded.\n")
        return
    completed = [r for r in records if r["status"] != "open"]
    if not completed:
        print("  No completed trades yet.\n")
        return

    print(f"\n{'='*80}")
    print(f"  TRADE HISTORY ({len(completed)} completed)")
    print(f"{'='*80}")
    for t in completed:
        icon = "✅" if t["status"] == "hit_target" else "❌" if t["status"] == "hit_sl" else "⏳"
        print(f"  {icon} {t['ticker']:20s} {t['trade_type']:3s} "
              f"entry={t['entry_price']:.1f} exit={t['exit_price'] or 0:.1f} "
              f"pnl={t['pnl_pct']:+.1f}%  grade={t['entry_grade']}  {t['entry_date']}")


def record_scan_signals(signals_df) -> int:
    """Auto-record all signals from a scanner DF into the journal. Returns count added."""
    if signals_df is None or signals_df.empty:
        return 0
    records = load_journal()
    existing = set()
    for r in records:
        existing.add((r.get("ticker", ""), r.get("trade_type", ""), r.get("entry_time", "")))
    added = 0
    for _, row in signals_df.iterrows():
        trade_type = "CE" if "CE" in str(row.get("type", "")) else "PE"
        entry_time = str(row.get("datetime", ""))
        ticker = row.get("symbol", row.get("ticker", ""))
        k = (ticker, trade_type, entry_time)
        if k in existing:
            continue
        entry_date = entry_time[:10] if entry_time and len(entry_time) >= 10 else str(row.get("_date", date.today()))
        records.append({
            "ticker": ticker,
            "trade_type": trade_type,
            "entry_date": entry_date,
            "entry_time": entry_time,
            "entry_price": row.get("entry", 0),
            "target": row.get("target", 0),
            "stop_loss": row.get("sl", 0),
            "strike": int(row.get("strike", 0)) if row.get("strike") else 0,
            "rr": row.get("rr", 0),
            "entry_grade": row.get("entry_grade", ""),
            "score": row.get("entry_quality", 0),
            "factors": row.get("factors", ""),
            "pattern": str(row.get("pattern", "")) or None,
            "stage": str(row.get("pattern_stage", "")) or None,
            "confidence": float(row.get("pattern_confidence", 0)) or None,
            "vwap": float(row.get("vwap", 0)) or None,
            "status": "open",
            "exit_date": None,
            "exit_price": None,
            "pnl_pct": None,
        })
        existing.add(k)
        added += 1
    if added:
        save_journal(records)
    return added


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 trade_journal.py [status|history|reset]")
        return

    cmd = sys.argv[1]

    if cmd == "status":
        check_open_trades()
        print_status()

    elif cmd == "history":
        check_open_trades()
        print_history()

    elif cmd == "reset":
        if os.path.exists(JOURNAL_FILE):
            os.remove(JOURNAL_FILE)
            print("  ✓ Journal cleared.\n")
        else:
            print("  No journal found.\n")

    else:
        print(f"  Unknown command: {cmd}")
        print("  Use: status | history | reset")


if __name__ == "__main__":
    main()
