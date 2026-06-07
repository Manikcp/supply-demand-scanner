"""
Telegram Notifier — sends trade alerts to a Telegram channel.

Config via environment variables (set in GitHub Actions secrets):
    TELEGRAM_BOT_TOKEN  — bot token from @BotFather
    TELEGRAM_CHAT_ID    — channel / group chat ID
"""

import os
import json
import requests
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

SENT_FILE = os.path.join(os.path.dirname(__file__), "sent_trades.json")

_ST_SECRETS = None
try:
    import streamlit as st
    _ST_SECRETS = st.secrets
except Exception:
    pass


def _get_telegram_config() -> tuple:
    if _ST_SECRETS is not None:
        try:
            token = _ST_SECRETS.get("telegram", {}).get("bot_token", "")
            chat_id = _ST_SECRETS.get("telegram", {}).get("chat_id", "")
            if token and chat_id:
                return token, chat_id
        except Exception:
            pass
    return os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", "")


def _load_sent() -> list:
    if not os.path.exists(SENT_FILE):
        return []
    try:
        with open(SENT_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def _save_sent(sent: list):
    with open(SENT_FILE, "w") as f:
        json.dump(sent, f, indent=2)


def _trade_key(trade: dict) -> str:
    return f"{trade.get('ticker','')}|{trade.get('trade_type','')}|{trade.get('entry_date','')}|{trade.get('entry_time','')}"


def already_sent(trade: dict) -> bool:
    key = _trade_key(trade)
    return key in _load_sent()


def mark_sent(trade: dict):
    sent = _load_sent()
    key = _trade_key(trade)
    if key not in sent:
        sent.append(key)
    _save_sent(sent)


def send_message(text: str) -> bool:
    token, chat_id = _get_telegram_config()
    if not token or not chat_id:
        print("[Telegram] BOT_TOKEN or CHAT_ID not set")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        }, timeout=15)
        if r.status_code == 200:
            return True
        print(f"[Telegram] API error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"[Telegram] send failed: {e}")
    return False


def format_trade_alert(trade: dict) -> str:
    tt = "🟢 CE BUY" if trade.get("trade_type") == "CE" else "🔴 PE BUY"
    ticker = trade.get("ticker", "?")
    entry = trade.get("entry_price", 0)
    target = trade.get("target", 0)
    sl = trade.get("stop_loss", 0)
    rr = trade.get("rr", 0)
    grade = trade.get("entry_grade", "?")
    strike = trade.get("strike", 0)
    entry_time = trade.get("entry_time", "")
    factors = trade.get("factors", "")

    lines = [
        f"<b>{tt} — {ticker}</b>",
        f"Entry: ₹{entry}",
        f"Target: ₹{target}",
        f"SL: ₹{sl}",
        f"RR: 1:{rr}",
        f"Grade: {grade}",
    ]
    if strike:
        lines.insert(3, f"Strike: {strike}")
    if entry_time:
        lines.insert(1, f"Time: {entry_time}")
    if factors:
        lines.append(f"Factors: {factors}")

    return "\n".join(lines)
