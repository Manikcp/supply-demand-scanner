"""
3-Step A+ Supply & Demand Strategy (Trade with Pat)

Step 1 — Identify Institutional Demand/Supply Zones
  - Look for explosive impulsive moves (3+ big consecutive candles)
  - Draw zone from candle body BEFORE the big push
  - Check for Fair Value Gap (FVG) as momentum confirmation
  - Mark confluence stack (EMA, structure)

Step 2 — Trade with the Trend (Break of Structure)
  - Mark swing highs and swing lows
  - Identify Break of Structure (BOS): swing high taken = uptrend
  - Only buy from demand in uptrend, sell from supply in downtrend

Step 3 — Entry Conditions
  - Wait for slow momentum as price approaches zone
  - Candle must CLOSE IN the zone or WICK INTO the zone
  - Do NOT enter if candle closes below demand zone / above supply zone
  - Enter on the confirmation candle (green candle after zone touch for demand)
  - SL tight to zone or below the wick
  - TP at recent price levels (1:1 or 1.5 R:R)

6 Keys to Valid Demand/Supply Zones:
  1. Fresh zone (not already used/retested)
  2. Close in or wick into zone (close through = invalid)
  3. Confluence stack (EMA as support/resistance)
  4. Slow momentum approach
  5. Higher timeframe alignment
  6. Clear institutional footprint (big volume on impulse)
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum

import numpy as np
import pandas as pd
import pandas_ta as ta


class SignalType(Enum):
    CE_BUY = "CE BUY"
    PE_BUY = "PE BUY"


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"


@dataclass
class Config:
    index_type: str = "NIFTY"
    use_vol_filt: bool = True

    lookback_bars: int = 600
    min_impulse_candles: int = 3
    max_consolidation_candles: int = 10
    impulse_body_mult: float = 1.3
    zone_atr_width: float = 1.5
    fvg_min_gap_atr: float = 0.3

    slow_momentum_bars: int = 3
    slow_momentum_max_body_pct: float = 0.4
    min_close_in_zone_pct: float = 0.3
    allow_wick_entry: bool = True

    trend_swing_lookback: int = 48
    bos_confirm_bars: int = 3

    ema_len: int = 20
    volume_avg_period: int = 20
    volume_spike_mult: float = 1.5
    atr_len: int = 14

    min_rr_ratio: float = 1.0
    target_rr: float = 1.5
    sl_atr_buffer: float = 0.5

    trading_capital: float = 100000
    risk_per_trade_percent: float = 1.5
    max_positions_per_day: int = 3
    max_loss_per_day_pct: float = 5.0
    min_score_threshold: int = 50
    require_trend_factor: bool = True
    block_opening_session: bool = True
    dedup_bar_cooldown: int = 5

    use_h1_zones: bool = True
    body_based_zones: bool = True
    next_bar_entry: bool = False

    @classmethod
    def from_json(cls, path: str = "config.json") -> "Config":
        import json, os
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = json.load(f)
        cfg = cls()
        general = data.get("general", {})
        cfg.trading_capital = general.get("trading_capital", 100000)
        cfg.risk_per_trade_percent = general.get("risk_per_trade_percent", 1.5)
        cfg.max_positions_per_day = general.get("max_positions_per_day", 3)
        cfg.max_loss_per_day_pct = general.get("max_loss_per_day_percent", 5.0)
        cfg.min_rr_ratio = general.get("min_reward_risk_ratio", 1.0)
        cfg.min_score_threshold = general.get("min_score_threshold", 50)
        cfg.require_trend_factor = general.get("require_trend_factor", True)
        cfg.block_opening_session = general.get("block_opening_session", True)
        cfg.sl_atr_buffer = general.get("sl_atr_buffer", 0.5)
        cfg.dedup_bar_cooldown = general.get("dedup_bar_cooldown", 5)
        cfg.zone_atr_width = general.get("zone_atr_width", 1.5)
        cfg.target_rr = general.get("target_rr", 1.5)
        sdd = data.get("supply_demand", {})
        cfg.min_impulse_candles = sdd.get("min_impulse_candles", 3)
        cfg.impulse_body_mult = sdd.get("impulse_body_mult", 1.3)
        cfg.fvg_min_gap_atr = sdd.get("fvg_min_gap_atr", 0.3)
        cfg.slow_momentum_max_body_pct = sdd.get("slow_momentum_max_body_pct", 0.4)
        cfg.trend_swing_lookback = sdd.get("trend_swing_lookback", 30)
        cfg.use_h1_zones = sdd.get("use_h1_zones", True)
        cfg.body_based_zones = sdd.get("body_based_zones", True)
        cfg.next_bar_entry = sdd.get("next_bar_entry", False)
        indicators = data.get("technical_indicators", {})
        rsi = indicators.get("rsi", {})
        cfg.ema_len = indicators.get("ema", {}).get("period", 20)
        st = indicators.get("supertrend", {})
        return cfg


def is_trading_session(dt: pd.Timestamp) -> Tuple[bool, str]:
    ist = dt.tz_convert("Asia/Kolkata") if dt.tzinfo else dt
    minutes = ist.hour * 60 + ist.minute
    if minutes < 555 or minutes > 930:
        return False, "closed"
    elif minutes < 600:
        return False, "opening"
    elif minutes < 840:
        return True, "regular"
    elif minutes > 900:
        return True, "pre_close"
    else:
        return True, "closing"


def find_swing_points(df: pd.DataFrame, lookback: int = 30) -> Tuple[List[int], List[int], List[float], List[float]]:
    high_idx: List[int] = []
    low_idx: List[int] = []
    high_vals: List[float] = []
    low_vals: List[float] = []
    n = len(df)
    if n < 10:
        return high_idx, low_idx, high_vals, low_vals
    window = max(3, lookback // 10)
    for i in range(window, n - window):
        if all(df["high"].iloc[i] >= df["high"].iloc[i - j] for j in range(1, window + 1)) and \
           all(df["high"].iloc[i] >= df["high"].iloc[i + j] for j in range(1, window + 1)):
            high_idx.append(i)
            high_vals.append(df["high"].iloc[i])
        if all(df["low"].iloc[i] <= df["low"].iloc[i - j] for j in range(1, window + 1)) and \
           all(df["low"].iloc[i] <= df["low"].iloc[i + j] for j in range(1, window + 1)):
            low_idx.append(i)
            low_vals.append(df["low"].iloc[i])
    return high_idx, low_idx, high_vals, low_vals


def detect_trend_bos(df: pd.DataFrame, lookback: int = 30) -> Tuple[str, float, float]:
    last_n = df.tail(lookback)
    hi, li, hv, lv = find_swing_points(last_n, lookback)
    if len(hi) < 2 or len(li) < 2:
        return "ranging", 0, 0
    recent_high = hv[-1] if hv else df["high"].iloc[-1]
    recent_low = lv[-1] if lv else df["low"].iloc[-1]
    prev_high = hv[-2] if len(hv) >= 2 else recent_high
    prev_low = lv[-2] if len(lv) >= 2 else recent_low
    bos_up = recent_high > prev_high
    bos_down = recent_low < prev_low
    if bos_up and not bos_down:
        return "bullish", recent_high, recent_low
    elif bos_down and not bos_up:
        return "bearish", recent_high, recent_low
    elif bos_up and bos_down:
        return "ranging", recent_high, recent_low
    return "ranging", recent_high, recent_low


def detect_impulsive_move(df: pd.DataFrame, atr: float,
                          start_idx: int, direction: str,
                          min_candles: int = 3,
                          body_mult: float = 1.3) -> Tuple[bool, int]:
    n = len(df)
    if start_idx >= n - min_candles:
        return False, 0
    body_sizes = df["close"].sub(df["open"]).abs()
    avg_body_series = body_sizes.rolling(20).mean()
    count = 0
    for i in range(start_idx, min(start_idx + min_candles + 3, n)):
        body = body_sizes.iloc[i]
        range_c = df["high"].iloc[i] - df["low"].iloc[i]
        if range_c == 0:
            continue
        avg_body = float(avg_body_series.iloc[i]) if i >= 20 else float(atr * 0.5)
        if np.isnan(avg_body) or avg_body == 0.0:
            avg_body = float(atr * 0.5) if atr > 0 else 1.0
        big_body = body > avg_body * body_mult
        good_candle = (body / range_c) > 0.4
        if direction == "up":
            if df["close"].iloc[i] > df["open"].iloc[i] and big_body and good_candle:
                count += 1
            else:
                if count >= min_candles:
                    return True, i - 1
                count = 0
        else:
            if df["close"].iloc[i] < df["open"].iloc[i] and big_body and good_candle:
                count += 1
            else:
                if count >= min_candles:
                    return True, i - 1
                count = 0
        if count >= min_candles:
            return True, i
    return count >= min_candles, start_idx + count - 1 if count >= min_candles else 0


def find_consolidation_before_impulse(df: pd.DataFrame, atr: float,
                                       impulse_start: int,
                                       max_candles: int = 8,
                                       use_body: bool = False) -> Optional[Tuple[int, int, float, float]]:
    if impulse_start < 2:
        return None
    zone_start = impulse_start
    if use_body:
        zone_high = max(df["open"].iloc[impulse_start - 1], df["close"].iloc[impulse_start - 1])
        zone_low = min(df["open"].iloc[impulse_start - 1], df["close"].iloc[impulse_start - 1])
    else:
        zone_high = df["high"].iloc[impulse_start - 1]
        zone_low = df["low"].iloc[impulse_start - 1]
    for j in range(impulse_start - 1, max(0, impulse_start - max_candles - 1), -1):
        body = abs(df["close"].iloc[j] - df["open"].iloc[j])
        rng = df["high"].iloc[j] - df["low"].iloc[j]
        if rng > atr * 2.5:
            break
        if use_body:
            zone_high = max(zone_high, df["open"].iloc[j], df["close"].iloc[j])
            zone_low = min(zone_low, df["open"].iloc[j], df["close"].iloc[j])
        else:
            zone_high = max(zone_high, df["high"].iloc[j])
            zone_low = min(zone_low, df["low"].iloc[j])
        zone_start = j
    zone_width = zone_high - zone_low
    if zone_width > atr * 3.0:
        return None
    return zone_start, impulse_start, zone_high, zone_low


def detect_fvg(df: pd.DataFrame, i: int, min_gap_atr: float = 0.3) -> bool:
    if i < 2 or i >= len(df):
        return False
    prev_high = df["high"].iloc[i - 1]
    prev_low = df["low"].iloc[i - 1]
    curr_high = df["high"].iloc[i]
    curr_low = df["low"].iloc[i]
    atr = (df["high"] - df["low"]).rolling(14).mean().iloc[i]
    if atr == 0:
        return False
    gap_up = curr_low > prev_high
    gap_down = curr_high < prev_low
    if gap_up or gap_down:
        gap_size = abs(curr_low - prev_high) if gap_up else abs(prev_low - curr_high)
        return bool(gap_size >= min_gap_atr * atr)
    return False


def check_slow_momentum(df: pd.DataFrame, i: int, lookback: int = 3,
                        max_body_pct: float = 0.4,
                        impulse_end: int = None) -> bool:
    if i < lookback:
        return False
    start = max(0, i - lookback)
    if impulse_end is not None:
        start = max(start, impulse_end + 1)
    if start > i:
        return True
    for j in range(start, i + 1):
        body = abs(df["close"].iloc[j] - df["open"].iloc[j])
        rng = df["high"].iloc[j] - df["low"].iloc[j]
        if rng == 0:
            continue
        if body / rng > (1 - max_body_pct):
            return False
    return True


def check_zone_entry(df: pd.DataFrame, i: int, zone_high: float, zone_low: float,
                     atr: float, direction: str,
                     min_close_in_pct: float = 0.3,
                     allow_wick: bool = True) -> Tuple[bool, str]:
    close = df["close"].iloc[i]
    high = df["high"].iloc[i]
    low = df["low"].iloc[i]

    if direction == "demand":
        zone_mid = (zone_high + zone_low) / 2
        zone_range = zone_high - zone_low
        if zone_range == 0:
            zone_range = atr * 0.2
        close_in_zone = zone_low <= close <= zone_high
        wick_in_zone = zone_low <= low <= zone_high or zone_low <= high <= zone_high
        close_below = close < zone_low
        if close_below:
            return False, "close_below_zone"
        if close_in_zone:
            overlap_pct = min(close, zone_high) - zone_low
            if overlap_pct / zone_range >= min_close_in_pct:
                return True, "close_in_zone"
        if allow_wick and wick_in_zone:
            return True, "wick_into_zone"
        return False, "not_in_zone"

    else:
        zone_mid = (zone_high + zone_low) / 2
        zone_range = zone_high - zone_low
        if zone_range == 0:
            zone_range = atr * 0.2
        close_in_zone = zone_low <= close <= zone_high
        wick_in_zone = zone_low <= low <= zone_high or zone_low <= high <= zone_high
        close_above = close > zone_high
        if close_above:
            return False, "close_above_zone"
        if close_in_zone:
            overlap_pct = zone_high - max(close, zone_low)
            if overlap_pct / zone_range >= min_close_in_pct:
                return True, "close_in_zone"
        if allow_wick and wick_in_zone:
            return True, "wick_into_zone"
        return False, "not_in_zone"


def is_fresh_zone(zone_start_idx: int, df: pd.DataFrame, zone_high: float, zone_low: float,
                  impulse_end_idx: int = None) -> bool:
    start = max(zone_start_idx + 1, (impulse_end_idx or 0) + 1)
    for j in range(start, len(df) - 1):
        close = df["close"].iloc[j]
        if zone_low <= close <= zone_high:
            return False
    return True


def detect_demand_zones(df: pd.DataFrame, atr_series: pd.Series,
                        cfg: Config) -> List[Dict]:
    zones: List[Dict] = []
    n = len(df)
    atr = atr_series.iloc[-1] if not atr_series.empty else df["close"].iloc[-1] * 0.01

    for i in range(50, n - cfg.min_impulse_candles):
        current_atr = atr_series.iloc[i] if i < len(atr_series) else atr
        if current_atr <= 0:
            continue
        found, end_idx = detect_impulsive_move(
            df, current_atr, i, "up",
            cfg.min_impulse_candles, cfg.impulse_body_mult
        )
        if found and end_idx >= i:
            consolidation = find_consolidation_before_impulse(
                df, current_atr, i, cfg.max_consolidation_candles,
                use_body=cfg.body_based_zones
            )
            if consolidation:
                zs, ze, zh, zl = consolidation
                fvg_found = any(detect_fvg(df, j, cfg.fvg_min_gap_atr)
                                for j in range(i, min(end_idx + 1, n)))
                fresh = is_fresh_zone(zs, df, zh, zl, impulse_end_idx=end_idx)
                if fresh:
                    zones.append({
                        "type": "demand",
                        "start_idx": zs,
                        "impulse_start": i,
                        "impulse_end": end_idx,
                        "zone_high": zh,
                        "zone_low": zl,
                        "zone_mid": (zh + zl) / 2,
                        "fvg": fvg_found,
                        "fresh": fresh,
                        "strength": 70 if fvg_found else 50,
                    })
    merged = merge_overlapping_zones(zones)
    return merged


def detect_supply_zones(df: pd.DataFrame, atr_series: pd.Series,
                         cfg: Config) -> List[Dict]:
    zones: List[Dict] = []
    n = len(df)
    atr = atr_series.iloc[-1] if not atr_series.empty else df["close"].iloc[-1] * 0.01

    for i in range(50, n - cfg.min_impulse_candles):
        current_atr = atr_series.iloc[i] if i < len(atr_series) else atr
        if current_atr <= 0:
            continue
        found, end_idx = detect_impulsive_move(
            df, current_atr, i, "down",
            cfg.min_impulse_candles, cfg.impulse_body_mult
        )
        if found and end_idx >= i:
            consolidation = find_consolidation_before_impulse(
                df, current_atr, i, cfg.max_consolidation_candles,
                use_body=cfg.body_based_zones
            )
            if consolidation:
                zs, ze, zh, zl = consolidation
                fvg_found = any(detect_fvg(df, j, cfg.fvg_min_gap_atr)
                                for j in range(i, min(end_idx + 1, n)))
                fresh = is_fresh_zone(zs, df, zh, zl, impulse_end_idx=end_idx)
                if fresh:
                    zones.append({
                        "type": "supply",
                        "start_idx": zs,
                        "impulse_start": i,
                        "impulse_end": end_idx,
                        "zone_high": zh,
                        "zone_low": zl,
                        "zone_mid": (zh + zl) / 2,
                        "fvg": fvg_found,
                        "fresh": fresh,
                        "strength": 70 if fvg_found else 50,
                    })
    merged = merge_overlapping_zones(zones)
    return merged


def merge_overlapping_zones(zones: List[Dict]) -> List[Dict]:
    if not zones:
        return []
    sorted_z = sorted(zones, key=lambda z: z["zone_low"])
    merged: List[Dict] = []
    current = sorted_z[0]
    for z in sorted_z[1:]:
        if z["zone_low"] <= current["zone_high"]:
            current["zone_high"] = max(current["zone_high"], z["zone_high"])
            current["zone_low"] = min(current["zone_low"], z["zone_low"])
            current["zone_mid"] = (current["zone_high"] + current["zone_low"]) / 2
            current["fvg"] = current["fvg"] or z["fvg"]
            current["strength"] = max(current["strength"], z["strength"])
            current["impulse_end"] = max(current["impulse_end"], z["impulse_end"])
        else:
            merged.append(current)
            current = z
    merged.append(current)
    return merged


def calculate_entry_score(zone: Dict, trend_ok: bool, entry_ok: bool,
                          slow_mom: bool, fvg: bool, regime: MarketRegime,
                          volume_spike: bool = False) -> dict:
    score = 0
    factors: List[str] = []

    if trend_ok:
        score += 25
        factors.append("trend_aligned")

    if entry_ok:
        score += 20
        factors.append("entry_confirmed")

    if slow_mom:
        score += 15
        factors.append("slow_momentum")

    if fvg:
        score += 15
        factors.append("fvg_confirm")

    if zone and zone.get("strength", 0) >= 70:
        score += 10
        factors.append("strong_zone")

    if volume_spike:
        score += 10
        factors.append("volume_spike")

    if regime == MarketRegime.TRENDING_UP or regime == MarketRegime.TRENDING_DOWN:
        score += 5
        factors.append("regime_align")

    return {
        "score": min(max(score, 0), 100),
        "grade": "A" if score >= 75 else "B" if score >= 55 else "C" if score >= 35 else "D",
        "factors": factors
    }


def run_signals(df_15m: pd.DataFrame, df_daily: Optional[pd.DataFrame],
                cfg: Optional[Config] = None,
                df_weekly: Optional[pd.DataFrame] = None,
                df_h1: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if cfg is None:
        cfg = Config()

    df = df_15m.copy()

    def _safe_series(s, index):
        if s is None:
            return pd.Series(np.nan, index=index)
        return s

    df["atr"] = _safe_series(
        ta.atr(df["high"], df["low"], df["close"], length=cfg.atr_len), df.index
    )
    df["atr"] = df["atr"].fillna(df["close"] * 0.01)

    df["ema"] = _safe_series(ta.ema(df["close"], length=cfg.ema_len), df.index)

    df["session_ok"], df["session"] = zip(*df.index.to_series().apply(is_trading_session))

    if "volume" in df.columns:
        df["avg_vol"] = df["volume"].rolling(cfg.volume_avg_period).mean()
        df["vol_spike"] = df["volume"] > df["avg_vol"] * cfg.volume_spike_mult
    else:
        df["avg_vol"] = 0
        df["vol_spike"] = False

    atr_series = df["atr"]
    trend, bos_high, bos_low = detect_trend_bos(df, cfg.trend_swing_lookback)

    demand_zones: List[Dict] = []
    supply_zones: List[Dict] = []

    if cfg.use_h1_zones and df_h1 is not None:
        df_h1c = df_h1.copy()
        atr_h1 = _safe_series(
            ta.atr(df_h1c["high"], df_h1c["low"], df_h1c["close"], length=cfg.atr_len),
            df_h1c.index
        )
        atr_h1 = atr_h1.fillna(df_h1c["close"] * 0.01)
        demand_zones = detect_demand_zones(df_h1c, atr_h1, cfg)
        supply_zones = detect_supply_zones(df_h1c, atr_h1, cfg)
        for z in demand_zones + supply_zones:
            z["impulse_end_dt"] = df_h1c.index[z["impulse_end"]]

    if not demand_zones and not supply_zones:
        demand_zones = detect_demand_zones(df, atr_series, cfg)
        supply_zones = detect_supply_zones(df, atr_series, cfg)
        for z in demand_zones + supply_zones:
            z["impulse_end_dt"] = df.index[z["impulse_end"]]

    regime = MarketRegime.TRENDING_UP if trend == "bullish" else \
             MarketRegime.TRENDING_DOWN if trend == "bearish" else \
             MarketRegime.RANGING

    current_price = df["close"].iloc[-1]
    atr = atr_series.iloc[-1] if not atr_series.empty else current_price * 0.01

    zone_half = atr * cfg.zone_atr_width

    signals: List[Dict[str, Any]] = []
    last_signal_bar: Dict[str, int] = {}

    signal_start_idx = max(100, len(df) - 200)

    # Check all qualifying demand zones (newest first), stop at first signal
    demand_candidates = [z for z in reversed(demand_zones) if z["zone_low"] < current_price]
    for dz in demand_candidates:
        if not dz.get("fresh", True):
            continue
        if trend not in ("bullish", "ranging"):
            continue

        impulse_start_5m = df.index.get_indexer([dz["impulse_end_dt"]], method='bfill')[0]
        if impulse_start_5m >= len(df) - 5:
            continue

        dz_high = dz["zone_high"] + zone_half
        dz_low = dz["zone_low"] - zone_half
        zone_key = f"DZ{round(dz_low, -1)}"
        last_bar = last_signal_bar.get(zone_key, -cfg.dedup_bar_cooldown)

        for i in range(max(signal_start_idx, impulse_start_5m + 1, last_bar + cfg.dedup_bar_cooldown), len(df)):
            row = df.iloc[i]
            if not row["session_ok"]:
                continue
            close = row["close"]
            low = row["low"]
            atr_i = row["atr"]

            in_zone_area = low <= dz_high and close >= dz_low - atr_i * 0.3
            if not in_zone_area:
                continue

            entry_ok, entry_reason = check_zone_entry(
                df, i, dz_high, dz_low, atr_i, "demand",
                cfg.min_close_in_zone_pct, cfg.allow_wick_entry
            )
            if not entry_ok:
                continue

            slow_mom = check_slow_momentum(df, i, cfg.slow_momentum_bars,
                                            cfg.slow_momentum_max_body_pct,
                                            impulse_end=impulse_start_5m)
            if not slow_mom:
                continue

            volume_spike = bool(row["vol_spike"]) if cfg.use_vol_filt else True
            if cfg.use_vol_filt and not volume_spike:
                continue

            if cfg.block_opening_session and row["session"] == "opening":
                continue

            if cfg.next_bar_entry:
                if i + 1 >= len(df):
                    continue
                next_close = df["close"].iloc[i + 1]
                next_open = df["open"].iloc[i + 1]
                if next_close <= next_open:
                    continue

            entry_price = close if not cfg.next_bar_entry else df["close"].iloc[i + 1]
            sl = dz_low - atr_i * cfg.sl_atr_buffer
            target_price = entry_price + (entry_price - sl) * cfg.target_rr

            half_range = df["high"].iloc[impulse_start_5m:i+1].max()
            if half_range >= target_price:
                continue

            risk = entry_price - sl
            reward = target_price - entry_price
            rr = round(reward / risk, 2) if risk > 0 else 0
            if rr < cfg.min_rr_ratio:
                continue

            trend_ok = trend == "bullish"
            quality = calculate_entry_score(
                dz, trend_ok, entry_ok, slow_mom,
                dz["fvg"], regime, volume_spike
            )
            if quality["score"] < cfg.min_score_threshold:
                continue

            need_trend = cfg.require_trend_factor and regime != MarketRegime.RANGING
            if need_trend and "trend_aligned" not in quality["factors"]:
                continue

            strike_step = 50.0 if cfg.index_type == "NIFTY" else 100.0
            atm = round(close / strike_step) * strike_step

            signals.append({
                "datetime": df.index[i + 1] if cfg.next_bar_entry else df.index[i],
                "type": "CE BUY",
                "index": cfg.index_type,
                "strike": int(atm),
                "entry": round(entry_price, 2),
                "support": round(dz_low, 2),
                "target": round(target_price, 2),
                "sl": round(sl, 2),
                "rr": rr,
                "entry_quality": quality["score"],
                "entry_grade": quality["grade"],
                "factors": ", ".join(quality["factors"]),
                "session": row["session"],
                "regime": regime.value,
                "market_structure": trend,
                "pattern": "Demand Zone Retest",
                "strong_signal": bool(dz.get("fvg", False)),
                "supertrend": trend,
                "rsi_zone": "neutral",
                "ema_aligned": bool(close > row["ema"]) if "ema" in row and pd.notna(row["ema"]) else False,
                "zone_entry_reason": entry_reason,
                "slow_momentum": slow_mom,
                "fvg_present": dz.get("fvg", False),
            })
            last_signal_bar[zone_key] = i
            break

    # Check all qualifying supply zones (newest first), stop at first signal
    supply_candidates = [z for z in reversed(supply_zones) if z["zone_high"] > current_price]
    for sz in supply_candidates:
        if not sz.get("fresh", True):
            continue
        if trend not in ("bearish", "ranging"):
            continue

        impulse_start_5m = df.index.get_indexer([sz["impulse_end_dt"]], method='bfill')[0]
        if impulse_start_5m >= len(df) - 5:
            continue

        sz_high = sz["zone_high"] + zone_half
        sz_low = sz["zone_low"] - zone_half
        zone_key = f"SZ{round(sz_high, -1)}"
        last_bar = last_signal_bar.get(zone_key, -cfg.dedup_bar_cooldown)

        for i in range(max(signal_start_idx, impulse_start_5m + 1, last_bar + cfg.dedup_bar_cooldown), len(df)):
            row = df.iloc[i]
            if not row["session_ok"]:
                continue
            close = row["close"]
            high = row["high"]
            atr_i = row["atr"]

            in_zone_area = high >= sz_low and close <= sz_high + atr_i * 0.3
            if not in_zone_area:
                continue

            entry_ok, entry_reason = check_zone_entry(
                df, i, sz_high, sz_low, atr_i, "supply",
                cfg.min_close_in_zone_pct, cfg.allow_wick_entry
            )
            if not entry_ok:
                continue

            slow_mom = check_slow_momentum(df, i, cfg.slow_momentum_bars,
                                            cfg.slow_momentum_max_body_pct,
                                            impulse_end=impulse_start_5m)
            if not slow_mom:
                continue

            volume_spike = bool(row["vol_spike"]) if cfg.use_vol_filt else True
            if cfg.use_vol_filt and not volume_spike:
                continue

            if cfg.block_opening_session and row["session"] == "opening":
                continue

            if cfg.next_bar_entry:
                if i + 1 >= len(df):
                    continue
                next_close = df["close"].iloc[i + 1]
                next_open = df["open"].iloc[i + 1]
                if next_close >= next_open:
                    continue

            entry_price = close if not cfg.next_bar_entry else df["close"].iloc[i + 1]
            sl = sz_high + atr_i * cfg.sl_atr_buffer
            target_price = close - (sl - close) * cfg.target_rr

            half_range = df["low"].iloc[impulse_start_5m:i+1].min()
            if half_range <= target_price:
                continue

            risk = sl - entry_price
            reward = entry_price - target_price
            rr = round(reward / risk, 2) if risk > 0 else 0
            if rr < cfg.min_rr_ratio:
                continue

            trend_ok = trend == "bearish"
            quality = calculate_entry_score(
                sz, trend_ok, entry_ok, slow_mom,
                sz["fvg"], regime, volume_spike
            )
            if quality["score"] < cfg.min_score_threshold:
                continue

            need_trend = cfg.require_trend_factor and regime != MarketRegime.RANGING
            if need_trend and "trend_aligned" not in quality["factors"]:
                continue

            strike_step = 50.0 if cfg.index_type == "NIFTY" else 100.0
            atm = round(close / strike_step) * strike_step

            signals.append({
                "datetime": df.index[i + 1] if cfg.next_bar_entry else df.index[i],
                "type": "PE BUY",
                "index": cfg.index_type,
                "strike": int(atm),
                "entry": round(entry_price, 2),
                "resistance": round(sz_high, 2),
                "target": round(target_price, 2),
                "sl": round(sl, 2),
                "rr": rr,
                "entry_quality": quality["score"],
                "entry_grade": quality["grade"],
                "factors": ", ".join(quality["factors"]),
                "session": row["session"],
                "regime": regime.value,
                "market_structure": trend,
                "pattern": "Supply Zone Retest",
                "strong_signal": bool(sz.get("fvg", False)),
                "supertrend": trend,
                "rsi_zone": "neutral",
                "ema_aligned": bool(close > row["ema"]) if "ema" in row and pd.notna(row["ema"]) else False,
                "zone_entry_reason": entry_reason,
                "slow_momentum": slow_mom,
                "fvg_present": sz.get("fvg", False),
            })
            last_signal_bar[zone_key] = i
            break

    signals_df = pd.DataFrame(signals)
    if not signals_df.empty:
        signals_df = signals_df.sort_values("datetime").reset_index(drop=True)

    return df, signals_df


def print_signal(row: dict) -> None:
    arrow = "🟢" if "CE" in row["type"] else "🔴"
    zone_type = "Demand Zone" if "CE" in row["type"] else "Supply Zone"
    fvg = "★ FVG" if row.get("fvg_present") else ""
    ticker = row.get("index", "") + str(row.get("strike", ""))

    print(f"\n{'='*60}")
    print(f"  {arrow} {row['type']}{fvg} — {ticker} @ ₹{row.get('entry', 0)} | Grade {row.get('entry_grade', 'N/A')} | ⏳")
    print(f"{'='*60}")
    print(f"  Pattern: {row.get('pattern', zone_type)}")
    print(f"  Entry   : ₹{row.get('entry', 0)}")
    print(f"  Target  : ₹{row.get('target', 0)}")
    print(f"  SL      : ₹{row['sl']}")
    if "support" in row:
        print(f"  Support : ₹{row['support']}")
    if "resistance" in row:
        print(f"  Resis   : ₹{row['resistance']}")
    print(f"  RR      : 1:{row['rr']}")
    print(f"  Trend   : {'🟢' if row.get('market_structure') == 'bullish' else '🔴'} {row.get('market_structure', 'N/A')} | Entry: {row.get('zone_entry_reason', 'N/A')}")
    print(f"  Strong  : {'Yes ★' if row.get('strong_signal') else 'No'} | Slow Mom: {'Yes' if row.get('slow_momentum') else 'No'} | FVG: {'Yes' if row.get('fvg_present') else 'No'}")
    print(f"  EMA     : {'Aligned' if row.get('ema_aligned') else 'Not Aligned'}")
    print(f"  Time    : {row['datetime']}")
    print(f"  Factors : {row['factors']}")
    print(f"{'='*60}")
