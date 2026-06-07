"""
Support & Resistance Level Detection Module - FIXED VERSION

Fixed Issues:
1. Better clustering using percentage + ATR hybrid
2. Improved quality scoring based on multiple factors
3. Market structure consideration (HH/HL/LL/LH)
4. Level strength based on touches and rejections
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

import numpy as np
import pandas as pd
from strategies.ta_helpers import atr as _atr, ema as _ema


class MarketStructure(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"
    TRANSITION = "transition"


class LevelType(Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"
    WEAK_SUPPORT = "weak_support"
    WEAK_RESISTANCE = "weak_resistance"


@dataclass
class SRLevel:
    price: float
    level_type: LevelType
    quality_score: float
    reactions: int
    source: str
    market_structure: MarketStructure
    distance_pct: float
    strength: str
    first_touch_idx: int
    last_touch_idx: int
    avg_rejection_size: float


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period:
        return (df["high"] - df["low"]).mean()
    atr = _atr(df["high"], df["low"], df["close"], period)
    return atr.iloc[-1] if atr is not None and not atr.empty else (df["high"] - df["low"]).mean()


def detect_market_structure(df: pd.DataFrame, lookback: int = 50) -> MarketStructure:
    if len(df) < lookback:
        return MarketStructure.RANGING
    
    recent = df.tail(lookback)
    highs = recent["high"].values
    lows = recent["low"].values
    closes = recent["close"].values
    
    swing_highs = []
    swing_lows = []
    
    for i in range(5, len(highs) - 5):
        if highs[i] == max(highs[i-5:i+6]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i-5:i+6]):
            swing_lows.append((i, lows[i]))
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return MarketStructure.RANGING
    
    hh_count = sum(1 for i in range(1, len(swing_highs)) if swing_highs[i][1] > swing_highs[i-1][1])
    lh_count = sum(1 for i in range(1, len(swing_highs)) if swing_highs[i][1] < swing_highs[i-1][1])
    hl_count = sum(1 for i in range(1, len(swing_lows)) if swing_lows[i][1] > swing_lows[i-1][1])
    ll_count = sum(1 for i in range(1, len(swing_lows)) if swing_lows[i][1] < swing_lows[i-1][1])
    
    total_swings = len(swing_highs) - 1 + len(swing_lows) - 1
    if total_swings == 0:
        return MarketStructure.RANGING
    
    bullish_score = (hh_count + hl_count) / total_swings
    bearish_score = (lh_count + ll_count) / total_swings
    
    ema_20 = _ema(pd.Series(closes), 20)
    ema_50 = _ema(pd.Series(closes), 50)
    
    if len(ema_20) > 0 and len(ema_50) > 0:
        trend_strength = (ema_20.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1]
    else:
        trend_strength = 0
    
    if bullish_score > 0.6 and trend_strength > 0.01:
        return MarketStructure.BULLISH
    elif bearish_score > 0.6 and trend_strength < -0.01:
        return MarketStructure.BEARISH
    elif abs(trend_strength) < 0.005:
        return MarketStructure.RANGING
    else:
        return MarketStructure.TRANSITION


def find_swing_points(df: pd.DataFrame, left: int = 5, right: int = 5) -> Tuple[List[Dict], List[Dict]]:
    resistance_levels: List[Dict] = []
    support_levels: List[Dict] = []

    for i in range(left, len(df) - right):
        window_high = df["high"].iloc[i - left:i + right + 1]
        window_low = df["low"].iloc[i - left:i + right + 1]

        if df["high"].iloc[i] == window_high.max():
            rejection_size = df["high"].iloc[i] - df["close"].iloc[i]
            resistance_levels.append({
                "datetime": df.index[i],
                "price": df["high"].iloc[i],
                "volume": df["volume"].iloc[i] if "volume" in df.columns else 0,
                "bar_index": i,
                "rejection_size": rejection_size,
                "body_size": abs(df["close"].iloc[i] - df["open"].iloc[i])
            })

        if df["low"].iloc[i] == window_low.min():
            rejection_size = df["close"].iloc[i] - df["low"].iloc[i]
            support_levels.append({
                "datetime": df.index[i],
                "price": df["low"].iloc[i],
                "volume": df["volume"].iloc[i] if "volume" in df.columns else 0,
                "bar_index": i,
                "rejection_size": rejection_size,
                "body_size": abs(df["close"].iloc[i] - df["open"].iloc[i])
            })

    resistance_levels = [
        l for j, l in enumerate(resistance_levels)
        if j == 0 or l["bar_index"] - resistance_levels[j - 1]["bar_index"] >= 8
    ]
    support_levels = [
        l for j, l in enumerate(support_levels)
        if j == 0 or l["bar_index"] - support_levels[j - 1]["bar_index"] >= 8
    ]

    return resistance_levels, support_levels


def cluster_levels_adaptive(levels: List[Dict], df: pd.DataFrame, 
                            min_reactions: int = 2) -> List[Dict]:
    if not levels:
        return []
    
    atr = calculate_atr(df)
    avg_price = df["close"].mean()
    
    levels_sorted = sorted(levels, key=lambda x: x["price"])
    clusters: List[List[Dict]] = []
    current_cluster = [levels_sorted[0]]
    
    for i in range(1, len(levels_sorted)):
        current_price = levels_sorted[i]["price"]
        cluster_price = current_cluster[-1]["price"]
        
        pct_tolerance = 0.008 * abs(current_price)
        atr_tolerance = atr * 0.4
        tolerance = max(pct_tolerance, atr_tolerance)
        
        if abs(current_price - cluster_price) <= tolerance:
            current_cluster.append(levels_sorted[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [levels_sorted[i]]
    clusters.append(current_cluster)
    
    clustered_levels: List[Dict] = []
    now_idx = len(df)
    
    for cluster in clusters:
        if len(cluster) < min_reactions:
            continue
        
        avg_price = float(np.mean([l["price"] for l in cluster]))
        total_volume = sum([l["volume"] for l in cluster])
        reaction_count = len(cluster)
        first_touch_idx = min([l["bar_index"] for l in cluster])
        last_touch_idx = max([l["bar_index"] for l in cluster])
        bars_since_touch = now_idx - last_touch_idx
        avg_rejection = float(np.mean([l["rejection_size"] for l in cluster]))
        
        recency_weight = 1.0 if bars_since_touch < 20 else 0.7 if bars_since_touch < 50 else 0.4
        reaction_weight = min(reaction_count / 5, 1.0)
        
        level_spread = max([l["price"] for l in cluster]) - min([l["price"] for l in cluster])
        precision = 1.0 - min(level_spread / (avg_price * 0.02), 1.0)
        
        rejection_quality = min(avg_rejection / atr, 1.0) if atr > 0 else 0.5
        
        quality_score = (
            recency_weight * 25 +
            reaction_weight * 30 +
            precision * 25 +
            rejection_quality * 20
        )
        
        clustered_levels.append({
            "price": avg_price,
            "reactions": reaction_count,
            "quality_score": round(quality_score, 1),
            "bars_since_touch": bars_since_touch,
            "total_volume": total_volume,
            "first_touch_idx": first_touch_idx,
            "last_touch_idx": last_touch_idx,
            "avg_rejection_size": round(avg_rejection, 2),
            "precision": round(precision, 2)
        })
    
    return sorted(clustered_levels, key=lambda x: x["quality_score"], reverse=True)


def find_weekly_sr_levels(df_weekly: pd.DataFrame, lookback_weeks: int = 260) -> Tuple[List[Dict], List[Dict]]:
    if df_weekly is None or len(df_weekly) < 20:
        return [], []
    
    lookback = min(len(df_weekly), lookback_weeks)
    resistance_levels, support_levels = find_swing_points(df_weekly.tail(lookback))
    
    if not resistance_levels and not support_levels:
        return [], []
    
    strong_resistances = cluster_levels_adaptive(resistance_levels, df_weekly, min_reactions=2)
    strong_supports = cluster_levels_adaptive(support_levels, df_weekly, min_reactions=2)
    
    for level in strong_resistances + strong_supports:
        level["source"] = "weekly"
    
    return strong_resistances, strong_supports


def confirm_levels_on_daily(weekly_levels: List[Dict], df_daily: pd.DataFrame,
                            tolerance_pct: float = 0.005) -> List[Dict]:
    if df_daily is None or len(df_daily) < 20 or not weekly_levels:
        return weekly_levels
    
    confirmed_levels: List[Dict] = []
    daily_highs = df_daily["high"].values
    daily_lows = df_daily["low"].values
    
    for level in weekly_levels:
        price = level["price"]
        tolerance = price * tolerance_pct
        
        touches = 0
        breaks = 0
        rejections = 0
        
        is_resistance = level.get("is_resistance", True)
        
        for i in range(len(df_daily)):
            high = daily_highs[i]
            low = daily_lows[i]
            close = df_daily["close"].iloc[i]
            
            if is_resistance:
                if abs(high - price) <= tolerance:
                    touches += 1
                    if close < price:
                        rejections += 1
                elif high > price + tolerance * 2:
                    breaks += 1
            else:
                if abs(low - price) <= tolerance:
                    touches += 1
                    if close > price:
                        rejections += 1
                elif low < price - tolerance * 2:
                    breaks += 1
        
        if touches >= 1 and breaks < 2:
            level["daily_touches"] = touches
            level["daily_rejections"] = rejections
            level["daily_confirmed"] = bool(rejections >= touches * 0.5)
            
            if level["daily_confirmed"]:
                level["quality_score"] = min(level.get("quality_score", 50) + 15, 100)
            
            confirmed_levels.append(level)
    
    return confirmed_levels


def get_all_sr_levels(df_15m: pd.DataFrame, df_daily: Optional[pd.DataFrame],
                      df_weekly: Optional[pd.DataFrame], avg_atr_15m: float,
                      lookback_weeks: int = 260, min_level_quality: int = 30) -> Tuple[List[Dict], List[Dict]]:
    
    market_structure = detect_market_structure(df_15m)
    
    weekly_resistances, weekly_supports = find_weekly_sr_levels(df_weekly, lookback_weeks)
    
    for level in weekly_resistances:
        level["is_resistance"] = True
    for level in weekly_supports:
        level["is_resistance"] = False
    
    confirmed_resistances = confirm_levels_on_daily(weekly_resistances, df_daily)
    confirmed_supports = confirm_levels_on_daily(weekly_supports, df_daily)
    
    lookback_15m = min(len(df_15m), 300)
    resistance_15m, support_15m = find_swing_points(df_15m.tail(lookback_15m))
    strong_resistances_15m = cluster_levels_adaptive(resistance_15m, df_15m, min_reactions=1)
    strong_supports_15m = cluster_levels_adaptive(support_15m, df_15m, min_reactions=1)
    
    for level in strong_resistances_15m:
        level["source"] = "15m"
        level["is_resistance"] = True
    for level in strong_supports_15m:
        level["source"] = "15m"
        level["is_resistance"] = False
    
    all_resistances: List[Dict] = []
    all_supports: List[Dict] = []
    
    for l in confirmed_resistances:
        all_resistances.append({
            "price": float(l["price"]),
            "reactions": l.get("reactions", 2),
            "quality_score": float(l.get("quality_score", 50)),
            "source": l.get("source", "weekly_daily"),
            "daily_confirmed": bool(l.get("daily_confirmed", False)),
            "daily_touches": l.get("daily_touches", 0),
            "daily_rejections": l.get("daily_rejections", 0),
            "market_structure": market_structure.value
        })
    
    for l in confirmed_supports:
        all_supports.append({
            "price": float(l["price"]),
            "reactions": l.get("reactions", 2),
            "quality_score": float(l.get("quality_score", 50)),
            "source": l.get("source", "weekly_daily"),
            "daily_confirmed": bool(l.get("daily_confirmed", False)),
            "daily_touches": l.get("daily_touches", 0),
            "daily_rejections": l.get("daily_rejections", 0),
            "market_structure": market_structure.value
        })
    
    current_price = float(df_15m["close"].iloc[-1])
    
    for l in strong_resistances_15m:
        exists = any(abs(float(l["price"]) - float(wl["price"])) / float(wl["price"]) < 0.004 for wl in all_resistances)
        if not exists:
            all_resistances.append({
                "price": float(l["price"]),
                "reactions": l["reactions"],
                "quality_score": float(l["quality_score"]),
                "source": "15m",
                "daily_confirmed": False,
                "market_structure": market_structure.value
            })
    
    for l in strong_supports_15m:
        exists = any(abs(float(l["price"]) - float(wl["price"])) / float(wl["price"]) < 0.004 for wl in all_supports)
        if not exists:
            all_supports.append({
                "price": float(l["price"]),
                "reactions": l["reactions"],
                "quality_score": float(l["quality_score"]),
                "source": "15m",
                "daily_confirmed": False,
                "market_structure": market_structure.value
            })
    
    def filter_and_score(levels: List[Dict], is_resistance: bool) -> List[Dict]:
        filtered = []
        for level in levels:
            qs = float(level.get("quality_score", 50))
            lp = float(level.get("price", 0))
            
            if qs < min_level_quality:
                continue
            
            distance_pct = abs(lp - current_price) / current_price * 100
            
            if market_structure == MarketStructure.BULLISH:
                if is_resistance:
                    qs = min(qs * 0.9, 100)
                else:
                    qs = min(qs * 1.1, 100)
            elif market_structure == MarketStructure.BEARISH:
                if is_resistance:
                    qs = min(qs * 1.1, 100)
                else:
                    qs = min(qs * 0.9, 100)
            
            level["quality_score"] = round(qs, 1)
            level["distance_pct"] = round(distance_pct, 2)
            
            if qs >= 70:
                level["strength"] = "strong"
            elif qs >= 50:
                level["strength"] = "moderate"
            else:
                level["strength"] = "weak"
            
            filtered.append(level)
        
        return sorted(filtered, key=lambda x: x["quality_score"], reverse=True)
    
    return filter_and_score(all_resistances, True), filter_and_score(all_supports, False)
