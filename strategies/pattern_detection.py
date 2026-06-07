"""
Chart Pattern Detection Module - FIXED VERSION

Fixed Issues:
1. Dynamic tolerance based on ATR (not fixed 3%)
2. Early entry signals (approaching pattern, not after breakout)
3. More pivots scanned (20 instead of 10)
4. Pattern confidence scoring
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from strategies.ta_helpers import atr as _atr


class PatternType(Enum):
    HEAD_AND_SHOULDERS = "Head & Shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "Inverse Head & Shoulders"
    DOUBLE_TOP = "Double Top"
    DOUBLE_BOTTOM = "Double Bottom"
    ASCENDING_TRIANGLE = "Ascending Triangle"
    DESCENDING_TRIANGLE = "Descending Triangle"
    SYMMETRICAL_TRIANGLE = "Symmetrical Triangle"
    BULL_FLAG = "Bull Flag"
    BEAR_FLAG = "Bear Flag"
    RISING_WEDGE = "Rising Wedge"
    FALLING_WEDGE = "Falling Wedge"
    CUP_AND_HANDLE = "Cup & Handle"
    ROUNDING_BOTTOM = "Rounding Bottom"
    NONE = "No Pattern"


class PatternStage(Enum):
    FORMING = "forming"
    APPROACHING = "approaching"
    BREAKOUT = "breakout"
    CONFIRMED = "confirmed"


@dataclass
class Pattern:
    pattern_type: PatternType
    direction: str
    reliability: str
    entry_price: Optional[float]
    target_price: Optional[float]
    stop_loss: Optional[float]
    pattern_score: float
    breakout_confirmed: bool
    neckline: Optional[float] = None
    measured_move: Optional[float] = None
    stage: PatternStage = PatternStage.FORMING
    distance_to_trigger: float = 0.0
    confidence: float = 0.0
    pivot_count: int = 0


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    atr = _atr(df["high"], df["low"], df["close"], period)
    return atr.iloc[-1] if atr is not None and not atr.empty else df["close"].iloc[-1] * 0.02


def dynamic_tolerance(price: float, atr: float, base_pct: float = 0.012) -> float:
    return max(atr * 1.0, price * base_pct)


def find_pivots(df: pd.DataFrame, left: int = 5, right: int = 5) -> pd.DataFrame:
    df = df.copy()
    df["pivot_high"] = np.nan
    df["pivot_low"] = np.nan
    df["pivot_high_idx"] = np.nan
    df["pivot_low_idx"] = np.nan

    for i in range(left, len(df) - right):
        window_high = df["high"].iloc[i - left:i + right + 1]
        window_low = df["low"].iloc[i - left:i + right + 1]

        if df["high"].iloc[i] == window_high.max():
            df.iloc[i, df.columns.get_loc("pivot_high")] = df["high"].iloc[i]
            df.iloc[i, df.columns.get_loc("pivot_high_idx")] = i

        if df["low"].iloc[i] == window_low.min():
            df.iloc[i, df.columns.get_loc("pivot_low")] = df["low"].iloc[i]
            df.iloc[i, df.columns.get_loc("pivot_low_idx")] = i

    return df


def _dedup_pivots(pivot_rows, col: str, idx_col: str, min_gap: int = 10):
    """Merge nearby pivots of same type, keeping the extreme value."""
    if not len(pivot_rows):
        return [], []
    prices, indices = [], []
    cluster_price = pivot_rows.iloc[0][col]
    cluster_idx = int(pivot_rows.iloc[0][idx_col])
    is_high = "high" in col
    for i in range(1, len(pivot_rows)):
        idx = int(pivot_rows.iloc[i][idx_col])
        price = pivot_rows.iloc[i][col]
        if idx - cluster_idx < min_gap:
            if is_high:
                if price > cluster_price:
                    cluster_price = price
            else:
                if price < cluster_price:
                    cluster_price = price
        else:
            prices.append(cluster_price)
            indices.append(cluster_idx)
            cluster_price = price
            cluster_idx = idx
    prices.append(cluster_price)
    indices.append(cluster_idx)
    return prices, indices


def calculate_pattern_confidence(pivot_count: int, symmetry: float, 
                                  volume_confirm: bool, stage: PatternStage) -> float:
    base = 50.0
    
    pivot_score = min(pivot_count * 5, 20)
    symmetry_score = symmetry * 20
    volume_score = 10 if volume_confirm else 0
    stage_score = {"forming": 5, "approaching": 15, "breakout": 10, "confirmed": 20}.get(stage.value, 5)
    
    return min(base + pivot_score + symmetry_score + volume_score + stage_score, 100.0)


def detect_head_and_shoulders(df: pd.DataFrame, lookback: int = 100, 
                               early_entry_pct: float = 0.02) -> Optional[Pattern]:
    atr = calculate_atr(df)
    df = find_pivots(df, left=5, right=5)

    pivot_rows = df[df["pivot_high"].notna()].tail(15)
    prices, indices = _dedup_pivots(pivot_rows, "pivot_high", "pivot_high_idx")
    if len(prices) < 3:
        return None

    current_price = df["close"].iloc[-1]

    ls_price, ls_idx = prices[-3], indices[-3]
    h_price, h_idx = prices[-2], indices[-2]
    rs_price, rs_idx = prices[-1], indices[-1]

    tolerance = dynamic_tolerance(ls_price, atr)
    shoulders_match = abs(ls_price - rs_price) <= tolerance
    head_higher = h_price > max(ls_price, rs_price) * 1.02

    if shoulders_match and head_higher:
        valley_ls_h = df["low"].iloc[ls_idx:h_idx + 1].min()
        valley_h_rs = df["low"].iloc[h_idx:rs_idx + 1].min()
        if valley_ls_h <= ls_price - atr * 0.5 and valley_h_rs <= rs_price - atr * 0.5:
            neckline = min(valley_ls_h, valley_h_rs)

            head_height = h_price - neckline
            target = neckline - head_height

            distance_to_neckline = (current_price - neckline) / current_price
            symmetry = 1.0 - abs(ls_price - rs_price) / max(ls_price, rs_price)

            if current_price <= neckline:
                stage = PatternStage.BREAKOUT
                entry_price = current_price
                sl = neckline + atr * 1.0
            elif distance_to_neckline <= early_entry_pct:
                stage = PatternStage.APPROACHING
                entry_price = neckline
                sl = h_price + (atr * 0.5)
            else:
                stage = PatternStage.FORMING
                entry_price = neckline
                sl = h_price + atr

            distance_to_trigger = abs(current_price - neckline) / atr

            confidence = calculate_pattern_confidence(
                pivot_count=3,
                symmetry=symmetry,
                volume_confirm=False,
                stage=stage
            )

            return Pattern(
                pattern_type=PatternType.HEAD_AND_SHOULDERS,
                direction="bearish",
                reliability="high" if confidence >= 70 else "medium" if confidence >= 50 else "low",
                entry_price=round(entry_price, 2),
                target_price=round(target, 2),
                stop_loss=round(sl, 2),
                pattern_score=85,
                breakout_confirmed=(stage == PatternStage.BREAKOUT),
                neckline=round(neckline, 2),
                measured_move=round(head_height, 2),
                stage=stage,
                distance_to_trigger=round(distance_to_trigger, 2),
                confidence=round(confidence, 1),
                pivot_count=3
            )

    return None


def detect_inverse_head_and_shoulders(df: pd.DataFrame, lookback: int = 100,
                                        early_entry_pct: float = 0.02) -> Optional[Pattern]:
    atr = calculate_atr(df)
    df = find_pivots(df, left=5, right=5)

    pivot_rows = df[df["pivot_low"].notna()].tail(15)
    prices, indices = _dedup_pivots(pivot_rows, "pivot_low", "pivot_low_idx")
    if len(prices) < 3:
        return None

    current_price = df["close"].iloc[-1]

    ls_price, ls_idx = prices[-3], indices[-3]
    h_price, h_idx = prices[-2], indices[-2]
    rs_price, rs_idx = prices[-1], indices[-1]

    tolerance = dynamic_tolerance(ls_price, atr)
    shoulders_match = abs(ls_price - rs_price) <= tolerance
    head_lower = h_price < min(ls_price, rs_price) * 0.98

    if shoulders_match and head_lower:
        peak_ls_h = df["high"].iloc[ls_idx:h_idx + 1].max()
        peak_h_rs = df["high"].iloc[h_idx:rs_idx + 1].max()
        if peak_ls_h >= ls_price + atr * 0.5 and peak_h_rs >= rs_price + atr * 0.5:
            neckline = max(peak_ls_h, peak_h_rs)

            head_height = neckline - h_price
            target = neckline + head_height

            distance_to_neckline = (neckline - current_price) / current_price
            symmetry = 1.0 - abs(ls_price - rs_price) / max(ls_price, rs_price)

            if current_price >= neckline:
                stage = PatternStage.BREAKOUT
                entry_price = current_price
                sl = neckline - atr * 1.0
            elif distance_to_neckline <= early_entry_pct:
                stage = PatternStage.APPROACHING
                entry_price = neckline
                sl = h_price - (atr * 0.5)
            else:
                stage = PatternStage.FORMING
                entry_price = neckline
                sl = h_price - atr

            distance_to_trigger = abs(current_price - neckline) / atr

            confidence = calculate_pattern_confidence(
                pivot_count=3,
                symmetry=symmetry,
                volume_confirm=False,
                stage=stage
            )

            return Pattern(
                pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
                direction="bullish",
                reliability="high" if confidence >= 70 else "medium" if confidence >= 50 else "low",
                entry_price=round(entry_price, 2),
                target_price=round(target, 2),
                stop_loss=round(sl, 2),
                pattern_score=90,
                breakout_confirmed=(stage == PatternStage.BREAKOUT),
                neckline=round(neckline, 2),
                measured_move=round(head_height, 2),
                stage=stage,
                distance_to_trigger=round(distance_to_trigger, 2),
                confidence=round(confidence, 1),
                pivot_count=3
            )

    return None


def detect_double_top(df: pd.DataFrame, lookback: int = 60,
                      early_entry_pct: float = 0.02) -> Optional[Pattern]:
    atr = calculate_atr(df)
    df = find_pivots(df, left=5, right=5)

    pivot_rows = df[df["pivot_high"].notna()].tail(15)
    prices, indices = _dedup_pivots(pivot_rows, "pivot_high", "pivot_high_idx")
    if len(prices) < 2:
        return None

    current_price = df["close"].iloc[-1]

    t2_price = prices[-1]
    t2_idx = indices[-1]
    for i in range(len(prices) - 1):
        t1_price = prices[i]
        t1_idx = indices[i]

        tolerance = dynamic_tolerance(t1_price, atr)
        tops_match = abs(t1_price - t2_price) <= tolerance

        if tops_match:
            valley = df["low"].iloc[t1_idx:t2_idx + 1].min()
            valley_depth = min(t1_price, t2_price) - valley
            if valley_depth < atr * 2.5:
                continue

            neckline = valley
            pattern_height = max(t1_price, t2_price) - neckline
            target = neckline - pattern_height

            distance_to_neckline = (current_price - neckline) / current_price
            if distance_to_neckline > early_entry_pct:
                continue
            
            symmetry = 1.0 - abs(t1_price - t2_price) / max(t1_price, t2_price)
            distance_to_neckline = (current_price - neckline) / current_price
            
            if current_price <= neckline:
                stage = PatternStage.BREAKOUT
                entry_price = current_price
                sl = neckline + atr * 1.0
            elif distance_to_neckline <= early_entry_pct:
                stage = PatternStage.APPROACHING
                entry_price = neckline
                sl = max(t1_price, t2_price) + (atr * 0.5)
            else:
                stage = PatternStage.FORMING
                entry_price = neckline
                sl = max(t1_price, t2_price) + atr
            
            distance_to_trigger = abs(current_price - neckline) / atr
            
            confidence = calculate_pattern_confidence(
                pivot_count=2,
                symmetry=symmetry,
                volume_confirm=False,
                stage=stage
            )

            return Pattern(
                pattern_type=PatternType.DOUBLE_TOP,
                direction="bearish",
                reliability="high" if confidence >= 70 else "medium" if confidence >= 50 else "low",
                entry_price=round(entry_price, 2),
                target_price=round(target, 2),
                stop_loss=round(sl, 2),
                pattern_score=80,
                breakout_confirmed=(stage == PatternStage.BREAKOUT),
                neckline=round(neckline, 2),
                measured_move=round(pattern_height, 2),
                stage=stage,
                distance_to_trigger=round(distance_to_trigger, 2),
                confidence=round(confidence, 1),
                pivot_count=2
            )

    return None


def detect_double_bottom(df: pd.DataFrame, lookback: int = 60,
                         early_entry_pct: float = 0.02) -> Optional[Pattern]:
    atr = calculate_atr(df)
    df = find_pivots(df, left=5, right=5)

    pivot_rows = df[df["pivot_low"].notna()].tail(15)
    prices, indices = _dedup_pivots(pivot_rows, "pivot_low", "pivot_low_idx")
    if len(prices) < 2:
        return None

    current_price = df["close"].iloc[-1]

    b2_price = prices[-1]
    b2_idx = indices[-1]
    for i in range(len(prices) - 1):
        b1_price = prices[i]
        b1_idx = indices[i]

        tolerance = dynamic_tolerance(b1_price, atr)
        bottoms_match = abs(b1_price - b2_price) <= tolerance

        if bottoms_match:
            peak = df["high"].iloc[b1_idx:b2_idx + 1].max()
            peak_height = peak - max(b1_price, b2_price)
            if peak_height < atr * 2.5:
                continue

            downtrend_start = df["close"].iloc[max(0, b1_idx - 30):b1_idx].max()
            if b1_idx >= 30 and b1_price >= downtrend_start * 0.98:
                continue

            neckline = peak
            pattern_height = neckline - min(b1_price, b2_price)
            target = neckline + pattern_height

            distance_to_neckline = (neckline - current_price) / current_price
            if distance_to_neckline > early_entry_pct:
                continue
            
            symmetry = 1.0 - abs(b1_price - b2_price) / max(b1_price, b2_price)
            distance_to_neckline = (neckline - current_price) / current_price
            
            if current_price >= neckline:
                stage = PatternStage.BREAKOUT
                entry_price = current_price
                sl = neckline - atr * 1.0
            elif distance_to_neckline <= early_entry_pct:
                stage = PatternStage.APPROACHING
                entry_price = neckline
                sl = min(b1_price, b2_price) - (atr * 0.5)
            else:
                stage = PatternStage.FORMING
                entry_price = neckline
                sl = min(b1_price, b2_price) - atr
            
            distance_to_trigger = abs(current_price - neckline) / atr
            
            confidence = calculate_pattern_confidence(
                pivot_count=2,
                symmetry=symmetry,
                volume_confirm=False,
                stage=stage
            )

            return Pattern(
                pattern_type=PatternType.DOUBLE_BOTTOM,
                direction="bullish",
                reliability="high" if confidence >= 70 else "medium" if confidence >= 50 else "low",
                entry_price=round(entry_price, 2),
                target_price=round(target, 2),
                stop_loss=round(sl, 2),
                pattern_score=80,
                breakout_confirmed=(stage == PatternStage.BREAKOUT),
                neckline=round(neckline, 2),
                measured_move=round(pattern_height, 2),
                stage=stage,
                distance_to_trigger=round(distance_to_trigger, 2),
                confidence=round(confidence, 1),
                pivot_count=2
            )

    return None


def detect_ascending_triangle(df: pd.DataFrame, lookback: int = 40,
                               early_entry_pct: float = 0.02) -> Optional[Pattern]:
    atr = calculate_atr(df)
    df = find_pivots(df, left=5, right=5)
    recent = df.tail(lookback)
    if len(recent) < 20:
        return None

    pivot_highs = recent[recent["pivot_high"].notna()]
    pivot_lows = recent[recent["pivot_low"].notna()]

    if len(pivot_highs) < 3 or len(pivot_lows) < 2:
        return None

    current_price = df["close"].iloc[-1]

    resistance_level = pivot_highs["pivot_high"].max()
    tolerance = dynamic_tolerance(resistance_level, atr, base_pct=0.01)
    highs_at_resistance = pivot_highs[
        abs(pivot_highs["pivot_high"] - resistance_level) <= tolerance
    ]

    if len(highs_at_resistance) < 3:
        return None

    lows_values = pivot_lows["pivot_low"].values
    lows_slope = np.polyfit(range(len(lows_values)), lows_values, 1)[0]
    lows_rise = lows_values[-1] - lows_values[0]
    if lows_slope > 0 and lows_rise >= atr * 0.5:
        pattern_height = resistance_level - recent["low"].min()
        target = resistance_level + pattern_height
        
        distance_to_resistance = (resistance_level - current_price) / current_price
        
        if current_price >= resistance_level:
            stage = PatternStage.BREAKOUT
            entry_price = current_price
            sl = resistance_level - atr * 1.0
        elif distance_to_resistance <= early_entry_pct:
            stage = PatternStage.APPROACHING
            entry_price = resistance_level
            sl = recent["low"].min() - (atr * 0.5)
        else:
            stage = PatternStage.FORMING
            entry_price = resistance_level
            sl = recent["low"].min() - atr
        
        distance_to_trigger = abs(current_price - resistance_level) / atr
        
        confidence = calculate_pattern_confidence(
            pivot_count=len(highs_at_resistance),
            symmetry=0.7,
            volume_confirm=False,
            stage=stage
        )

        return Pattern(
            pattern_type=PatternType.ASCENDING_TRIANGLE,
            direction="bullish",
            reliability="high" if confidence >= 70 else "medium" if confidence >= 50 else "low",
            entry_price=round(entry_price, 2),
            target_price=round(target, 2),
            stop_loss=round(sl, 2),
            pattern_score=75,
            breakout_confirmed=(stage == PatternStage.BREAKOUT),
            neckline=round(resistance_level, 2),
            measured_move=round(pattern_height, 2),
            stage=stage,
            distance_to_trigger=round(distance_to_trigger, 2),
            confidence=round(confidence, 1),
            pivot_count=len(highs_at_resistance)
        )

    return None


def detect_descending_triangle(df: pd.DataFrame, lookback: int = 40,
                                early_entry_pct: float = 0.02) -> Optional[Pattern]:
    atr = calculate_atr(df)
    df = find_pivots(df, left=5, right=5)
    recent = df.tail(lookback)
    if len(recent) < 20:
        return None

    pivot_highs = recent[recent["pivot_high"].notna()]
    pivot_lows = recent[recent["pivot_low"].notna()]

    if len(pivot_highs) < 2 or len(pivot_lows) < 3:
        return None

    current_price = df["close"].iloc[-1]

    support_level = pivot_lows["pivot_low"].min()
    tolerance = dynamic_tolerance(support_level, atr, base_pct=0.01)
    lows_at_support = pivot_lows[
        abs(pivot_lows["pivot_low"] - support_level) <= tolerance
    ]

    if len(lows_at_support) < 3:
        return None

    highs_values = pivot_highs["pivot_high"].values
    highs_slope = np.polyfit(range(len(highs_values)), highs_values, 1)[0]
    highs_fall = highs_values[0] - highs_values[-1]
    if highs_slope < 0 and highs_fall >= atr * 0.5:
        pattern_height = recent["high"].max() - support_level
        target = support_level - pattern_height
        
        distance_to_support = (current_price - support_level) / current_price
        
        if current_price <= support_level:
            stage = PatternStage.BREAKOUT
            entry_price = current_price
            sl = support_level + atr * 1.0
        elif distance_to_support <= early_entry_pct:
            stage = PatternStage.APPROACHING
            entry_price = support_level
            sl = recent["high"].max() + (atr * 0.5)
        else:
            stage = PatternStage.FORMING
            entry_price = support_level
            sl = recent["high"].max() + atr
        
        distance_to_trigger = abs(current_price - support_level) / atr
        
        confidence = calculate_pattern_confidence(
            pivot_count=len(lows_at_support),
            symmetry=0.7,
            volume_confirm=False,
            stage=stage
        )

        return Pattern(
            pattern_type=PatternType.DESCENDING_TRIANGLE,
            direction="bearish",
            reliability="high" if confidence >= 70 else "medium" if confidence >= 50 else "low",
            entry_price=round(entry_price, 2),
            target_price=round(target, 2),
            stop_loss=round(sl, 2),
            pattern_score=75,
            breakout_confirmed=(stage == PatternStage.BREAKOUT),
            neckline=round(support_level, 2),
            measured_move=round(pattern_height, 2),
            stage=stage,
            distance_to_trigger=round(distance_to_trigger, 2),
            confidence=round(confidence, 1),
            pivot_count=len(lows_at_support)
        )

    return None


def detect_bull_flag(df: pd.DataFrame, lookback: int = 30) -> Optional[Pattern]:
    atr = calculate_atr(df)
    recent = df.tail(lookback)
    if len(recent) < 15:
        return None

    pole = recent.head(10)
    flag = recent.tail(15)

    pole_start = pole["close"].iloc[0]
    pole_end = pole["close"].iloc[-1]
    pole_move = (pole_end - pole_start) / pole_start

    flag_high = flag["high"].max()
    flag_low = flag["low"].min()
    flag_range = (flag_high - flag_low) / pole_end
    current_price = df["close"].iloc[-1]

    if pole_move > 0.03 and flag_range < pole_move * 0.5:
        pattern_height = pole_end - pole_start
        target = flag_high + pattern_height
        
        if current_price >= flag_high:
            stage = PatternStage.BREAKOUT
            entry_price = current_price
            sl = flag_low - (atr * 0.5)
        else:
            stage = PatternStage.FORMING
            entry_price = flag_high
            sl = flag_low - atr
        
        distance_to_trigger = abs(current_price - flag_high) / atr
        confidence = calculate_pattern_confidence(2, 0.8, False, stage)

        return Pattern(
            pattern_type=PatternType.BULL_FLAG,
            direction="bullish",
            reliability="high" if confidence >= 70 else "medium",
            entry_price=round(entry_price, 2),
            target_price=round(target, 2),
            stop_loss=round(sl, 2),
            pattern_score=70,
            breakout_confirmed=(stage == PatternStage.BREAKOUT),
            measured_move=round(pattern_height, 2),
            stage=stage,
            distance_to_trigger=round(distance_to_trigger, 2),
            confidence=round(confidence, 1),
            pivot_count=2
        )

    return None


def detect_bear_flag(df: pd.DataFrame, lookback: int = 30) -> Optional[Pattern]:
    atr = calculate_atr(df)
    recent = df.tail(lookback)
    if len(recent) < 15:
        return None

    pole = recent.head(10)
    flag = recent.tail(15)

    pole_start = pole["close"].iloc[0]
    pole_end = pole["close"].iloc[-1]
    pole_move = (pole_start - pole_end) / pole_start

    flag_high = flag["high"].max()
    flag_low = flag["low"].min()
    flag_range = (flag_high - flag_low) / pole_end
    current_price = df["close"].iloc[-1]

    if pole_move > 0.03 and flag_range < pole_move * 0.5:
        pattern_height = pole_start - pole_end
        target = flag_low - pattern_height
        
        if current_price <= flag_low:
            stage = PatternStage.BREAKOUT
            entry_price = current_price
            sl = flag_high + (atr * 0.5)
        else:
            stage = PatternStage.FORMING
            entry_price = flag_low
            sl = flag_high + atr
        
        distance_to_trigger = abs(current_price - flag_low) / atr
        confidence = calculate_pattern_confidence(2, 0.8, False, stage)

        return Pattern(
            pattern_type=PatternType.BEAR_FLAG,
            direction="bearish",
            reliability="high" if confidence >= 70 else "medium",
            entry_price=round(entry_price, 2),
            target_price=round(target, 2),
            stop_loss=round(sl, 2),
            pattern_score=70,
            breakout_confirmed=(stage == PatternStage.BREAKOUT),
            measured_move=round(pattern_height, 2),
            stage=stage,
            distance_to_trigger=round(distance_to_trigger, 2),
            confidence=round(confidence, 1),
            pivot_count=2
        )

    return None


def detect_rising_wedge(df: pd.DataFrame, lookback: int = 40) -> Optional[Pattern]:
    atr = calculate_atr(df)
    recent = df.tail(lookback)
    if len(recent) < 20:
        return None

    highs = recent["high"].values
    lows = recent["low"].values
    current_price = df["close"].iloc[-1]

    x = np.arange(len(highs))
    high_slope = np.polyfit(x, highs, 1)[0]
    low_slope = np.polyfit(x, lows, 1)[0]

    if high_slope > 0 and low_slope > 0 and low_slope > high_slope:
        convergence = (highs[-1] - lows[-1]) / (highs[0] - lows[0])
        if convergence < 0.7:
            pattern_height = max(highs) - min(lows)
            target = min(lows) - pattern_height
            
            lower_line = lows[-1]
            if current_price <= lower_line:
                stage = PatternStage.BREAKOUT
                entry_price = current_price
                sl = max(highs) + (atr * 1.0)
            else:
                stage = PatternStage.FORMING
                entry_price = lower_line
                sl = max(highs) + atr
            
            distance_to_trigger = abs(current_price - lower_line) / atr
            confidence = calculate_pattern_confidence(3, convergence, False, stage)

            return Pattern(
                pattern_type=PatternType.RISING_WEDGE,
                direction="bearish",
                reliability="medium",
                entry_price=round(entry_price, 2),
                target_price=round(target, 2),
                stop_loss=round(sl, 2),
                pattern_score=65,
                breakout_confirmed=(stage == PatternStage.BREAKOUT),
                measured_move=round(pattern_height, 2),
                stage=stage,
                distance_to_trigger=round(distance_to_trigger, 2),
                confidence=round(confidence, 1),
                pivot_count=3
            )

    return None


def detect_falling_wedge(df: pd.DataFrame, lookback: int = 40) -> Optional[Pattern]:
    atr = calculate_atr(df)
    recent = df.tail(lookback)
    if len(recent) < 20:
        return None

    highs = recent["high"].values
    lows = recent["low"].values
    current_price = df["close"].iloc[-1]

    x = np.arange(len(highs))
    high_slope = np.polyfit(x, highs, 1)[0]
    low_slope = np.polyfit(x, lows, 1)[0]

    if high_slope < 0 and low_slope < 0 and abs(high_slope) > abs(low_slope):
        convergence = (highs[-1] - lows[-1]) / (highs[0] - lows[0])
        if convergence < 0.7:
            pattern_height = max(highs) - min(lows)
            target = max(highs) + pattern_height
            
            upper_line = highs[-1]
            if current_price >= upper_line:
                stage = PatternStage.BREAKOUT
                entry_price = current_price
                sl = min(lows) - (atr * 1.0)
            else:
                stage = PatternStage.FORMING
                entry_price = upper_line
                sl = min(lows) - atr
            
            distance_to_trigger = abs(current_price - upper_line) / atr
            confidence = calculate_pattern_confidence(3, convergence, False, stage)

            return Pattern(
                pattern_type=PatternType.FALLING_WEDGE,
                direction="bullish",
                reliability="medium",
                entry_price=round(entry_price, 2),
                target_price=round(target, 2),
                stop_loss=round(sl, 2),
                pattern_score=70,
                breakout_confirmed=(stage == PatternStage.BREAKOUT),
                measured_move=round(pattern_height, 2),
                stage=stage,
                distance_to_trigger=round(distance_to_trigger, 2),
                confidence=round(confidence, 1),
                pivot_count=3
            )

    return None


def detect_cup_and_handle(df: pd.DataFrame, lookback: int = 80) -> Optional[Pattern]:
    atr = calculate_atr(df)
    if len(df) < lookback:
        return None

    recent = df.tail(lookback)
    cup = recent.head(50)
    handle = recent.tail(20)
    current_price = df["close"].iloc[-1]

    cup_high = cup["high"].max()
    cup_low = cup["low"].min()
    cup_depth = (cup_high - cup_low) / cup_high

    if 0.1 < cup_depth < 0.35:
        cup_mid = (cup_high + cup_low) / 2
        left_side = cup["close"].iloc[:10].mean()
        right_side = cup["close"].iloc[-10:].mean()
        rounded = abs(left_side - right_side) / cup_high < 0.05

        if rounded:
            handle_high = handle["high"].max()
            handle_low = handle["low"].min()
            handle_depth = (handle_high - handle_low) / handle_high

            if handle_depth < cup_depth * 0.5:
                pattern_height = cup_high - cup_low
                target = cup_high + pattern_height
                
                if current_price >= cup_high:
                    stage = PatternStage.BREAKOUT
                    entry_price = current_price
                    sl = handle_low - (atr * 0.5)
                else:
                    stage = PatternStage.FORMING
                    entry_price = cup_high
                    sl = handle_low - atr
                
                distance_to_trigger = abs(current_price - cup_high) / atr
                confidence = calculate_pattern_confidence(5, 0.8, False, stage)

                return Pattern(
                    pattern_type=PatternType.CUP_AND_HANDLE,
                    direction="bullish",
                    reliability="high" if confidence >= 70 else "medium",
                    entry_price=round(entry_price, 2),
                    target_price=round(target, 2),
                    stop_loss=round(sl, 2),
                    pattern_score=85,
                    breakout_confirmed=(stage == PatternStage.BREAKOUT),
                    neckline=round(cup_high, 2),
                    measured_move=round(pattern_height, 2),
                    stage=stage,
                    distance_to_trigger=round(distance_to_trigger, 2),
                    confidence=round(confidence, 1),
                    pivot_count=5
                )

    return None


def detect_all_patterns(df: pd.DataFrame, pattern_lookback: int = 100, 
                        min_pattern_bars: int = 20,
                        prefer_early: bool = True,
                        min_confidence: float = 50.0) -> Optional[Pattern]:
    patterns: List[Optional[Pattern]] = [
        detect_inverse_head_and_shoulders(df, pattern_lookback),
        detect_head_and_shoulders(df, pattern_lookback),
        detect_double_bottom(df, min_pattern_bars),
        detect_double_top(df, min_pattern_bars),
        detect_cup_and_handle(df, pattern_lookback),
        detect_ascending_triangle(df, min_pattern_bars),
        detect_descending_triangle(df, min_pattern_bars),
        detect_falling_wedge(df, min_pattern_bars),
        detect_rising_wedge(df, min_pattern_bars),
        detect_bull_flag(df, 30),
        detect_bear_flag(df, 30),
    ]

    valid_patterns = [p for p in patterns if p is not None and p.confidence >= min_confidence]

    if not valid_patterns:
        return None

    if prefer_early:
        forming_or_approaching = [p for p in valid_patterns 
                                  if p.stage in [PatternStage.APPROACHING, PatternStage.BREAKOUT]]
        if forming_or_approaching:
            return max(forming_or_approaching, key=lambda p: p.confidence)

    return max(valid_patterns, key=lambda p: (p.confidence, p.pattern_score))
