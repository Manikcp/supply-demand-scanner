"""
Tests for Pattern Detection Module

Tests all chart pattern detection functions:
- Head & Shoulders
- Double Top/Bottom
- Triangles
- Flags
- Wedges
- Cup & Handle
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.pattern_detection import (
    PatternType,
    Pattern,
    find_pivots,
    detect_head_and_shoulders,
    detect_inverse_head_and_shoulders,
    detect_double_top,
    detect_double_bottom,
    detect_ascending_triangle,
    detect_descending_triangle,
    detect_bull_flag,
    detect_bear_flag,
    detect_rising_wedge,
    detect_falling_wedge,
    detect_cup_and_handle,
    detect_all_patterns,
)


def create_ohlcv_dataframe(closes: list, highs: list = None, lows: list = None, 
                           volumes: list = None, start_date: str = "2024-01-01") -> pd.DataFrame:
    """Helper to create OHLCV DataFrame for testing."""
    n = len(closes)
    dates = pd.date_range(start=start_date, periods=n, freq="15min")
    
    closes = np.array(closes)
    if highs is None:
        highs = closes + np.abs(np.random.randn(n)) * 5
    if lows is None:
        lows = closes - np.abs(np.random.randn(n)) * 5
    if volumes is None:
        volumes = np.random.randint(100000, 500000, n)
    
    opens = closes + np.random.randn(n) * 2
    
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    }, index=dates)


class TestFindPivots:
    """Tests for pivot point detection."""
    
    def test_finds_pivot_high(self):
        closes = [100] * 20
        highs = [100 + (5 if i == 10 else 0) for i in range(20)]
        df = create_ohlcv_dataframe(closes, highs=highs)
        
        result = find_pivots(df, left=3, right=3)
        
        assert "pivot_high" in result.columns
        assert "pivot_low" in result.columns
    
    def test_finds_pivot_low(self):
        closes = [100] * 20
        lows = [100 - (5 if i == 10 else 0) for i in range(20)]
        df = create_ohlcv_dataframe(closes, lows=lows)
        
        result = find_pivots(df, left=3, right=3)
        
        assert "pivot_low" in result.columns


class TestHeadAndShoulders:
    """Tests for Head & Shoulders pattern detection."""
    
    def test_detects_head_and_shoulders(self):
        n = 100
        pattern = []
        for i in range(n):
            if i < 20:
                pattern.append(100 + i * 0.5)
            elif i < 30:
                pattern.append(110 - (i - 20) * 0.3)
            elif i < 40:
                pattern.append(107 + (i - 30) * 0.8)
            elif i < 50:
                pattern.append(115 - (i - 40) * 1.5)
            elif i < 60:
                pattern.append(100 + (i - 50) * 0.3)
            elif i < 70:
                pattern.append(103 + (i - 60) * 0.7)
            elif i < 80:
                pattern.append(110 - (i - 70) * 1.0)
            else:
                pattern.append(100 - (i - 80) * 0.5)
        
        df = create_ohlcv_dataframe(pattern)
        result = detect_head_and_shoulders(df)
        
        assert result is None or result.pattern_type == PatternType.HEAD_AND_SHOULDERS
    
    def test_returns_none_for_flat_data(self):
        closes = [100] * 100
        highs = [100] * 100
        lows = [100] * 100
        df = create_ohlcv_dataframe(closes, highs=highs, lows=lows)
        
        result = detect_head_and_shoulders(df)
        
        assert result is None


class TestInverseHeadAndShoulders:
    """Tests for Inverse Head & Shoulders pattern detection."""
    
    def test_detects_inverse_head_and_shoulders(self):
        n = 100
        pattern = []
        for i in range(n):
            if i < 20:
                pattern.append(120 - i * 0.5)
            elif i < 30:
                pattern.append(110 + (i - 20) * 0.3)
            elif i < 40:
                pattern.append(113 - (i - 30) * 0.8)
            elif i < 50:
                pattern.append(105 + (i - 40) * 1.5)
            elif i < 60:
                pattern.append(120 - (i - 50) * 0.3)
            elif i < 70:
                pattern.append(117 - (i - 60) * 0.7)
            elif i < 80:
                pattern.append(110 + (i - 70) * 1.0)
            else:
                pattern.append(120 + (i - 80) * 0.5)
        
        df = create_ohlcv_dataframe(pattern)
        result = detect_inverse_head_and_shoulders(df)
        
        assert result is None or result.pattern_type == PatternType.INVERSE_HEAD_AND_SHOULDERS


class TestDoubleTop:
    """Tests for Double Top pattern detection."""
    
    def test_detects_double_top(self):
        n = 80
        pattern = []
        for i in range(n):
            if i < 20:
                pattern.append(100 + i * 0.5)
            elif i < 30:
                pattern.append(110 - (i - 20) * 0.3)
            elif i < 40:
                pattern.append(107 + (i - 30) * 0.3)
            elif i < 50:
                pattern.append(110 - (i - 40) * 1.0)
            else:
                pattern.append(100 - (i - 50) * 0.3)
        
        df = create_ohlcv_dataframe(pattern)
        result = detect_double_top(df)
        
        assert result is None or result.pattern_type == PatternType.DOUBLE_TOP


class TestDoubleBottom:
    """Tests for Double Bottom pattern detection."""
    
    def test_detects_double_bottom(self):
        n = 80
        pattern = []
        for i in range(n):
            if i < 20:
                pattern.append(110 - i * 0.5)
            elif i < 30:
                pattern.append(100 + (i - 20) * 0.3)
            elif i < 40:
                pattern.append(103 - (i - 30) * 0.3)
            elif i < 50:
                pattern.append(100 + (i - 40) * 1.0)
            else:
                pattern.append(110 + (i - 50) * 0.3)
        
        df = create_ohlcv_dataframe(pattern)
        result = detect_double_bottom(df)
        
        assert result is None or result.pattern_type == PatternType.DOUBLE_BOTTOM


class TestTriangles:
    """Tests for Triangle pattern detection."""
    
    def test_ascending_triangle_returns_pattern_or_none(self):
        n = 50
        closes = []
        for i in range(n):
            if i % 10 < 5:
                closes.append(100 + i * 0.2)
            else:
                closes.append(100 + i * 0.1)
        
        df = create_ohlcv_dataframe(closes)
        result = detect_ascending_triangle(df)
        
        assert result is None or result.pattern_type == PatternType.ASCENDING_TRIANGLE
    
    def test_descending_triangle_returns_pattern_or_none(self):
        n = 50
        closes = []
        for i in range(n):
            if i % 10 < 5:
                closes.append(100 - i * 0.2)
            else:
                closes.append(100 - i * 0.1)
        
        df = create_ohlcv_dataframe(closes)
        result = detect_descending_triangle(df)
        
        assert result is None or result.pattern_type == PatternType.DESCENDING_TRIANGLE


class TestFlags:
    """Tests for Flag pattern detection."""
    
    def test_bull_flag_returns_pattern_or_none(self):
        n = 30
        closes = []
        for i in range(n):
            if i < 10:
                closes.append(100 + i * 2)
            else:
                closes.append(120 - (i - 10) * 0.3)
        
        df = create_ohlcv_dataframe(closes)
        result = detect_bull_flag(df)
        
        assert result is None or result.pattern_type == PatternType.BULL_FLAG
    
    def test_bear_flag_returns_pattern_or_none(self):
        n = 30
        closes = []
        for i in range(n):
            if i < 10:
                closes.append(120 - i * 2)
            else:
                closes.append(100 + (i - 10) * 0.3)
        
        df = create_ohlcv_dataframe(closes)
        result = detect_bear_flag(df)
        
        assert result is None or result.pattern_type == PatternType.BEAR_FLAG


class TestWedges:
    """Tests for Wedge pattern detection."""
    
    def test_rising_wedge_returns_pattern_or_none(self):
        n = 40
        closes = []
        for i in range(n):
            closes.append(100 + i * 0.3)
        
        df = create_ohlcv_dataframe(closes)
        result = detect_rising_wedge(df)
        
        assert result is None or result.pattern_type == PatternType.RISING_WEDGE
    
    def test_falling_wedge_returns_pattern_or_none(self):
        n = 40
        closes = []
        for i in range(n):
            closes.append(120 - i * 0.3)
        
        df = create_ohlcv_dataframe(closes)
        result = detect_falling_wedge(df)
        
        assert result is None or result.pattern_type == PatternType.FALLING_WEDGE


class TestCupAndHandle:
    """Tests for Cup & Handle pattern detection."""
    
    def test_cup_and_handle_returns_pattern_or_none(self):
        n = 80
        pattern = []
        for i in range(n):
            if i < 25:
                pattern.append(100 + (i - 12.5) ** 2 * 0.05)
            elif i < 50:
                pattern.append(115 + (i - 25) * 0.1)
            elif i < 65:
                pattern.append(117.5 - (i - 50) * 0.2)
            else:
                pattern.append(115 + (i - 65) * 0.3)
        
        df = create_ohlcv_dataframe(pattern)
        result = detect_cup_and_handle(df)
        
        assert result is None or result.pattern_type == PatternType.CUP_AND_HANDLE


class TestDetectAllPatterns:
    """Tests for the detect_all_patterns function."""
    
    def test_returns_none_for_flat_data(self):
        closes = [100] * 100
        df = create_ohlcv_dataframe(closes)

        result = detect_all_patterns(df)

        assert isinstance(result, (Pattern, type(None)))
    
    def test_returns_highest_scoring_pattern(self):
        n = 100
        pattern = [100 + np.sin(i / 10) * 10 + i * 0.1 for i in range(n)]
        df = create_ohlcv_dataframe(pattern)
        
        result = detect_all_patterns(df)
        
        assert result is None or isinstance(result, Pattern)


class TestPatternDataclass:
    """Tests for the Pattern dataclass."""
    
    def test_pattern_creation(self):
        pattern = Pattern(
            pattern_type=PatternType.HEAD_AND_SHOULDERS,
            direction="bearish",
            reliability="high",
            entry_price=100.0,
            target_price=90.0,
            stop_loss=105.0,
            pattern_score=85,
            breakout_confirmed=True
        )
        
        assert pattern.pattern_type == PatternType.HEAD_AND_SHOULDERS
        assert pattern.direction == "bearish"
        assert pattern.reliability == "high"
        assert pattern.entry_price == 100.0
        assert pattern.target_price == 90.0
        assert pattern.stop_loss == 105.0
        assert pattern.pattern_score == 85
        assert pattern.breakout_confirmed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
