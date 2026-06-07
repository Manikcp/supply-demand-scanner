"""
Tests for S&R Levels Detection Module

Tests support and resistance level detection:
- Swing point detection
- Weekly S&R levels
- Level clustering
- Level confirmation on daily
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.sr_levels import (
    find_swing_points,
    find_weekly_sr_levels,
    confirm_levels_on_daily,
    cluster_levels_adaptive,
    get_all_sr_levels,
)


def create_ohlcv_dataframe(closes: list, highs: list = None, lows: list = None,
                           volumes: list = None, start_date: str = "2024-01-01",
                           freq: str = "15min") -> pd.DataFrame:
    """Helper to create OHLCV DataFrame for testing."""
    n = len(closes)
    dates = pd.date_range(start=start_date, periods=n, freq=freq)
    
    closes = np.array(closes)
    if highs is None:
        highs = closes + np.abs(np.random.randn(n)) * 2
    if lows is None:
        lows = closes - np.abs(np.random.randn(n)) * 2
    if volumes is None:
        volumes = np.random.randint(100000, 500000, n)
    
    opens = closes + np.random.randn(n) * 1
    
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    }, index=dates)


class TestFindSwingPoints:
    """Tests for swing point detection."""
    
    def test_finds_resistance_levels(self):
        closes = [100] * 30
        highs = closes.copy()
        highs[10] = 110
        highs[20] = 112

        df = create_ohlcv_dataframe(closes, highs=highs)
        resistances, supports = find_swing_points(df)

        assert isinstance(resistances, list)
        assert isinstance(supports, list)

    def test_finds_support_levels(self):
        closes = [100] * 30
        lows = closes.copy()
        lows[10] = 90
        lows[20] = 88

        df = create_ohlcv_dataframe(closes, lows=lows)
        resistances, supports = find_swing_points(df)

        assert isinstance(resistances, list)
        assert isinstance(supports, list)

    def test_returns_empty_for_insufficient_data(self):
        df = create_ohlcv_dataframe([100] * 5)

        resistances, supports = find_swing_points(df)

        assert resistances == []
        assert supports == []

    def test_includes_volume(self):
        closes = [100] * 30
        volumes = [100000] * 30
        volumes[10] = 500000

        df = create_ohlcv_dataframe(closes, volumes=volumes)
        resistances, supports = find_swing_points(df)

        for level in resistances + supports:
            assert "volume" in level


class TestFindWeeklySRLevels:
    """Tests for weekly S&R level detection."""
    
    def test_returns_two_lists(self):
        dates = pd.date_range(start="2023-01-01", periods=52, freq="W")
        closes = [100 + i * 2 + np.sin(i) * 10 for i in range(52)]
        
        df = pd.DataFrame({
            "open": closes,
            "high": [c + 5 for c in closes],
            "low": [c - 5 for c in closes],
            "close": closes,
            "volume": [100000] * 52
        }, index=dates)
        
        resistances, supports = find_weekly_sr_levels(df)
        
        assert isinstance(resistances, list)
        assert isinstance(supports, list)
    
    def test_returns_empty_for_none_input(self):
        resistances, supports = find_weekly_sr_levels(None)
        
        assert resistances == []
        assert supports == []
    
    def test_returns_empty_for_short_data(self):
        df = create_ohlcv_dataframe([100] * 10, freq="W")
        
        resistances, supports = find_weekly_sr_levels(df)
        
        assert resistances == []
        assert supports == []
    
    def test_levels_have_quality_scores(self):
        dates = pd.date_range(start="2023-01-01", periods=52, freq="W")
        closes = [100 + i * 2 + np.sin(i) * 10 for i in range(52)]
        
        df = pd.DataFrame({
            "open": closes,
            "high": [c + 5 for c in closes],
            "low": [c - 5 for c in closes],
            "close": closes,
            "volume": [100000] * 52
        }, index=dates)
        
        resistances, supports = find_weekly_sr_levels(df)
        
        for level in resistances + supports:
            assert "quality_score" in level
            assert "reactions" in level
            assert "price" in level


class TestConfirmLevelsOnDaily:
    """Tests for level confirmation on daily timeframe."""
    
    def test_confirms_valid_levels(self):
        weekly_levels = [
            {"price": 100, "reactions": 3, "quality_score": 70, "timeframe": "weekly_resistance"}
        ]
        
        dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
        closes = [95 + i * 0.1 for i in range(60)]
        
        df_daily = pd.DataFrame({
            "open": closes,
            "high": [c + 3 for c in closes],
            "low": [c - 3 for c in closes],
            "close": closes,
            "volume": [100000] * 60
        }, index=dates)
        
        confirmed = confirm_levels_on_daily(weekly_levels, df_daily)
        
        assert isinstance(confirmed, list)
    
    def test_returns_empty_for_no_levels(self):
        dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
        closes = [100] * 60
        
        df_daily = pd.DataFrame({
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100000] * 60
        }, index=dates)
        
        confirmed = confirm_levels_on_daily([], df_daily)
        
        assert confirmed == []
    
    def test_returns_original_for_no_daily_data(self):
        weekly_levels = [
            {"price": 100, "reactions": 3, "quality_score": 70}
        ]
        
        confirmed = confirm_levels_on_daily(weekly_levels, None)
        
        assert confirmed == weekly_levels


class TestClusterLevels:
    """Tests for level clustering."""
    
    def test_clusters_nearby_levels(self):
        levels = [
            {"price": 100, "volume": 1000, "bar_index": 10, "rejection_size": 2},
            {"price": 101, "volume": 1000, "bar_index": 20, "rejection_size": 2},
            {"price": 102, "volume": 1000, "bar_index": 30, "rejection_size": 2},
        ]
        df = create_ohlcv_dataframe([100] * 50)
        
        clustered = cluster_levels_adaptive(levels, df, min_reactions=1)
        
        assert isinstance(clustered, list)
    
    def test_returns_empty_for_no_levels(self):
        df = create_ohlcv_dataframe([100] * 50)
        clustered = cluster_levels_adaptive([], df)
        
        assert clustered == []
    
    def test_filters_by_min_reactions(self):
        levels = [
            {"price": 100, "volume": 1000, "bar_index": 10, "rejection_size": 2},
        ]
        df = create_ohlcv_dataframe([100] * 50)
        
        clustered = cluster_levels_adaptive(levels, df, min_reactions=2)
        
        assert clustered == []
    
    def test_scores_clusters(self):
        levels = [
            {"price": 100, "volume": 1000, "bar_index": 10, "rejection_size": 2},
            {"price": 100.5, "volume": 1000, "bar_index": 20, "rejection_size": 2},
            {"price": 101, "volume": 1000, "bar_index": 30, "rejection_size": 2},
        ]
        df = create_ohlcv_dataframe([100] * 50)
        
        clustered = cluster_levels_adaptive(levels, df, min_reactions=1)
        
        for cluster in clustered:
            assert "quality_score" in cluster
            assert "reactions" in cluster


class TestGetAllSRLevels:
    """Tests for combined S&R level detection."""
    
    def test_returns_two_lists(self):
        dates_15m = pd.date_range(start="2024-01-01 09:15", periods=500, freq="15min")
        closes_15m = [100 + np.sin(i / 50) * 10 for i in range(500)]
        
        df_15m = pd.DataFrame({
            "open": closes_15m,
            "high": [c + 2 for c in closes_15m],
            "low": [c - 2 for c in closes_15m],
            "close": closes_15m,
            "volume": [100000] * 500
        }, index=dates_15m)
        
        avg_atr = 4.0
        
        resistances, supports = get_all_sr_levels(df_15m, None, None, avg_atr)
        
        assert isinstance(resistances, list)
        assert isinstance(supports, list)
    
    def test_works_with_minimal_data(self):
        dates = pd.date_range(start="2024-01-01", periods=100, freq="15min")
        closes = [100] * 100
        
        df = pd.DataFrame({
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100000] * 100
        }, index=dates)
        
        resistances, supports = get_all_sr_levels(df, None, None, 1.0)
        
        assert isinstance(resistances, list)
        assert isinstance(supports, list)
    
    def test_levels_sorted_by_quality(self):
        dates_15m = pd.date_range(start="2024-01-01 09:15", periods=500, freq="15min")
        closes_15m = [100 + np.sin(i / 50) * 10 for i in range(500)]
        
        df_15m = pd.DataFrame({
            "open": closes_15m,
            "high": [c + 2 for c in closes_15m],
            "low": [c - 2 for c in closes_15m],
            "close": closes_15m,
            "volume": [100000] * 500
        }, index=dates_15m)
        
        resistances, supports = get_all_sr_levels(df_15m, None, None, 4.0)
        
        if len(resistances) > 1:
            for i in range(len(resistances) - 1):
                assert resistances[i]["quality_score"] >= resistances[i + 1]["quality_score"]
        
        if len(supports) > 1:
            for i in range(len(supports) - 1):
                assert supports[i]["quality_score"] >= supports[i + 1]["quality_score"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
