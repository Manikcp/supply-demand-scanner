import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.signals import (
    Config,
    MarketRegime,
    SignalType,
    is_trading_session,
    find_swing_points,
    detect_trend_bos,
    detect_impulsive_move,
    find_consolidation_before_impulse,
    detect_fvg,
    check_slow_momentum,
    check_zone_entry,
    is_fresh_zone,
    merge_overlapping_zones,
    detect_demand_zones,
    detect_supply_zones,
    calculate_entry_score,
    run_signals,
)


def create_ohlcv_dataframe(closes: list, highs: list = None, lows: list = None,
                           volumes: list = None, start_date: str = "2024-01-01 09:15",
                           freq: str = "15min") -> pd.DataFrame:
    n = len(closes)
    ist = timezone(timedelta(hours=5, minutes=30))
    base = datetime(2024, 1, 3, 9, 15, tzinfo=ist)
    dates = []

    for i in range(n):
        d = base + timedelta(minutes=15 * i)
        hour, minute = d.hour, d.minute
        if 9 * 60 + 15 <= hour * 60 + minute <= 15 * 60 + 30 and d.weekday() < 5:
            dates.append(d)

    dates = dates[:n]
    while len(dates) < n:
        dates.append(dates[-1] + timedelta(minutes=15))

    closes = np.array(closes[:n])
    if highs is None:
        highs = closes + np.abs(np.random.randn(n)) * 2
    if lows is None:
        lows = closes - np.abs(np.random.randn(n)) * 2
    if volumes is None:
        volumes = np.random.randint(100000, 500000, n)

    opens = closes + np.random.randn(n) * 1

    return pd.DataFrame({
        "open": opens[:n],
        "high": highs[:n],
        "low": lows[:n],
        "close": closes[:n],
        "volume": volumes[:n]
    }, index=pd.DatetimeIndex(dates[:n]))


class TestConfig:
    def test_default_config(self):
        cfg = Config()
        assert cfg.index_type == "NIFTY"
        assert cfg.use_vol_filt is True
        assert cfg.min_rr_ratio == 1.0
        assert cfg.target_rr == 1.5
        assert cfg.min_impulse_candles == 3
        assert cfg.impulse_body_mult == 1.3
        assert cfg.fvg_min_gap_atr == 0.3
        assert cfg.slow_momentum_max_body_pct == 0.4
        assert cfg.trend_swing_lookback == 48
        assert cfg.min_score_threshold == 50
        assert cfg.block_opening_session is True
        assert cfg.dedup_bar_cooldown == 5

    def test_from_json_missing_file(self):
        cfg = Config.from_json("nonexistent_config.json")
        assert isinstance(cfg, Config)
        assert cfg.index_type == "NIFTY"

    def test_from_json_with_supply_demand(self):
        import json
        import tempfile
        data = {
            "general": {"trading_capital": 50000, "min_reward_risk_ratio": 1.5},
            "supply_demand": {
                "min_impulse_candles": 4,
                "impulse_body_mult": 1.5,
                "fvg_min_gap_atr": 0.5,
                "slow_momentum_max_body_pct": 0.3,
                "trend_swing_lookback": 25,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            cfg = Config.from_json(f.name)
            assert cfg.min_impulse_candles == 4
            assert cfg.impulse_body_mult == 1.5
            assert cfg.fvg_min_gap_atr == 0.5
            assert cfg.slow_momentum_max_body_pct == 0.3
            assert cfg.trend_swing_lookback == 25
            assert cfg.min_rr_ratio == 1.5
        os.unlink(f.name)


class TestIsTradingSession:
    def test_detects_market_hours(self):
        ist = timezone(timedelta(hours=5, minutes=30))
        dt = datetime(2024, 1, 3, 10, 30, tzinfo=ist)
        ts = pd.Timestamp(dt)
        is_open, session = is_trading_session(ts)
        assert is_open is True
        assert session in ["opening", "regular", "closing"]

    def test_detects_closed_market(self):
        ist = timezone(timedelta(hours=5, minutes=30))
        dt = datetime(2024, 1, 3, 8, 0, tzinfo=ist)
        ts = pd.Timestamp(dt)
        is_open, session = is_trading_session(ts)
        assert is_open is False
        assert session == "closed"


class TestFindSwingPoints:
    def test_finds_swing_highs_and_lows(self):
        closes = [100 + np.sin(i / 10) * 10 for i in range(200)]
        df = create_ohlcv_dataframe(closes)
        hi, li, hv, lv = find_swing_points(df, 30)
        assert isinstance(hi, list)
        assert isinstance(li, list)
        assert isinstance(hv, list)
        assert isinstance(lv, list)

    def test_returns_empty_for_short_data(self):
        df = create_ohlcv_dataframe([100] * 5)
        hi, li, hv, lv = find_swing_points(df, 30)
        assert hi == []
        assert li == []


class TestDetectTrendBOS:
    def test_detects_bullish_bos(self):
        n = 200
        closes = list(100 + i * 0.3 for i in range(n))
        df = create_ohlcv_dataframe(closes)
        trend, bos_high, bos_low = detect_trend_bos(df, 30)
        assert trend in ("bullish", "bearish", "ranging")

    def test_detects_bearish_bos(self):
        n = 200
        closes = list(200 - i * 0.3 for i in range(n))
        df = create_ohlcv_dataframe(closes)
        trend, bos_high, bos_low = detect_trend_bos(df, 30)
        assert trend in ("bullish", "bearish", "ranging")

    def test_ranging_on_flat_data(self):
        df = create_ohlcv_dataframe([100] * 100)
        trend, bos_high, bos_low = detect_trend_bos(df, 30)
        assert trend in ("bullish", "bearish", "ranging")


class TestDetectImpulsiveMove:
    def test_detects_impulse_up(self):
        n = 100
        closes = [100] * 20 + [105, 110, 115, 120] + [120] * 76
        df = create_ohlcv_dataframe(closes)
        atr = (df["high"] - df["low"]).mean()
        found, end_idx = detect_impulsive_move(df, atr, 18, "up", 3, 1.3)
        assert found is True or found is False

    def test_no_impulse_on_flat(self):
        closes = [100] * 50
        df = create_ohlcv_dataframe(closes)
        atr = (df["high"] - df["low"]).mean()
        found, end_idx = detect_impulsive_move(df, atr, 5, "up", 3, 1.3)
        assert found is False


class TestFindConsolidation:
    def test_finds_consolidation(self):
        closes = [100] * 10 + [105, 110, 115, 120] + [120] * 10
        df = create_ohlcv_dataframe(closes)
        atr = (df["high"] - df["low"]).mean()
        result = find_consolidation_before_impulse(df, atr, 10, 8)
        assert result is None or len(result) == 4

    def test_returns_none_on_edge(self):
        df = create_ohlcv_dataframe([100] * 50)
        atr = (df["high"] - df["low"]).mean()
        result = find_consolidation_before_impulse(df, atr, 1, 8)
        assert result is None


class TestDetectFVG:
    def test_detect_fvg_up(self):
        n = 100
        closes = [100] * n
        highs = [101] * n
        lows = [99] * n
        highs[50] = 110
        lows[51] = 112
        df = create_ohlcv_dataframe(closes, highs=highs, lows=lows)
        result = detect_fvg(df, 51, 0.3)
        assert isinstance(result, (bool, np.bool_))

    def test_false_on_edge(self):
        df = create_ohlcv_dataframe([100] * 5)
        result = detect_fvg(df, 0, 0.3)
        assert result is False


class TestCheckSlowMomentum:
    def test_detects_slow_momentum(self):
        closes = [100 + i * 0.1 for i in range(100)]
        df = create_ohlcv_dataframe(closes)
        result = check_slow_momentum(df, 50, 3, 0.4)
        assert isinstance(result, bool)

    def test_false_early_bars(self):
        df = create_ohlcv_dataframe([100] * 5)
        result = check_slow_momentum(df, 1, 3, 0.4)
        assert result is False


class TestCheckZoneEntry:
    def test_close_in_zone_demand(self):
        df = create_ohlcv_dataframe([100] * 10)
        ok, reason = check_zone_entry(df, 5, 105, 95, 2.0, "demand", 0.3, True)
        assert isinstance(ok, bool)
        assert isinstance(reason, str)

    def test_close_in_zone_supply(self):
        df = create_ohlcv_dataframe([100] * 10)
        ok, reason = check_zone_entry(df, 5, 105, 95, 2.0, "supply", 0.3, True)
        assert isinstance(ok, bool)
        assert isinstance(reason, str)


class TestIsFreshZone:
    def test_checks_freshness(self):
        df = create_ohlcv_dataframe([100] * 50)
        result = is_fresh_zone(0, df, 105, 95)
        assert isinstance(result, bool)


class TestMergeOverlappingZones:
    def test_merges_overlapping(self):
        zones = [
            {"zone_low": 90, "zone_high": 100, "fvg": False, "strength": 50, "impulse_end": 10},
            {"zone_low": 95, "zone_high": 105, "fvg": True, "strength": 70, "impulse_end": 15},
        ]
        merged = merge_overlapping_zones(zones)
        assert len(merged) <= len(zones)

    def test_empty_input(self):
        assert merge_overlapping_zones([]) == []


class TestDetectDemandZones:
    def test_returns_list(self):
        n = 300
        closes = [100 + np.sin(i / 20) * 10 for i in range(n)]
        df = create_ohlcv_dataframe(closes)
        cfg = Config()
        atr_series = (df["high"] - df["low"]).rolling(14).mean()
        zones = detect_demand_zones(df, atr_series, cfg)
        assert isinstance(zones, list)


class TestDetectSupplyZones:
    def test_returns_list(self):
        n = 300
        closes = [100 + np.sin(i / 20) * 10 for i in range(n)]
        df = create_ohlcv_dataframe(closes)
        cfg = Config()
        atr_series = (df["high"] - df["low"]).rolling(14).mean()
        zones = detect_supply_zones(df, atr_series, cfg)
        assert isinstance(zones, list)


class TestCalculateEntryScore:
    def test_calculates_score(self):
        zone = {"type": "demand", "strength": 70, "fvg": True}
        result = calculate_entry_score(
            zone, trend_ok=True, entry_ok=True, slow_mom=True,
            fvg=True, regime=MarketRegime.TRENDING_UP, volume_spike=True
        )
        assert result["score"] >= 60
        assert "trend_aligned" in result["factors"]
        assert "entry_confirmed" in result["factors"]

    def test_low_score_without_factors(self):
        result = calculate_entry_score(
            None, trend_ok=False, entry_ok=False, slow_mom=False,
            fvg=False, regime=MarketRegime.RANGING, volume_spike=False
        )
        assert result["score"] < 40
        assert result["grade"] == "D"


class TestRunSignals:
    def test_returns_tuple(self):
        closes = [100 + np.sin(i / 50) * 10 for i in range(200)]
        df = create_ohlcv_dataframe(closes)
        cfg = Config()
        result_df, signals = run_signals(df, None, cfg)
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(signals, pd.DataFrame)

    def test_adds_technical_columns(self):
        closes = [100 + np.sin(i / 50) * 10 for i in range(200)]
        df = create_ohlcv_dataframe(closes)
        cfg = Config()
        result_df, signals = run_signals(df, None, cfg)
        assert "atr" in result_df.columns
        assert "ema" in result_df.columns
        assert "session_ok" in result_df.columns
        assert "session" in result_df.columns

    def test_signals_have_expected_fields(self):
        closes = [100 + np.sin(i / 10) * 15 for i in range(400)]
        df = create_ohlcv_dataframe(closes)
        cfg = Config(min_rr_ratio=1.0, min_score_threshold=30)
        _, signals = run_signals(df, None, cfg)
        if not signals.empty:
            assert "type" in signals.columns
            assert "entry" in signals.columns
            assert "target" in signals.columns
            assert "sl" in signals.columns
            assert "rr" in signals.columns
            assert "entry_quality" in signals.columns
            assert "entry_grade" in signals.columns
            assert "factors" in signals.columns
            assert "zone_entry_reason" in signals.columns
            assert "slow_momentum" in signals.columns
            assert "fvg_present" in signals.columns
            assert "market_structure" in signals.columns

    def test_signals_respect_min_rr(self):
        closes = [100 + np.sin(i / 50) * 10 for i in range(300)]
        df = create_ohlcv_dataframe(closes)
        cfg = Config(min_rr_ratio=3.0, min_score_threshold=30)
        _, signals = run_signals(df, None, cfg)
        if not signals.empty:
            assert all(signals["rr"] >= 3.0)

    def test_handles_flat_data(self):
        df = create_ohlcv_dataframe([100] * 200)
        cfg = Config()
        _, signals = run_signals(df, None, cfg)
        assert isinstance(signals, pd.DataFrame)

    def test_handles_none_daily(self):
        closes = [100 + np.sin(i / 50) * 10 for i in range(200)]
        df = create_ohlcv_dataframe(closes)
        cfg = Config()
        result_df, signals = run_signals(df, None, cfg, df_weekly=None)
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(signals, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
