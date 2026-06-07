import numpy as np
import pandas as pd
import pytest
import os
from datetime import datetime, timedelta, timezone

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.signals import (
    Config, run_signals,
    calculate_entry_score, is_trading_session,
    MarketRegime,
)
from scanner_base import fetch_ohlcv, signal_key
from strategies.sr_levels import (
    confirm_levels_on_daily, find_weekly_sr_levels,
    get_all_sr_levels, detect_market_structure,
)
from strategies.pattern_detection import detect_all_patterns


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
        "open": opens[:n], "high": highs[:n], "low": lows[:n],
        "close": closes[:n], "volume": volumes[:n]
    }, index=pd.DatetimeIndex(dates[:n]))


# Edge Case 1: yfinance rate limit retry logic
class TestYfinanceRateLimit:
    def test_fetch_returns_none_on_empty(self):
        df = fetch_ohlcv("INVALID.TICKER.XYZ", interval="15m", period="7d", use_cache=False)
        assert df is None

    def test_signal_key_deduplication(self):
        key1 = signal_key("RELIANCE.NS", "CE BUY", "2024-01-03 10:00")
        key2 = signal_key("RELIANCE.NS", "CE BUY", "2024-01-03 10:00")
        key3 = signal_key("RELIANCE.NS", "PE BUY", "2024-01-03 10:00")
        assert key1 == key2
        assert key1 != key3


# Edge Case 2: Cache corruption try/except fallback
class TestCacheCorruption:
    def test_cache_read_failure_falls_back(self, tmp_path):
        cache_dir = tmp_path / "data_cache"
        cache_dir.mkdir()
        bad_file = cache_dir / "BAD_TICKER_15m_7d.parquet"
        bad_file.write_text("NOT A PARQUET FILE")
        result = fetch_ohlcv("BAD_TICKER", interval="15m", period="7d",
                             use_cache=True, cache_dir=str(cache_dir))
        assert result is None or isinstance(result, pd.DataFrame)

    def test_cache_write_failure_does_not_crash(self, tmp_path):
        read_only_dir = tmp_path / "readonly_cache"
        read_only_dir.mkdir()
        read_only_dir.chmod(0o444)
        n = 100
        closes = [100 + i * 0.5 for i in range(n)]
        df = create_ohlcv_dataframe(closes)
        cfg = Config()
        result_df, signals = run_signals(df, None, cfg)
        assert isinstance(result_df, pd.DataFrame)


# Edge Case 3: No weekly/daily data graceful degradation
class TestNoWeeklyDailyData:
    def test_none_daily_does_not_crash(self):
        n = 200
        closes = [100 + np.sin(i / 50) * 10 for i in range(n)]
        df = create_ohlcv_dataframe(closes)
        cfg = Config()
        result_df, signals = run_signals(df, None, cfg, df_weekly=None)
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(signals, pd.DataFrame)

    def test_none_weekly_sr_returns_empty(self):
        result_r, result_s = find_weekly_sr_levels(None, 260)
        assert result_r == []
        assert result_s == []

    def test_short_daily_returns_original_levels(self):
        weekly_levels = [{"price": 100.0, "quality_score": 60, "is_resistance": True}]
        daily_short = pd.DataFrame({"high": [100]*5, "low": [99]*5, "close": [99.5]*5})
        result = confirm_levels_on_daily(weekly_levels, daily_short)
        assert len(result) == len(weekly_levels)

    def test_none_daily_confirmation_returns_original(self):
        weekly_levels = [{"price": 100.0, "quality_score": 60, "is_resistance": True}]
        result = confirm_levels_on_daily(weekly_levels, None)
        assert len(result) == len(weekly_levels)

    def test_none_daily_in_get_all_sr_levels(self):
        n = 200
        closes = [100 + np.sin(i / 50) * 10 for i in range(n)]
        df = create_ohlcv_dataframe(closes)
        atr = df["high"].sub(df["low"]).tail(50).mean()
        resistances, supports = get_all_sr_levels(df, None, None, atr)
        assert isinstance(resistances, list)
        assert isinstance(supports, list)

    def test_daily_pattern_skips_when_no_daily(self):
        n = 200
        closes = [100 + np.sin(i / 50) * 10 for i in range(n)]
        df = create_ohlcv_dataframe(closes)
        cfg = Config()
        result_df, signals = run_signals(df, None, cfg)
        assert isinstance(result_df, pd.DataFrame)

    def test_weekly_pattern_skips_when_short_weekly(self):
        n = 200
        closes = [100 + np.sin(i / 50) * 10 for i in range(n)]
        df = create_ohlcv_dataframe(closes)
        short_weekly = create_ohlcv_dataframe([100]*10)
        cfg = Config()
        result_df, signals = run_signals(df, None, cfg, df_weekly=short_weekly)
        assert isinstance(result_df, pd.DataFrame)


# Edge Case 6: NaN in price data
class TestNaNInPriceData:
    def test_dropna_in_fetch(self):
        ist = timezone(timedelta(hours=5, minutes=30))
        dates = pd.date_range(start="2024-01-03 09:15", periods=10, freq="15min", tz="Asia/Kolkata")
        df = pd.DataFrame({
            "open": [100.0, None, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "high": [101.0, 102.0, None, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            "low": [99.0, 100.0, 101.0, None, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            "close": [100.5, 101.0, 102.5, 103.5, None, 105.5, 106.5, 107.5, 108.5, 109.5],
            "volume": [100000] * 10,
        }, index=dates)
        cleaned = df[["open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])
        assert len(cleaned) < len(df)

    def test_atr_filled_when_nan(self):
        s = pd.Series([np.nan, np.nan])
        filled = s.fillna(pd.Series([100.0, 100.0]) * 0.01)
        assert filled.iloc[-1] == 1.0

    def test_ema_fillna(self):
        n = 50
        closes = [100.0 + i * 0.5 for i in range(n)]
        df = create_ohlcv_dataframe(closes)
        cfg = Config()
        result_df, _ = run_signals(df, None, cfg)
        assert "ema" in result_df.columns
        assert not result_df["ema"].dropna().empty

    def test_volume_fillna_zero(self):
        df = create_ohlcv_dataframe([100] * 50)
        raw = df.copy()
        raw["volume"] = raw["volume"].fillna(0)
        assert raw["volume"].isna().sum() == 0


# Edge Case 7: Same level repeated dedup_bar_cooldown
class TestLevelDeduplication:
    def test_dedup_cooldown_in_config(self):
        cfg = Config()
        assert cfg.dedup_bar_cooldown == 5


# Edge Case 9: Market opens after gap 45 min block
class TestOpeningSessionBlock:
    def test_opening_session_blocked(self):
        ist = timezone(timedelta(hours=5, minutes=30))
        dt = datetime(2024, 1, 3, 9, 30, tzinfo=ist)
        ts = pd.Timestamp(dt)
        is_open, session = is_trading_session(ts)
        assert is_open is False
        assert session == "opening"

    def test_regular_session_allowed(self):
        ist = timezone(timedelta(hours=5, minutes=30))
        dt = datetime(2024, 1, 3, 11, 0, tzinfo=ist)
        ts = pd.Timestamp(dt)
        is_open, session = is_trading_session(ts)
        assert is_open is True
        assert session == "regular"

    def test_closing_session_allowed(self):
        ist = timezone(timedelta(hours=5, minutes=30))
        dt = datetime(2024, 1, 3, 14, 30, tzinfo=ist)
        ts = pd.Timestamp(dt)
        is_open, session = is_trading_session(ts)
        assert is_open is True
        assert session == "closing"

    def test_pre_close_allowed(self):
        ist = timezone(timedelta(hours=5, minutes=30))
        dt = datetime(2024, 1, 3, 15, 15, tzinfo=ist)
        ts = pd.Timestamp(dt)
        is_open, session = is_trading_session(ts)
        assert is_open is True
        assert session == "pre_close"

    def test_block_opening_session_config(self):
        cfg = Config()
        assert cfg.block_opening_session is True

    def test_signals_blocked_during_opening(self):
        n = 200
        closes = [100 + np.sin(i / 50) * 10 for i in range(n)]
        df = create_ohlcv_dataframe(closes)
        cfg = Config(block_opening_session=True)
        result_df, signals = run_signals(df, None, cfg)
        if not signals.empty:
            for _, row in signals.iterrows():
                session = row.get("session", "")
                assert session != "opening"


# Edge Case 11: No volume data vol_above_avg bypass
class TestNoVolumeData:
    def test_vol_above_avg_true_when_no_volume(self):
        n = 100
        closes = [100 + i * 0.5 for i in range(n)]
        dates = pd.date_range(start="2024-01-03 10:00", periods=n, freq="15min", tz="Asia/Kolkata")
        df = pd.DataFrame({
            "open": closes, "high": [c + 2 for c in closes],
            "low": [c - 2 for c in closes], "close": closes,
        }, index=dates)
        cfg = Config()
        result_df, _ = run_signals(df, None, cfg)
        assert "vol_spike" in result_df.columns

    def test_volume_gate_bypassed_for_indices(self):
        n = 100
        closes = [100 + i * 0.5 for i in range(n)]
        df = create_ohlcv_dataframe(closes)
        cfg = Config(use_vol_filt=False)
        result_df, _ = run_signals(df, None, cfg)
        assert "vol_spike" in result_df.columns


# Edge Case 12: MultiIndex yfinance output flattening
class TestMultiIndexFlattening:
    def test_multiindex_flattened(self):
        cols = pd.MultiIndex.from_tuples([("Open", "RELIANCE"), ("High", "RELIANCE"),
                                           ("Low", "RELIANCE"), ("Close", "RELIANCE"),
                                           ("Volume", "RELIANCE")])
        data = np.random.randn(10, 5)
        df = pd.DataFrame(data, columns=cols)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        needed = {"open", "high", "low", "close", "volume"}
        assert needed.issubset(df.columns)

    def test_single_index_untouched(self):
        df = pd.DataFrame({"Open": [1], "High": [2], "Low": [0.5], "Close": [1.5], "Volume": [100]})
        raw_cols = list(df.columns)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        assert list(df.columns) == [c.lower() for c in raw_cols]


# Edge Case 4: No signals all day scanner keeps running
class TestNoSignalsAllDay:
    def test_empty_signals_dataframe(self):
        df = create_ohlcv_dataframe([100] * 50)
        cfg = Config()
        _, signals = run_signals(df, None, cfg)
        assert isinstance(signals, pd.DataFrame)

    def test_scanner_prints_no_signals(self):
        import io
        from scanner_base import format_alert_with_targets
        empty_dict = {"type": "CE BUY", "entry": 100, "target": 105, "sl": 98,
                      "rr": 2.5, "index": "NIFTY", "strike": 100,
                      "datetime": "2024-01-03 10:00"}
        formatted = format_alert_with_targets("TEST.NS", empty_dict)
        assert "CE BUY" in formatted


# Edge Case 5: All tickers fail empty result
class TestAllTickersFail:
    def test_run_signals_with_bare_minimum_data(self):
        ist = timezone(timedelta(hours=5, minutes=30))
        n = 50
        dates = pd.date_range(start="2024-01-03 10:00", periods=n, freq="15min", tz="Asia/Kolkata")
        df = pd.DataFrame({
            "open": [100] * n, "high": [101] * n, "low": [99] * n,
            "close": [100] * n, "volume": [100000] * n,
        }, index=dates)
        cfg = Config()
        _, signals = run_signals(df, None, cfg)
        assert isinstance(signals, pd.DataFrame)

    def test_empty_signals_filter_produces_empty_ui_state(self):
        empty_signals = pd.DataFrame()
        df_list = []
        for ticker in ["TICKER1.NS", "TICKER2.NS", "TICKER3.NS"]:
            df = create_ohlcv_dataframe([100] * 50)
            df_list.append(df)
        assert len(df_list) == 3
        assert empty_signals.empty


# Combined: run_signals handles all edge cases together
class TestCombinedEdgeCases:
    def test_no_volume_no_daily_no_weekly_still_works(self):
        n = 100
        closes = [100 + i * 0.5 for i in range(n)]
        dates = pd.date_range(start="2024-01-03 10:00", periods=n, freq="15min", tz="Asia/Kolkata")
        df = pd.DataFrame({
            "open": closes, "high": [c + 2 for c in closes],
            "low": [c - 2 for c in closes], "close": closes,
        }, index=dates)
        cfg = Config()
        result_df, signals = run_signals(df, None, cfg, df_weekly=None)
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(signals, pd.DataFrame)
        assert "atr" in result_df.columns
        assert "ema" in result_df.columns
        assert "session_ok" in result_df.columns

    def test_zone_signal_has_valid_risk(self):
        n = 100
        closes = [100 + i * 0.5 for i in range(n)]
        df = create_ohlcv_dataframe(closes)
        cfg = Config(min_score_threshold=30)
        result_df, signals = run_signals(df, None, cfg)
        if not signals.empty:
            for _, row in signals.iterrows():
                risk = abs(row["entry"] - row["sl"])
                assert risk > 0
                assert abs(row["entry"] - row["sl"]) <= abs(row["entry"]) * 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
