"""
Strategies Package

3-Step A+ Supply & Demand Strategy (Trade with Pat):
- supply_demand_v2: Core strategy - supply/demand zone detection, trend BOS, entry conditions
- pattern_detection: Chart pattern recognition
- sr_levels: Support & resistance detection with market structure
"""

from .pattern_detection import (
    PatternType,
    PatternStage,
    Pattern,
    find_pivots,
    calculate_atr,
    dynamic_tolerance,
    calculate_pattern_confidence,
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

from .sr_levels import (
    MarketStructure,
    LevelType,
    SRLevel,
    detect_market_structure,
    find_swing_points,
    cluster_levels_adaptive,
    find_weekly_sr_levels,
    confirm_levels_on_daily,
    get_all_sr_levels,
)

from .supply_demand_v2 import (
    SignalType,
    MarketRegime,
    Config,
    is_trading_session,
    find_swing_points as sd_find_swing_points,
    detect_trend_bos,
    detect_impulsive_move,
    find_consolidation_before_impulse,
    detect_fvg,
    check_slow_momentum,
    check_zone_entry,
    is_fresh_zone,
    detect_demand_zones,
    detect_supply_zones,
    merge_overlapping_zones,
    calculate_entry_score,
    run_signals,
    print_signal,
)

__all__ = [
    "PatternType", "PatternStage", "Pattern",
    "find_pivots", "calculate_atr", "dynamic_tolerance",
    "calculate_pattern_confidence",
    "detect_head_and_shoulders", "detect_inverse_head_and_shoulders",
    "detect_double_top", "detect_double_bottom",
    "detect_ascending_triangle", "detect_descending_triangle",
    "detect_bull_flag", "detect_bear_flag",
    "detect_rising_wedge", "detect_falling_wedge",
    "detect_cup_and_handle", "detect_all_patterns",
    "MarketStructure", "LevelType", "SRLevel",
    "detect_market_structure", "find_swing_points",
    "cluster_levels_adaptive", "find_weekly_sr_levels",
    "confirm_levels_on_daily", "get_all_sr_levels",
    "SignalType", "MarketRegime", "Config",
    "is_trading_session",
    "detect_trend_bos", "detect_impulsive_move",
    "find_consolidation_before_impulse",
    "detect_fvg", "check_slow_momentum",
    "check_zone_entry", "is_fresh_zone",
    "detect_demand_zones", "detect_supply_zones",
    "merge_overlapping_zones", "calculate_entry_score",
    "run_signals", "print_signal",
]
