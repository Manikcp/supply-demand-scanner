"""
3-Step A+ Supply & Demand Strategy (Trade with Pat)

Step 1 — Identify institutional demand/supply zones from impulsive moves
Step 2 — Trade with the trend via Break of Structure (BOS)
Step 3 — Entry conditions: slow momentum, close in zone, confirmation candle

This module provides backward compatibility by re-exporting from submodules.
"""

from .supply_demand_v2 import (
    SignalType,
    MarketRegime,
    Config,
    is_trading_session,
    find_swing_points,
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

from .pattern_detection import PatternType, PatternStage, Pattern, detect_all_patterns
from .sr_levels import MarketStructure, detect_market_structure, get_all_sr_levels
