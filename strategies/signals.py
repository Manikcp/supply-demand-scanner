"""
Signal Generation — 3-Step A+ Supply & Demand Strategy

Re-exports from supply_demand_v2 for backward compatibility.
"""
from .supply_demand_v2 import (
    SignalType,
    MarketRegime,
    Config,
    is_trading_session,
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
    find_swing_points,
    calculate_entry_score,
    run_signals,
    print_signal,
)
