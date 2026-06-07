# S&R Algo v3 — Multi-Timeframe Pattern Scanner

**Professional-grade trading scanner with multi-timeframe analysis and pattern detection.**

---

## Features

- **Multi-timeframe Analysis**: Daily patterns + 15-min entries
- **11 Chart Patterns**: H&S, Double Top/Bottom, Triangles, Flags, Wedges, Cup & Handle
- **S&R Quality Scoring**: Level strength based on recency, reactions, volume
- **Entry Quality Grading**: A/B/C/D with contributing factors
- **Interactive Charts**: Candlestick charts with S&R levels, targets, SL
- **Real-time Scanning**: All 226 FNO stocks + 7 indices

---

## Detected Patterns

| Pattern | Direction | Reliability | Detected On |
|---------|-----------|-------------|-------------|
| Head & Shoulders | Bearish | High | Daily |
| Inverse Head & Shoulders | Bullish | High | Daily |
| Double Top | Bearish | High | Daily |
| Double Bottom | Bullish | High | Daily |
| Cup & Handle | Bullish | High | Daily |
| Ascending Triangle | Bullish | High | Daily/15min |
| Descending Triangle | Bearish | High | Daily/15min |
| Symmetrical Triangle | Either | Medium | Daily/15min |
| Rising Wedge | Bearish | Medium | Daily/15min |
| Falling Wedge | Bullish | Medium | Daily/15min |
| Bull Flag | Bullish | High | 15min |
| Bear Flag | Bearish | High | 15min |

---

## How It Works

### Signal Generation Flow

```
┌─────────────────────────────────────────────────────────┐
│                    DAILY CHART                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Pattern Detection (H&S, Double Top/Bottom, etc) │   │
│  │ - Pivot point identification                     │   │
│  │ - Pattern validation                             │   │
│  │ - Breakout confirmation                          │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │ If Pattern Found:                               │   │
│  │ - Target = Pattern measured move                │   │
│  │ - SL = Pattern invalidation level               │   │
│  │ - Pattern Score = +30 points                    │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   15-MIN CHART                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │ S&R Level Detection                             │   │
│  │ - Pivot highs/lows                              │   │
│  │ - Level clustering                              │   │
│  │ - Quality scoring (recency, reactions, volume)  │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Entry Timing                                    │   │
│  │ - Price in S&R zone (ATR-based)                 │   │
│  │ - Candlestick confirmation                      │   │
│  │ - Volume above average                          │   │
│  │ - Trend alignment check                         │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   SIGNAL OUTPUT                         │
│  Entry Grade: A/B/C/D (based on quality score)          │
│  Pattern: If detected on daily                          │
│  Entry, Target, SL, RR                                  │
│  Contributing Factors                                   │
└─────────────────────────────────────────────────────────┘
```

### Entry Quality Scoring

| Factor | Score | Description |
|--------|-------|-------------|
| Pattern Detected | +30 | Chart pattern on daily chart |
| Breakout Confirmed | +10 | Pattern breakout happened |
| Trend Alignment | +20 | EMA trend in signal direction |
| Candlestick Pattern | +15 | Hammer, Engulfing, etc. |
| Strong Level | +15 | Level quality >= 70 |
| Volume Confirmation | +10 | Above 20-bar average |
| Multiple Touches | +5 | 4+ reactions at level |

**Grading:**
- **Grade A (80%+)**: High conviction - full position size
- **Grade B (60-79%)**: Good setup - 75% position
- **Grade C (40-59%)**: Acceptable - 50% position
- **Grade D (<40%)**: Filtered out

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Web UI
```bash
cd "/Users/manikappa.panchal/Desktop/fo_trading_framework copy 3"
python3 -m streamlit run app.py
```

Opens at `http://localhost:8501`

### 3. Run CLI Scanner
```bash
python3 run_twp_scanner.py
```

---

## Timeframe Configuration

| Timeframe | Period | Purpose |
|-----------|--------|---------|
| Weekly | 5 years (260 weeks) | Major S&R levels, macro patterns |
| Daily | 365 days | Pattern detection, trend confirmation |
| 15-min | 24 bars (1 day) | Entry timing, intraday signals |

---

## UI Features

- **One-click Scan**: Scan all 226 tickers
- **Filter Options**: Signal type, minimum RR, minimum grade
- **Pattern Table**: See all pattern-based signals
- **Interactive Charts**: Candlestick + S&R + Entry/Target/SL
- **CSV Export**: Download signals for analysis

---

## Output Example

### Signal with Pattern
```
▲ CE BUY — Grade A (92%)
  NIFTY22000 @ ₹22045

  Pattern: Inverse Head & Shoulders (high reliability)
  
  Support: ₹21980 (Reactions: 4, Quality: 78)
  Target: ₹22300 (Pattern measured move)
  SL: ₹21940
  RR: 1:3.2

  Factors: pattern:Inverse H&S, breakout, trend, candle, volume
  Session: regular
```

### Signal without Pattern (S&R Only)
```
▼ PE BUY — Grade B (65%)
  BANKNIFTY45000 @ ₹44985

  Resistance: ₹45020 (Reactions: 3, Quality: 65)
  Target: ₹44850
  SL: ₹45050
  RR: 1:2.1

  Factors: trend, candle, moderate_level, volume
```

---

## Project Structure

```
fo_trading_framework/
├── strategies/
│   └── twp_algo_v6.py      # Multi-timeframe strategy engine
├── app.py                   # Streamlit web UI
├── run_twp_scanner.py       # CLI live scanner
├── fno_tickers.csv          # 226 tickers
├── config.json              # Global config
├── requirements.txt         # Dependencies
└── README.md
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback_weeks` | 260 | Lookback for Weekly S&R detection (5 years) |
| `daily_lookback_days` | 365 | Lookback for Daily chart analysis |
| `intraday_lookback_bars` | 24 | 15-min bars for intraday analysis |
| `min_reactions` | 2 | Minimum touches at level |
| `zone_atr_mult` | 0.25 | Zone width = ATR × mult |
| `min_rr_ratio` | 2.0 | Minimum reward:risk |
| `min_level_quality` | 30 | Minimum level quality score |
| `pattern_lookback` | 100 | Bars for pattern detection |
| `trend_ema_fast` | 20 | Fast EMA period |
| `trend_ema_slow` | 50 | Slow EMA period |

---

## Supported Tickers

**Indices (7):** NIFTY50, BANKNIFTY, SENSEX, FINNIFTY, MIDCAPSELECT, BANKEX, NIFTYJR

**FNO Stocks (219):** All NSE F&O stocks in `fno_tickers.csv`

---

## Requirements

```
python >= 3.9
pandas
pandas-ta
yfinance
numpy
streamlit
plotly
```

---

## Risk Disclaimer

> This software is for **educational and research purposes only**.
> It does **not** constitute financial advice.
> 
> **Options trading risks:**
> - Theta decay (time value loss)
> - Vega risk (IV changes)
> - Gamma risk (delta changes near expiry)
> - Liquidity risk (wide spreads)
> 
> This strategy uses spot price for levels. Actual option premiums may differ.
> Always paper-trade before using real capital.



Structure	Condition	Meaning
BULLISH	bullish_score > 0.6 AND trend_strength > 1%	Clear uptrend — higher highs & higher lows, EMA20 well above EMA50
BEARISH	bearish_score > 0.6 AND trend_strength < -1%	Clear downtrend — lower highs & lower lows, EMA20 well below EMA50
RANGING	|trend_strength| < 0.5%	Sideways/consolidating market — EMAs are nearly equal, no directional bias
TRANSITION	None of the above	Market is shifting — some directional signals exist but they're mixed or weak; the market is transitioning between states

commnd to clean cache.
find "/Users/manikappa.panchal/Desktop/fo_trading_framework copy 4" -type d \( -name "__pycache__" -o -name ".pytest_cache" \) -exec rm -rf {} +
