import numpy as np
import pandas as pd


def ema(close: pd.Series, length: int) -> pd.Series:
    return close.ewm(span=length, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()
