import numpy as np
import pandas as pd
from typing import List

def compute_volatility(df: pd.DataFrame, period: int, tickers: List[str]) -> pd.DataFrame:
    out = df.copy()
    for ticker in tickers:
        price_col = f"Adj Close_{ticker}"
        r = np.log(df[price_col] / df[price_col].shift(1)).fillna(0)
        vol = r.rolling(window=period).std() * np.sqrt(period)
        out[f"{ticker}_volatility"] = vol.fillna(0)

    return out


def compute_momentum(df, period, tickers):
    out = df.copy()
    for ticker in tickers:
        price_col = f"Adj Close_{ticker}"
        mom = np.log(df[price_col] / df[price_col].shift(period)).fillna(0)
        out[f"{ticker}_momentum"] = mom

    return out


def compute_return(df, tickers):
    out = df.copy()
    for ticker in tickers:
        price_col = f"Adj Close_{ticker}"
        ret = np.log(df[price_col] / df[price_col].shift(1)).fillna(0)
        out[f"{ticker}_returns"] = ret

    return out
