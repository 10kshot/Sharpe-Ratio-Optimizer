import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


def get_prices_yahoo(tickers, start="2018-01-01", end=None, auto_adjust=True) -> pd.DataFrame:
    """
    Fetch close prices from Yahoo Finance using yfinance.

    Returns:
        DataFrame indexed by date with tickers as columns.
    """
    if yf is None:
        raise ImportError("yfinance is not installed. Run: python -m pip install yfinance")

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False
    )

    # Multi-index when multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = tickers if isinstance(tickers, list) else [tickers]

    prices = prices.dropna(how="all")
    return prices


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    Compute returns from prices.

    method:
        - "log": log returns ln(Pt/Pt-1)
        - "simple": arithmetic returns (Pt-Pt-1)/Pt-1
    """
    if method not in {"log", "simple"}:
        raise ValueError("method must be 'log' or 'simple'")

    if method == "simple":
        rets = prices.pct_change()
    else:
        rets = np.log(prices / prices.shift(1))

    return rets.dropna()
