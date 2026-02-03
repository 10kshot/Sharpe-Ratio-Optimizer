import numpy as np
import pandas as pd


def perf_stats(r: pd.Series, rf_annual: float = 0.0) -> pd.Series:
    """
    Basic performance stats for daily returns series r.
    rf_annual: annual risk-free rate used for annualized Sharpe computation.
    """
    ann_ret = 252.0 * r.mean()
    ann_vol = np.sqrt(252.0) * r.std()
    sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else np.nan

    cum = (1 + r).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd = float(drawdown.min())

    return pd.Series({
        "Ann Return": float(ann_ret),
        "Ann Vol": float(ann_vol),
        "Sharpe": float(sharpe),
        "Max Drawdown": max_dd
    })
