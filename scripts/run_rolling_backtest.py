import numpy as np
import pandas as pd

from src.data import get_prices_yahoo, compute_returns
from src.backtest import rolling_sharpe_backtest, equal_weight_returns, cumulative_returns
from src.metrics import perf_stats
from src.plots import plot_equity_curves, plot_weights


def main():
    tickers = ["SPY", "QQQ", "TLT", "GLD"]

    prices = get_prices_yahoo(tickers, start="2018-01-01")
    returns = compute_returns(prices, method="log")

    rf_annual = 0.04
    rf_daily = rf_annual / 252.0

    lookback_days = 252
    rebalance = "M"

    rolling_rets, weights = rolling_sharpe_backtest(
        returns,
        lookback=lookback_days,
        rebalance_freq=rebalance,
        rf_daily=rf_daily
    )

    aligned = returns.loc[rolling_rets.index]
    eq_rets = equal_weight_returns(aligned)
    spy_rets = aligned["SPY"]

    curves = {
        "Rolling Max Sharpe": cumulative_returns(rolling_rets),
        "Equal Weight": cumulative_returns(eq_rets),
        "SPY Buy & Hold": cumulative_returns(spy_rets),
    }

    plot_equity_curves(curves)
    plot_weights(weights)

    stats = pd.DataFrame({
        "Rolling Sharpe": perf_stats(rolling_rets, rf_annual),
        "Equal Weight": perf_stats(eq_rets, rf_annual),
        "SPY": perf_stats(spy_rets, rf_annual),
    })
    print("\nPerformance Summary (Annualized):")
    print(stats)


if __name__ == "__main__":
    main()
