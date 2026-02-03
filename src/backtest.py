import numpy as np
import pandas as pd

from .optimizer import estimate_mu_sigma, maximize_sharpe


def equal_weight_returns(returns: pd.DataFrame) -> pd.Series:
    n = returns.shape[1]
    w = np.ones(n) / n
    return returns @ w


def cumulative_returns(r: pd.Series) -> pd.Series:
    return (1 + r).cumprod()


def rolling_sharpe_backtest(
    returns: pd.DataFrame,
    lookback: int = 252,
    rebalance_freq: str = "M",
    rf_daily: float = 0.0
):
    """
    Rolling max-Sharpe portfolio (out-of-sample).

    returns: daily returns (T x N)
    lookback: rolling estimation window length in trading days
    rebalance_freq: "M" monthly, "W" weekly, etc.
    rf_daily: daily risk-free rate (must match returns frequency)
    """
    rebalance_dates = returns.resample(rebalance_freq).last().index

    portfolio_returns = []
    weight_history = []

    for date in rebalance_dates:
        end_loc = returns.index.get_indexer([date], method="ffill")[0]
        if end_loc < lookback:
            continue

        train = returns.iloc[end_loc - lookback:end_loc]
        mu, sigma, _ = estimate_mu_sigma(train)

        opt = maximize_sharpe(mu, sigma, rf=rf_daily, no_short=True)
        w = opt["weights"]

        # Apply weights out-of-sample until next rebalance date
        try:
            next_date = rebalance_dates[rebalance_dates.get_loc(date) + 1]
            test = returns.loc[date:next_date].iloc[1:]  # exclude rebalance day
        except (KeyError, IndexError):
            break

        port_rets = test.values @ w
        portfolio_returns.append(pd.Series(port_rets, index=test.index))
        weight_history.append(pd.Series(w, index=returns.columns, name=date))

    if len(portfolio_returns) == 0:
        raise RuntimeError("Not enough data to run rolling backtest (increase history or reduce lookback).")

    portfolio_returns = pd.concat(portfolio_returns).sort_index()
    weight_history = pd.DataFrame(weight_history)

    return portfolio_returns, weight_history
