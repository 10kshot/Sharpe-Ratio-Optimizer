import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

try:
    import yfinance as yf
except ImportError:
    yf = None

def get_prices_yahoo(tickers, start="2018-01-01", end=None, auto_adjust=True):
    """
    Fetch adjusted close prices from Yahoo Finance using yfinance.
    Requires: pip install yfinance
    """
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False
    )

    # yfinance returns multi-index columns if multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = tickers if isinstance(tickers, list) else [tickers]

    prices = prices.dropna(how="all")
    return prices

def compute_returns(prices: pd.DataFrame, method="log") -> pd.DataFrame:
    """
    prices: DataFrame of prices with datetime index and tickers as columns
    method: "log" or "simple"
    """
    if method not in {"log", "simple"}:
        raise ValueError("method must be 'log' or 'simple'")

    if method == "simple":
        rets = prices.pct_change()
    else:
        rets = np.log(prices / prices.shift(1))

    return rets.dropna()

def estimate_mu_sigma(returns: pd.DataFrame):
    """
    From returns matrix R (T x N):
      mu = E[r] ~ mean (N,)
      Sigma = cov matrix (N x N)
    """
    mu = returns.mean().values          # (N,)
    sigma = returns.cov().values        # (N, N)
    tickers = list(returns.columns)
    return mu, sigma, tickers

def portfolio_return(w, mu):
    # mu_p = w^T mu
    return float(np.dot(w, mu))

def portfolio_volatility(w, sigma):
    # sigma_p = sqrt(w^T Sigma w)
    return float(np.sqrt(w @ sigma @ w))

def sharpe_ratio(w, mu, sigma, rf):
    """
    Sharpe(w) = (w^T mu - rf) / sqrt(w^T Sigma w)
    rf must be in the SAME units as mu (e.g., daily if mu is daily).
    """
    vol = portfolio_volatility(w, sigma)
    if vol <= 0:
        return -np.inf
    return (portfolio_return(w, mu) - rf) / vol

def maximize_sharpe(mu, sigma, rf=0.0, no_short=True):
    """
    Solve:
      max_w (w^T mu - rf) / sqrt(w^T Sigma w)
    s.t.
      sum(w) = 1
      w_i >= 0   (if no_short True)
    """
    n = len(mu)

    # Initial guess: equal weights
    w0 = np.ones(n) / n

    # Constraints: sum(w) = 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Bounds: no short-selling
    if no_short:
        bounds = [(0.0, 1.0)] * n
    else:
        bounds = None  # allow negative weights

    # We minimize negative Sharpe
    def objective(w):
        return -sharpe_ratio(w, mu, sigma, rf)

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12}
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    w_star = result.x
    sr_star = sharpe_ratio(w_star, mu, sigma, rf)
    ret_star = portfolio_return(w_star, mu)
    vol_star = portfolio_volatility(w_star, sigma)

    return {
        "weights": w_star,
        "sharpe": sr_star,
        "return": ret_star,
        "vol": vol_star
    }

def simulate_random_portfolios(mu, sigma, rf=0.0, n_sims=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(mu)

    W = rng.random((n_sims, n))
    W = W / W.sum(axis=1, keepdims=True)

    rets = W @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, sigma, W))
    sharpes = (rets - rf) / vols

    return pd.DataFrame({"ret": rets, "vol": vols, "sharpe": sharpes})


def plot_risk_return_cloud(cloud: pd.DataFrame, opt_point: dict, title="Risk-Return Space"):
    plt.figure()
    plt.scatter(cloud["vol"], cloud["ret"], s=8)
    plt.scatter([opt_point["vol"]], [opt_point["return"]], marker="*", s=250)
    plt.xlabel("Volatility (σ)")
    plt.ylabel("Expected Return (μ)")
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Pick a small, interpretable universe
    tickers = ["SPY", "QQQ", "TLT", "GLD"]  # US equities, tech, bonds, gold

    # 1) Prices
    prices = get_prices_yahoo(tickers, start="2018-01-01")

    # 2) Returns (log returns recommended)
    returns = compute_returns(prices, method="log")

    # 3) Estimate mu and Sigma (daily)
    mu, sigma, names = estimate_mu_sigma(returns)

    # Risk-free rate: choose DAILY rf to match daily mu
    # Example: 4% annual -> daily approx 0.04/252 (simple approximation)
    rf_annual = 0.04
    rf_daily = rf_annual / 252.0

    # 4) Optimize
    opt = maximize_sharpe(mu, sigma, rf=rf_daily, no_short=True)

    # Print results
    print("Tickers:", names)
    print("Optimal weights:")
    for t, w in zip(names, opt["weights"]):
        print(f"  {t}: {w:.4f}")

    # Annualize for reporting (optional)
    # If using log returns, annual return approx 252*mu_p; annual vol approx sqrt(252)*sigma_p
    mu_p_daily = opt["return"]
    vol_p_daily = opt["vol"]
    sharpe_daily = opt["sharpe"]

    mu_p_annual = 252.0 * mu_p_daily
    vol_p_annual = np.sqrt(252.0) * vol_p_daily
    sharpe_annual = (mu_p_annual - rf_annual) / vol_p_annual

    print("\nPortfolio stats (daily):")
    print(f"  return:  {mu_p_daily:.6f}")
    print(f"  vol:     {vol_p_daily:.6f}")
    print(f"  sharpe:  {sharpe_daily:.4f}")

    print("\nPortfolio stats (annualized):")
    print(f"  return:  {mu_p_annual:.4f}")
    print(f"  vol:     {vol_p_annual:.4f}")
    print(f"  sharpe:  {sharpe_annual:.4f}")

    # 5) Visualize random portfolio cloud + optimizer point
    cloud = simulate_random_portfolios(mu, sigma, rf=rf_daily, n_sims=8000)
    plot_risk_return_cloud(cloud, opt, title="Random Portfolios + Max Sharpe Portfolio")

def rolling_sharpe_backtest(
    returns: pd.DataFrame,
    lookback=252,
    rebalance_freq="M",
    rf_daily=0.0
):
    """
    Rolling max-Sharpe portfolio (out-of-sample).

    returns: daily returns (T x N)
    lookback: estimation window (days)
    rebalance_freq: "M" (monthly) or "W" (weekly)
    """

    # Rebalance dates (end of period)
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

        # Out-of-sample period (until next rebalance)
        try:
            next_date = rebalance_dates[rebalance_dates.get_loc(date) + 1]
            test = returns.loc[date:next_date].iloc[1:]
        except (KeyError, IndexError):
            break

        port_rets = test.values @ w
        portfolio_returns.append(pd.Series(port_rets, index=test.index))
        weight_history.append(pd.Series(w, index=returns.columns, name=date))

    portfolio_returns = pd.concat(portfolio_returns)
    weight_history = pd.DataFrame(weight_history)

    return portfolio_returns, weight_history

def equal_weight_returns(returns):
    n = returns.shape[1]
    w = np.ones(n) / n
    return returns @ w

def cumulative_returns(r):
    return (1 + r).cumprod()

lookback_days = 252
rebalance = "M"

rolling_rets, weights = rolling_sharpe_backtest(
    returns,
    lookback=lookback_days,
    rebalance_freq=rebalance,
    rf_daily=rf_daily
)

eq_rets = equal_weight_returns(returns.loc[rolling_rets.index])
spy_rets = returns.loc[rolling_rets.index, "SPY"]

# Cumulative performance
cum_sharpe = cumulative_returns(rolling_rets)
cum_eq = cumulative_returns(eq_rets)
cum_spy = cumulative_returns(spy_rets)

plt.figure(figsize=(10, 5))
plt.plot(cum_sharpe, label="Rolling Max Sharpe")
plt.plot(cum_eq, label="Equal Weight")
plt.plot(cum_spy, label="SPY Buy & Hold")
plt.legend()
plt.title("Out-of-Sample Performance")
plt.ylabel("Cumulative Return")
plt.show()

def perf_stats(r, rf=0.0):
    ann_ret = 252 * r.mean()
    ann_vol = np.sqrt(252) * r.std()
    sharpe = (ann_ret - rf) / ann_vol

    cum = (1 + r).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()

    return pd.Series({
        "Ann Return": ann_ret,
        "Ann Vol": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd
    })

stats = pd.DataFrame({
    "Rolling Sharpe": perf_stats(rolling_rets, rf_annual),
    "Equal Weight": perf_stats(eq_rets, rf_annual),
    "SPY": perf_stats(spy_rets, rf_annual)
})

print(stats)