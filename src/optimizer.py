import numpy as np
import pandas as pd
from scipy.optimize import minimize


def estimate_mu_sigma(returns: pd.DataFrame):
    """
    From returns matrix (T x N):
      mu: mean returns (N,)
      sigma: covariance matrix (N x N)
    """
    mu = returns.mean().values
    sigma = returns.cov().values
    tickers = list(returns.columns)
    return mu, sigma, tickers


def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(np.dot(w, mu))


def portfolio_volatility(w: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sqrt(w @ sigma @ w))


def sharpe_ratio(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, rf: float) -> float:
    """
    Sharpe(w) = (w^T mu - rf) / sqrt(w^T Sigma w)
    rf must be in SAME units as mu (daily vs annual).
    """
    vol = portfolio_volatility(w, sigma)
    if vol <= 0:
        return -np.inf
    return (portfolio_return(w, mu) - rf) / vol


def maximize_sharpe(
    mu: np.ndarray,
    sigma: np.ndarray,
    rf: float = 0.0,
    no_short: bool = True,
    min_w: float = 0.02,
    max_w: float = 0.40,
):
    """
    Maximize Sharpe ratio with constraints:
      sum(w) = 1
      min_w <= w_i <= max_w  (floor prevents degenerate single-asset solutions;
                               cap limits concentration risk)

    Args:
        mu:       Expected daily returns (N,)
        sigma:    Covariance matrix (N x N)
        rf:       Daily risk-free rate
        no_short: If False, removes lower bound (allows short positions)
        min_w:    Minimum weight per asset (default 2%)
        max_w:    Maximum weight per asset (default 40%)
    """
    n = len(mu)
    w0 = np.ones(n) / n

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if no_short:
        bounds = [(min_w, max_w)] * n
    else:
        bounds = [(None, max_w)] * n

    def objective(w):
        return -sharpe_ratio(w, mu, sigma, rf)

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12}
    )

    if not res.success:
        # Fall back to equal weight rather than crashing
        w_star = w0
    else:
        w_star = np.clip(res.x, 0, max_w)
        w_star /= w_star.sum()   # re-normalise after clipping

    return {
        "weights": w_star,
        "sharpe": sharpe_ratio(w_star, mu, sigma, rf),
        "return": portfolio_return(w_star, mu),
        "vol": portfolio_volatility(w_star, sigma),
    }
