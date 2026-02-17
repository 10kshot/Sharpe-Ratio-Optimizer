# Sharpe Ratio Portfolio Optimizer  
**Rolling Out-of-Sample Mean–Variance Optimization**

This project implements a constrained Sharpe-ratio maximizing portfolio optimizer and evaluates it using a rolling, out-of-sample backtest. The goal is to demonstrate practical portfolio construction, risk modeling, and backtesting methodology while avoiding common pitfalls such as look-ahead bias.

---

## Overview

The optimizer constructs portfolios that maximize risk-adjusted return (Sharpe ratio) using historical asset returns. Portfolio weights are estimated using a rolling historical window and rebalanced periodically, then evaluated strictly out-of-sample.

Key components:
- Mean–variance portfolio optimization
- Sharpe ratio maximization under constraints
- Rolling estimation and rebalancing
- Out-of-sample backtesting
- Benchmark comparison and risk metrics

---

## Mathematical Framework

### Returns

Given asset prices P_{i,t}, log returns are computed as:

r_{i,t} = ln(P_{i,t} / P_{i,t-1})


Log returns are preferred due to their time-additivity and stability for statistical modeling.

---

### Expected Returns and Covariance

Using historical returns over a lookback window of length T:

μ = mean(r)

Σ = cov(r)

where:
- μ is an N-dimensional vector of expected asset returns
- Σ is an N x N covariance matrix capturing asset variances and correlations

---

### Portfolio Return and Risk

For portfolio weights w:

μ_p = wᵀ μ

σ_p = sqrt(wᵀ Σ w)

---

### Sharpe Ratio Objective

The Sharpe ratio is defined as:

Sharpe(w) = (μ_p − r_f) / σ_p

where \( r_f \) is the risk-free rate.

---

### Optimization Problem

The optimizer solves:

maximize   (wᵀ μ − r_f) / sqrt(wᵀ Σ w)

subject to sum(w) = 1
           w_i ≥ 0

This is a nonlinear constrained optimization problem solved numerically using Sequential Least Squares Programming (SLSQP).

---

## Rolling Out-of-Sample Backtest

To avoid look-ahead bias, the strategy is evaluated using a rolling framework:

1. Select a historical lookback window (e.g. 252 trading days)
2. Estimate μ and Σ using only past data
3. Optimize portfolio weights to maximize Sharpe ratio
4. Hold the portfolio for the next rebalance period (out-of-sample)
5. Repeat through time

Rebalancing is performed monthly by default.

---

## Benchmarks and Evaluation

The optimized portfolio is compared against:
- Equal-weight portfolio
- Buy-and-hold SPY

Performance metrics:
- Annual Return = 252 × mean(daily returns)
- Annual Volatility = sqrt(252) × std(daily returns)
- Sharpe Ratio = (Annual Return − r_f) / Annual Volatility
- Max Drawdown = min(cumulative_return / rolling_max − 1)

---
