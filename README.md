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

Given asset prices \( P_{i,t} \), log returns are computed as:

\[
r_{i,t} = \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right)
\]

Log returns are preferred due to their time-additivity and stability for statistical modeling.

---

### Expected Returns and Covariance

Using historical returns over a lookback window of length \( T \):

\[
\boldsymbol{\mu} = \mathbb{E}[r] \approx \frac{1}{T} \sum_{t=1}^{T} r_t
\]

\[
\Sigma = \text{Cov}(r)
\]

where:
- \( \boldsymbol{\mu} \in \mathbb{R}^N \) is the vector of expected asset returns
- \( \Sigma \in \mathbb{R}^{N \times N} \) is the covariance matrix

---

### Portfolio Return and Risk

For portfolio weights \( \mathbf{w} \):

\[
\mu_p = \mathbf{w}^\top \boldsymbol{\mu}
\]

\[
\sigma_p = \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}
\]

---

### Sharpe Ratio Objective

The Sharpe ratio is defined as:

\[
\text{Sharpe}(\mathbf{w}) =
\frac{\mu_p - r_f}{\sigma_p}
=
\frac{\mathbf{w}^\top \boldsymbol{\mu} - r_f}
{\sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}}
\]

where \( r_f \) is the risk-free rate.

---

### Optimization Problem

The optimizer solves:

\[
\max_{\mathbf{w}}
\quad
\frac{\mathbf{w}^\top \boldsymbol{\mu} - r_f}
{\sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}}
\]

Subject to:
- \( \sum_i w_i = 1 \) (fully invested)
- \( w_i \ge 0 \) (no short selling)

This is a nonlinear constrained optimization problem solved numerically using Sequential Least Squares Programming (SLSQP).

---

## Rolling Out-of-Sample Backtest

To avoid look-ahead bias, the strategy is evaluated using a rolling framework:

1. Select a historical lookback window (e.g. 252 trading days)
2. Estimate \( \boldsymbol{\mu} \) and \( \Sigma \) using only past data
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
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown

---

## Project Structure

