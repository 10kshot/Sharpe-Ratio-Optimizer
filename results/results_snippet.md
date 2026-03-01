## Results

**Backtest period:** Jan 2019 – Jan 2026 (out-of-sample)
**Asset universe:** AAPL · MSFT · GOOGL · AMZN · JPM · JNJ · XOM · GLD · TLT
**Lookback:** 252 trading days · **Rebalance:** Monthly · **Max position:** 40%

![Backtest Results](results/backtest_results.png)

### Performance Summary

| Strategy | Ann. Return | Ann. Volatility | Sharpe Ratio | Max Drawdown |
|---|---|---|---|---|
| **Sharpe Optimizer** | 18.6% | 14.9% | **1.08** | -17.5% |
| Equal Weight | 15.4% | 16.2% | 0.79 | -29.0% |
| SPY Buy & Hold | 13.9% | 19.7% | 0.58 | -38.4% |

### Final Portfolio Allocation

| Ticker | Weight |
|---|---|
| GLD | 40.0% |
| JNJ | 32.1% |
| GOOGL | 13.2% |
| XOM | 4.7% |
| TLT | 2.0% |
| JPM | 2.0% |
| AAPL | 2.0% |
| AMZN | 2.0% |
| MSFT | 2.0% |

> The optimizer maximises **risk-adjusted** return, not raw return.
> In high-momentum bull markets (e.g. 2020–2021) passive SPY exposure will typically win on absolute return,
> while the optimizer's value shows up in lower drawdowns and more stable Sharpe during volatile or bearish periods.

### Limitations & Future Work

- **Estimation error:** Sample mean is noisy. Future: Ledoit-Wolf or Black-Litterman priors.
- **No transaction costs:** Monthly rebalancing is low-turnover but costs are not modelled.
- **Universe selection bias:** Assets chosen with some hindsight. Production use requires rules-based universe construction.
- **Potential extensions:** DCC-GARCH covariance, regime-switching allocation, 50+ asset universe with sector constraints.
