"""
results.py — Save backtest metrics to JSON and generate README results block.
"""
import json
import os
import pandas as pd


def save_results(metrics_df: pd.DataFrame, weight_history: pd.DataFrame, output_dir: str = "results"):
    """
    Persist backtest results to:
      results/metrics.json          — machine-readable stats
      results/results_snippet.md    — copy-paste block for README
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_path = os.path.join(output_dir, "metrics.json")
    payload = {
        "metrics": {
            row: {col: round(float(val), 6) for col, val in metrics_df.loc[row].items()}
            for row in metrics_df.index
        },
        "final_weights": {
            ticker: round(float(w), 6)
            for ticker, w in weight_history.iloc[-1].sort_values(ascending=False).items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Metrics saved  → {json_path}")

    # ── Markdown snippet ───────────────────────────────────────────────────────
    md_path  = os.path.join(output_dir, "results_snippet.md")
    port_row = metrics_df.loc["Sharpe Optimizer"]
    eq_row   = metrics_df.loc["Equal Weight"]
    spy_row  = metrics_df.loc["SPY Buy & Hold"]

    start_dt = weight_history.index[0].strftime("%b %Y")
    end_dt   = weight_history.index[-1].strftime("%b %Y")
    tickers  = " · ".join(weight_history.columns.tolist())

    last_w   = weight_history.iloc[-1].sort_values(ascending=False)
    w_rows   = "\n".join(
        f"| {t} | {w:.1%} |" for t, w in last_w.items()
    )

    snippet = f"""## Results

**Backtest period:** {start_dt} – {end_dt} (out-of-sample)
**Asset universe:** {tickers}
**Lookback:** 252 trading days · **Rebalance:** Monthly · **Max position:** 40%

![Backtest Results](results/backtest_results.png)

### Performance Summary

| Strategy | Ann. Return | Ann. Volatility | Sharpe Ratio | Max Drawdown |
|---|---|---|---|---|
| **Sharpe Optimizer** | {port_row['Ann Return']:.1%} | {port_row['Ann Vol']:.1%} | **{port_row['Sharpe']:.2f}** | {port_row['Max Drawdown']:.1%} |
| Equal Weight | {eq_row['Ann Return']:.1%} | {eq_row['Ann Vol']:.1%} | {eq_row['Sharpe']:.2f} | {eq_row['Max Drawdown']:.1%} |
| SPY Buy & Hold | {spy_row['Ann Return']:.1%} | {spy_row['Ann Vol']:.1%} | {spy_row['Sharpe']:.2f} | {spy_row['Max Drawdown']:.1%} |

### Final Portfolio Allocation

| Ticker | Weight |
|---|---|
{w_rows}

> The optimizer maximises **risk-adjusted** return, not raw return.
> In high-momentum bull markets (e.g. 2020–2021) passive SPY exposure will typically win on absolute return,
> while the optimizer's value shows up in lower drawdowns and more stable Sharpe during volatile or bearish periods.

### Limitations & Future Work

- **Estimation error:** Sample mean is noisy. Future: Ledoit-Wolf or Black-Litterman priors.
- **No transaction costs:** Monthly rebalancing is low-turnover but costs are not modelled.
- **Universe selection bias:** Assets chosen with some hindsight. Production use requires rules-based universe construction.
- **Potential extensions:** DCC-GARCH covariance, regime-switching allocation, 50+ asset universe with sector constraints.
"""

    with open(md_path, "w") as f:
        f.write(snippet)
    print(f"  README snippet → {md_path}")
    print("\n── Paste the contents of results/results_snippet.md into your README.md ──\n")
