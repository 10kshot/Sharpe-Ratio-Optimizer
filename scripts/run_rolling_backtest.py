"""
run_rolling_backtest.py
=======================
Entry point for the Sharpe Ratio Portfolio Optimizer backtest.

Usage:
    python scripts/run_rolling_backtest.py

Outputs (written to results/):
    backtest_results.png   — 4-panel chart (cumulative returns, drawdown,
                             performance table, final weights)
    metrics.json           — machine-readable performance stats
    results_snippet.md     — copy-paste block ready for README.md
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.data import get_prices_yahoo, compute_returns
from src.backtest import rolling_sharpe_backtest, equal_weight_returns, cumulative_returns
from src.metrics import perf_stats
from src.plots import plot_results
from src.results import save_results


def main():
    # ── Configuration ──────────────────────────────────────────────────────────
    # Diversified universe: US tech, financials, defensives, commodities, bonds.
    # The mix of correlated (tech) and uncorrelated (GLD, TLT) assets gives the
    # optimizer meaningful diversification to work with.
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "XOM", "GLD", "TLT"]

    START        = "2018-01-01"   # fetch from here so we have a full lookback window
    RF_ANNUAL    = 0.025          # approximate average risk-free rate over period
    LOOKBACK     = 252            # 1 trading year estimation window
    REBALANCE    = "ME"           # month-end rebalance  ("M" for older pandas)
    MIN_W        = 0.02           # 2% floor — prevents degenerate single-asset solutions
    MAX_W        = 0.40           # 40% cap  — limits concentration risk
    OUTPUT_DIR   = "results"

    # ── Data ───────────────────────────────────────────────────────────────────
    print("Fetching price data …")
    all_tickers = TICKERS + ["SPY"]
    prices      = get_prices_yahoo(all_tickers, start=START)
    returns     = compute_returns(prices, method="log")

    asset_rets = returns[TICKERS]
    spy_rets_full = returns["SPY"]

    rf_daily = RF_ANNUAL / 252.0

    # ── Rolling backtest ───────────────────────────────────────────────────────
    print("Running rolling backtest …")
    try:
        port_rets, weight_history = rolling_sharpe_backtest(
            asset_rets,
            lookback=LOOKBACK,
            rebalance_freq=REBALANCE,
            rf_daily=rf_daily,
            min_w=MIN_W,
            max_w=MAX_W,
        )
    except TypeError:
        # Fallback for older pandas where "ME" isn't supported
        port_rets, weight_history = rolling_sharpe_backtest(
            asset_rets,
            lookback=LOOKBACK,
            rebalance_freq="M",
            rf_daily=rf_daily,
            min_w=MIN_W,
            max_w=MAX_W,
        )

    # ── Align benchmarks to OOS window ────────────────────────────────────────
    oos_index = port_rets.index
    aligned   = asset_rets.reindex(oos_index).dropna()
    port_rets = port_rets.reindex(aligned.index)

    eq_rets  = equal_weight_returns(aligned)
    spy_rets = spy_rets_full.reindex(aligned.index)

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame({
        "Sharpe Optimizer": perf_stats(port_rets, RF_ANNUAL),
        "Equal Weight":     perf_stats(eq_rets,   RF_ANNUAL),
        "SPY Buy & Hold":   perf_stats(spy_rets,  RF_ANNUAL),
    }).T

    print("\n── Performance Summary (Annualised) ──────────────────────────────────")
    fmt = metrics_df.copy()
    for col in ["Ann Return", "Ann Vol", "Max Drawdown"]:
        fmt[col] = fmt[col].map("{:.1%}".format)
    fmt["Sharpe"] = fmt["Sharpe"].map("{:.2f}".format)
    print(fmt.to_string())

    last_w = weight_history.iloc[-1].sort_values(ascending=False)
    print("\n── Final Portfolio Weights ───────────────────────────────────────────")
    for ticker, w in last_w.items():
        print(f"  {ticker:6s}  {w:.1%}")

    # ── Save outputs ───────────────────────────────────────────────────────────
    print(f"\nSaving outputs to {OUTPUT_DIR}/ …")
    chart_path = os.path.join(OUTPUT_DIR, "backtest_results.png")
    plot_results(port_rets, eq_rets, spy_rets, weight_history, metrics_df, output_path=chart_path)
    save_results(metrics_df, weight_history, output_dir=OUTPUT_DIR)

    print("\nDone. Add results/backtest_results.png to your repo root, then paste")
    print("results/results_snippet.md into your README.md Results section.")


if __name__ == "__main__":
    main()
