import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curves(curves: dict, title: str = "Out-of-Sample Performance"):
    """
    curves: dict[str, pd.Series] where series is cumulative return index.
    """
    plt.figure(figsize=(10, 5))
    for name, series in curves.items():
        plt.plot(series, label=name)
    plt.legend()
    plt.title(title)
    plt.ylabel("Cumulative Return")
    plt.show()


def plot_weights(weights: pd.DataFrame, title: str = "Rolling Weights (Rebalance Dates)"):
    """
    weights: DataFrame where rows are rebalance dates and columns are tickers.
    """
    ax = weights.plot(kind="area", stacked=True, figsize=(10, 5))
    ax.set_title(title)
    ax.set_ylabel("Weight")
    ax.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    plt.show()
