import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os

# Theme
BG    = "#161b22"
DARK  = "#0d1117"
TEXT  = "#c9d1d9"
GRID  = "#30363d"
PANEL = "#21262d"
COLORS = {
    "Sharpe Optimizer": "#00d4aa",
    "Equal Weight":     "#f0a500",
    "SPY Buy & Hold":   "#e05c5c",
}
BAR_PALETTE = [
    "#00d4aa", "#58a6ff", "#f0a500", "#e05c5c",
    "#a371f7", "#79c0ff", "#ffa657", "#7ee787", "#ff7b72",
]


def _style_ax(ax, title=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def plot_results(
    port_rets,
    eq_rets,
    spy_rets,
    weight_history,
    metrics_df,
    output_path="results/backtest_results.png",
):
    """
    Generate the full 4-panel results chart and save to output_path.
    Panels: cumulative returns | drawdown | performance table | final weights bar
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor(DARK)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35, height_ratios=[2.2, 1.5, 2])

    # 1. Cumulative returns
    ax1 = fig.add_subplot(gs[0, :])
    _style_ax(ax1, "Cumulative Out-of-Sample Returns")
    for rets, lbl in [(port_rets, "Sharpe Optimizer"), (eq_rets, "Equal Weight"), (spy_rets, "SPY Buy & Hold")]:
        cum = ((1 + rets).cumprod() - 1) * 100
        ax1.plot(cum.index, cum.values, label=lbl, color=COLORS[lbl], linewidth=2)
    ax1.set_ylabel("Cumulative Return (%)", color=TEXT)
    ax1.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    _style_ax(ax2, "Rolling Drawdown")
    for rets, lbl in [(port_rets, "Sharpe Optimizer"), (eq_rets, "Equal Weight"), (spy_rets, "SPY Buy & Hold")]:
        cum = (1 + rets).cumprod()
        dd  = (cum / cum.cummax() - 1) * 100
        ax2.fill_between(dd.index, dd.values, alpha=0.18, color=COLORS[lbl])
        ax2.plot(dd.index, dd.values, label=lbl, color=COLORS[lbl], linewidth=1.5)
    ax2.set_ylabel("Drawdown (%)", color=TEXT)
    ax2.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # 3. Performance table
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor(BG)
    ax3.axis("off")
    ax3.set_title("Performance Summary", color=TEXT, fontsize=10, fontweight="bold", pad=8)
    rows     = list(metrics_df.index)
    col_keys = ["Ann Return", "Ann Vol", "Sharpe", "Max Drawdown"]
    col_lbls = ["Ann. Ret.", "Ann. Vol.", "Sharpe", "Max DD"]

    def _fmt(v, k):
        return f"{v:.2f}" if k == "Sharpe" else f"{v:.1%}"

    cell_data = [[_fmt(metrics_df.loc[r, k], k) for k in col_keys] for r in rows]
    tbl = ax3.table(cellText=cell_data, rowLabels=rows, colLabels=col_lbls,
                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1c2128" if r % 2 == 0 else BG)
        cell.set_text_props(color=TEXT)
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor(PANEL)
            cell.set_text_props(color="#58a6ff", fontweight="bold")
        if c == -1 and r > 0:
            cell.set_facecolor(PANEL)
            cell.set_text_props(color=list(COLORS.values())[r - 1], fontweight="bold")

    # 4. Final weights bar
    ax4 = fig.add_subplot(gs[2, 1])
    _style_ax(ax4, "Final Portfolio Allocation")
    last_w = weight_history.iloc[-1].sort_values(ascending=False)
    bars = ax4.bar(last_w.index, last_w.values * 100,
                   color=BAR_PALETTE[:len(last_w)], edgecolor=DARK, linewidth=0.8)
    ax4.set_ylabel("Weight (%)", color=TEXT)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax4.tick_params(axis="x", rotation=35, labelsize=8)
    for bar, val in zip(bars, last_w.values):
        if val > 0.03:
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                     f"{val:.1%}", ha="center", va="bottom", color=TEXT, fontsize=7.5)

    tickers  = " ".join(weight_history.columns.tolist())
    start_y  = port_rets.index[0].year
    end_y    = port_rets.index[-1].year
    plt.suptitle(
        f"Sharpe Ratio Portfolio Optimizer — Backtest Results\n"
        f"Universe: {tickers}  |  OOS: {start_y}–{end_y}  |  Max Position 40%",
        color=TEXT, fontsize=10, fontweight="bold", y=1.005,
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Chart saved → {output_path}")


# kept for backward compatibility
def plot_equity_curves(curves, title="Out-of-Sample Performance", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK)
    _style_ax(ax, title)
    for name, series in curves.items():
        pct = (series - 1) * 100
        ax.plot(pct, label=name, color=COLORS.get(name, "#8b949e"), linewidth=2)
    ax.set_ylabel("Cumulative Return (%)", color=TEXT)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Chart saved → {save_path}")
    plt.close()


def plot_weights(weights, title="Rolling Weights (Rebalance Dates)", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK)
    _style_ax(ax, title)
    weights.plot(kind="area", stacked=True, ax=ax, color=BAR_PALETTE[:weights.shape[1]], alpha=0.85)
    ax.set_ylabel("Weight", color=TEXT)
    ax.legend(loc="upper left", ncol=2, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Chart saved → {save_path}")
    plt.close()
