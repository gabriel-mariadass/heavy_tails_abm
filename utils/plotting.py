# Plotting helpers: CCDF, phase diagrams, calibration curves, posteriors

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

# ACM single-column style: 3.3 inch wide, 9pt font, no Type 3 fonts
_ACM_STYLE = {
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (3.3, 2.5),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
    "axes.spines.top": False,
    "axes.spines.right": False}


def _ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _apply_style():
    plt.rcParams.update(_ACM_STYLE)


def plot_ccdf(data, label, ax, color, fit_result=None):
    # Plot empirical CCDF on log-log scale, optionally with power-law fit line.
    # When fit_result is provided, xlim starts at xmin and fit line stops at
    # data.max() (no extrapolation beyond the data).
    data = np.asarray(data, dtype=float)
    data = np.sort(data[data > 0])
    n = len(data)
    if n == 0:
        return

    ccdf = 1.0 - np.arange(1, n + 1) / n
    ax.plot(data, ccdf, color=color, linewidth=1.0, label=label)

    if fit_result is not None:
        alpha = fit_result["alpha"]
        xmin  = fit_result["xmin"]
        xs = np.logspace(np.log10(xmin), np.log10(data.max()), 200)
        # power-law CCDF: P(X > x) ~ (x/xmin)^-(alpha-1)
        ys = (xs / xmin) ** (-(alpha - 1))
        # scale to match empirical at xmin
        idx = np.searchsorted(data, xmin)
        if idx < n:
            ys *= ccdf[idx] / ys[0]
        ax.plot(
            xs, ys,
            linestyle="--", color=color, linewidth=0.8,
            label=r"Power law $\alpha={:.2f}$".format(alpha))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel("P(X > x)")

    # xlim: start at xmin when fit is shown; guard against zero lower bound
    if fit_result is not None:
        ax.set_xlim(left=fit_result["xmin"])
    xl = ax.get_xlim()
    if xl[0] <= 0:
        ax.set_xlim(left=data[0])
    yl = ax.get_ylim()
    if yl[0] <= 0:
        ax.set_ylim(bottom=1.0 / (n + 1))

    ax.legend(frameon=False)


def plot_phase_diagram(param_grid, exponents, param_name, title, save_path,
                       L_values=None, cmap="viridis"):
    # Plot tail exponent vs control parameter, one curve per grid size.
    # title kept for API compatibility; no ax.set_title() call.
    _ensure_figures_dir()
    _apply_style()

    exponents  = np.asarray(exponents, dtype=float)
    param_grid = np.asarray(param_grid, dtype=float)

    fig, ax = plt.subplots()

    if exponents.ndim == 1 or L_values is None:
        ax.plot(param_grid, exponents.ravel(), "o-", markersize=3, linewidth=1)
    else:
        colors = plt.cm.get_cmap(cmap, len(L_values))
        for i, L in enumerate(L_values):
            ax.plot(
                param_grid, exponents[i],
                "o-", markersize=3, linewidth=1,
                color=colors(i), label=f"L={L}")
        ax.legend(frameon=False)

    ax.set_xlabel(f"${param_name}$")
    ax.set_ylabel("Tail exponent")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_calibration_mle(param_grid, alpha_sim, alpha_emp, param_name,
                         emp_label, title, save_path, p_star=None,
                         std_sim=None):
    # Plot simulated vs empirical tail exponent, mark best-fit parameter.
    # std_sim: optional array of per-point std; shades ±1 std band at alpha=0.2.
    # title kept for API compatibility; no ax.set_title() call.
    _ensure_figures_dir()
    _apply_style()

    param_grid = np.asarray(param_grid, dtype=float)
    alpha_sim  = np.asarray(alpha_sim,  dtype=float)

    fig, ax = plt.subplots()
    ax.plot(param_grid, alpha_sim, "o-", markersize=3, linewidth=1,
            color="steelblue", label="Simulated")

    if std_sim is not None:
        std_sim = np.asarray(std_sim, dtype=float)
        valid = ~(np.isnan(alpha_sim) | np.isnan(std_sim))
        ax.fill_between(
            param_grid[valid],
            alpha_sim[valid] - std_sim[valid],
            alpha_sim[valid] + std_sim[valid],
            color="steelblue", alpha=0.2, linewidth=0)

    ax.axhline(alpha_emp, linestyle="--", color="tomato", linewidth=1,
               label=emp_label)

    if p_star is not None:
        ax.axvline(p_star, linestyle=":", color="gray", linewidth=0.8,
                   label=f"${param_name}^* = {p_star:.3f}$")

    ax.set_xlabel(f"${param_name}$")
    ax.set_ylabel("Tail exponent")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_abc_posterior(samples, weights, param_name, title, save_path):
    # Plot ABC posterior as a weighted histogram.
    # title kept for API compatibility; no ax.set_title() call.
    _ensure_figures_dir()
    _apply_style()

    samples = np.asarray(samples, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    fig, ax = plt.subplots()
    ax.hist(samples, weights=weights, bins=20, color="steelblue",
            edgecolor="white", linewidth=0.4, density=True, alpha=0.8)
    ax.set_xlabel(f"${param_name}$")
    ax.set_ylabel("Posterior density")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_return_series(returns, title, save_path):
    # Plot return time series.
    # title kept for API compatibility; no ax.set_title() call.
    _ensure_figures_dir()
    _apply_style()

    fig, ax = plt.subplots()
    ax.plot(returns, linewidth=0.5, color="steelblue")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Return")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig
