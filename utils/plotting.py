# Plotting helpers: CCDF, phase diagrams, calibration curves, posteriors

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

# ACM single-column style: 3.5 inch wide, 9pt font
_ACM_STYLE = {
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (3.5, 2.8),
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
    # Plot empirical CCDF on log-log scale, optionally with power-law fit line
    data = np.asarray(data, dtype=float)
    data = np.sort(data[data > 0])
    n = len(data)
    if n == 0:
        return

    ccdf = 1.0 - np.arange(1, n + 1) / n
    ax.plot(data, ccdf, color=color, linewidth=1.0, label=label)

    if fit_result is not None:
        alpha = fit_result["alpha"]
        xmin = fit_result["xmin"]
        xs = np.logspace(np.log10(xmin), np.log10(data.max()), 200)
        # power-law CCDF: P(X > x) ~ (x/xmin)^-(alpha-1)
        ys = (xs / xmin) ** (-(alpha - 1))
        # scale to match empirical at xmin
        idx = np.searchsorted(data, xmin)
        if idx < n:
            scale = ccdf[idx]
            ys *= scale / ys[0]
        ax.plot(
            xs, ys,
            linestyle="--",
            color=color,
            linewidth=0.8,
            label=rf"Power law $\alpha={alpha:.2f}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel("P(X > x)")
    ax.legend(frameon=False)


def plot_phase_diagram(param_grid, exponents, param_name, title, save_path,
                       L_values=None, cmap="viridis"):
    # Plot tail exponent vs control parameter, one curve per grid size
    _ensure_figures_dir()
    _apply_style()

    exponents = np.asarray(exponents, dtype=float)
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
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_calibration_mle(param_grid, alpha_sim, alpha_emp, param_name,
                         emp_label, title, save_path, p_star=None):
    # Plot simulated vs empirical tail exponent, mark best-fit parameter
    _ensure_figures_dir()
    _apply_style()

    fig, ax = plt.subplots()
    ax.plot(param_grid, alpha_sim, "o-", markersize=3, linewidth=1,
            color="steelblue", label="Simulated")
    ax.axhline(alpha_emp, linestyle="--", color="tomato", linewidth=1,
               label=emp_label)
    if p_star is not None:
        ax.axvline(p_star, linestyle=":", color="gray", linewidth=0.8,
                   label=f"${param_name}^*={p_star:.3f}$")

    ax.set_xlabel(f"${param_name}$")
    ax.set_ylabel("Tail exponent")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_abc_posterior(samples, weights, param_name, title, save_path):
    # Plot ABC posterior as a weighted histogram
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
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_return_series(returns, title, save_path):
    # Plot return time series
    _ensure_figures_dir()
    _apply_style()

    fig, ax = plt.subplots()
    ax.plot(returns, linewidth=0.5, color="steelblue")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Return")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_comparison(
    param_grid_cb, exponents_cb, L_values_cb,
    param_grid_ofc, exponents_ofc, L_values_ofc,
    save_path):
    # Side-by-side phase diagrams for CB and OFC
    _ensure_figures_dir()
    _apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    cmap_cb = plt.cm.get_cmap("Blues", len(L_values_cb) + 2)
    for i, L in enumerate(L_values_cb):
        axes[0].plot(param_grid_cb, exponents_cb[i], "o-", markersize=2,
                     linewidth=1, color=cmap_cb(i + 2), label=f"L={L}")
    axes[0].axvline(0.593, linestyle="--", color="gray", linewidth=0.8,
                    label=r"$p_c=0.593$")
    axes[0].set_xlabel("$p$")
    axes[0].set_ylabel(r"$\alpha$ (tail exponent)")
    axes[0].set_title("Cont-Bouchaud")
    axes[0].legend(frameon=False, fontsize=7)

    cmap_ofc = plt.cm.get_cmap("Oranges", len(L_values_ofc) + 2)
    for i, L in enumerate(L_values_ofc):
        axes[1].plot(param_grid_ofc, exponents_ofc[i], "o-", markersize=2,
                     linewidth=1, color=cmap_ofc(i + 2), label=f"L={L}")
    axes[1].set_xlabel(r"$\alpha_\mathrm{OFC}$")
    axes[1].set_ylabel("b-value (G-R)")
    axes[1].set_title("OFC")
    axes[1].legend(frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig
