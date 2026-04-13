# Plotting helpers: CCDF, phase diagrams, calibration curves, posteriors
# Single source of truth for all figure generation in the project.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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


# ── Low-level primitive ───────────────────────────────────────────────────────

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
        ys = (xs / xmin) ** (-(alpha - 1))
        idx = np.searchsorted(data, xmin)
        if idx < n:
            ys *= ccdf[idx] / ys[0]
        ax.plot(xs, ys, linestyle="--", color=color, linewidth=0.8,
                label=r"Power law $\alpha={:.2f}$".format(alpha))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel("P(X > x)")

    if fit_result is not None:
        ax.set_xlim(left=fit_result["xmin"])
    xl = ax.get_xlim()
    if xl[0] <= 0:
        ax.set_xlim(left=data[0])
    yl = ax.get_ylim()
    if yl[0] <= 0:
        ax.set_ylim(bottom=1.0 / (n + 1))

    ax.legend(frameon=False)


# ── CB demo figures ───────────────────────────────────────────────────────────

def plot_cb_demo_ccdf(abs_ret, fit_res, p, L, save_path):
    # CCDF of absolute CB returns with power-law fit overlay.
    _ensure_figures_dir()
    _apply_style()
    fig, ax = plt.subplots()
    plot_ccdf(abs_ret, label="CB |returns|", ax=ax,
              color="steelblue", fit_result=fit_res)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


# ── CB phase diagram ──────────────────────────────────────────────────────────

def plot_cb_phase_diagram(exponents, p_grid, L_values, p_c, save_path):
    # Tail exponent vs percolation probability for each L.
    # Vertical dashed line at the percolation threshold p_c.
    _ensure_figures_dir()
    _apply_style()

    exponents = np.asarray(exponents, dtype=float)
    p_grid    = np.asarray(p_grid,    dtype=float)
    colors    = cm.Blues(np.linspace(0.4, 0.9, len(L_values)))

    fig, ax = plt.subplots()
    for i, L in enumerate(L_values):
        mask = ~np.isnan(exponents[i])
        ax.plot(p_grid[mask], exponents[i][mask], "o-",
                markersize=3, linewidth=1, color=colors[i], label=f"L={L}")

    ax.axvline(p_c, linestyle="--", color="gray", linewidth=0.8,
               label=rf"$p_c={p_c}$")
    ax.set_xlabel("$p$")
    ax.set_ylabel(r"$\alpha$ (tail exponent)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


# ── CB calibration ────────────────────────────────────────────────────────────

def plot_calibration_mle(param_grid, alpha_sim, alpha_emp, param_name,
                         emp_label, title, save_path, p_star=None,
                         std_sim=None):
    # Simulated vs empirical tail exponent with optional ±1-std band.
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
        ax.fill_between(param_grid[valid],
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
    # ABC posterior as a weighted histogram.
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


# ── CB return time series ─────────────────────────────────────────────────────

def plot_return_series(returns, title, save_path):
    # Return time series.
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


# ── OFC demo figures ──────────────────────────────────────────────────────────

def plot_ofc_demo_timeseries(sizes, L, alpha_ofc, save_path):
    # Semilogy of OFC avalanche sizes (first 2 000 events).
    _ensure_figures_dir()
    _apply_style()

    fig, ax = plt.subplots()
    ax.semilogy(sizes[:2000], linewidth=0.4, color="darkorange")
    ax.set_xlabel("Event index")
    ax.set_ylabel("Avalanche size")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_ofc_demo_ccdf(sizes, fit_res, L, alpha_ofc, save_path):
    # CCDF of OFC avalanche sizes with power-law fit overlay.
    _ensure_figures_dir()
    _apply_style()

    fig, ax = plt.subplots()
    plot_ccdf(sizes.astype(float), label="OFC avalanches", ax=ax,
              color="darkorange", fit_result=fit_res)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


# ── OFC phase diagram ─────────────────────────────────────────────────────────

def plot_ofc_phase_diagram(b_means, b_stds, alpha_grid, L_values, n_seeds,
                           save_path):
    # b-value vs alpha_OFC for each L with SEM error bars.
    # L=64: solid+circle, L=32: dashed+square, L=128: dotted+triangle.
    # Vertical dashed line at first alpha where L=64 b-value drops below 1.0.
    _ensure_figures_dir()
    _apply_style()

    b_means    = np.asarray(b_means, dtype=float)
    b_stds     = np.asarray(b_stds,  dtype=float)
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    colors     = cm.Oranges(np.linspace(0.35, 0.95, len(L_values)))
    _style     = {32: ("s", "--"), 64: ("o", "-"), 128: ("^", ":")}

    fig, ax = plt.subplots()
    for i, L in enumerate(L_values):
        mask  = ~np.isnan(b_means[i])
        yerr  = b_stds[i][mask] / np.sqrt(n_seeds)
        marker, ls = _style.get(L, ("o", "-"))
        ax.errorbar(alpha_grid[mask], b_means[i][mask], yerr=yerr,
                    fmt=marker + ls, markersize=3, linewidth=1,
                    color=colors[i], label=f"L={L}",
                    capsize=2, elinewidth=0.7)

    if 64 in list(L_values):
        idx64 = list(L_values).index(64)
        b64   = b_means[idx64]
        below = (~np.isnan(b64)) & (b64 < 1.0)
        if below.any():
            alpha_cross = alpha_grid[below][0]
            ax.axvline(alpha_cross, linestyle="--", color="dimgray",
                       linewidth=0.8,
                       label=f"b=1 crossing (\u03b1={alpha_cross:.2f})")

    ax.set_xlabel(r"$\alpha_{\mathrm{OFC}}$")
    ax.set_ylabel("b-value (G-R)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


# ── Generic phase diagram (used by CB via plot_phase_diagram) ─────────────────

def plot_phase_diagram(param_grid, exponents, param_name, title, save_path,
                       L_values=None, cmap="viridis"):
    # Generic tail-exponent vs control-parameter plot; title unused.
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
            ax.plot(param_grid, exponents[i], "o-", markersize=3, linewidth=1,
                    color=colors(i), label=f"L={L}")
        ax.legend(frameon=False)

    ax.set_xlabel(f"${param_name}$")
    ax.set_ylabel("Tail exponent")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig
