# OFC phase diagram: sweep alpha_ofc and L, measure G-R b-value

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from models.ofc import simulate_ofc
from utils.powerlaw_fit import gutenberg_richter_b
from utils.plotting import FIGURES_DIR, _apply_style, _ensure_figures_dir

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# experiment parameters
ALPHA_GRID = np.linspace(0.10, 0.24, 10)
L_VALUES = [32, 64]
N_EVENTS = 5_000
N_SEEDS = 3


def run_ofc_phase_diagram():
    # Grid search over (L, alpha_ofc).
    # Returns (b_means, b_stds), each shape (len(L_VALUES), len(ALPHA_GRID)).
    b_means = np.full((len(L_VALUES), len(ALPHA_GRID)), np.nan)
    b_stds  = np.full((len(L_VALUES), len(ALPHA_GRID)), np.nan)

    total = len(L_VALUES) * len(ALPHA_GRID) * N_SEEDS * N_EVENTS
    with tqdm(total=total, desc="OFC phase diagram", unit="ev") as pbar:
        for i, L in enumerate(L_VALUES):
            for j, alpha_ofc in enumerate(ALPHA_GRID):
                bs = []
                for seed in range(N_SEEDS):
                    sizes = simulate_ofc(L, alpha_ofc, N_EVENTS, seed=seed, pbar=pbar)
                    sizes = sizes[sizes > 0]
                    if len(sizes) < 50:
                        continue
                    try:
                        b = gutenberg_richter_b(np.log10(sizes.astype(float)))
                        bs.append(b)
                    except Exception:
                        pass
                if bs:
                    b_means[i, j] = float(np.mean(bs))
                    b_stds[i, j]  = float(np.std(bs, ddof=1) if len(bs) > 1 else 0.0)

    return b_means, b_stds


def plot_ofc_phase_diagram(b_means, b_stds):
    # Plot b-value vs alpha_ofc for each L with std error bars.
    # L=64: solid line + circle. L=32: dashed line + square.
    # Vertical dashed line where L=64 b-value first drops below 1.0.
    _ensure_figures_dir()
    _apply_style()

    save_path = os.path.join(FIGURES_DIR, "ofc_phase_diagram.pdf")

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    colors = cm.Oranges(np.linspace(0.4, 0.9, len(L_VALUES)))

    # marker/linestyle per L value
    _style = {32: ("s", "--"), 64: ("o", "-")}

    for i, L in enumerate(L_VALUES):
        mask = ~np.isnan(b_means[i])
        yerr = b_stds[i][mask] / np.sqrt(N_SEEDS)
        marker, ls = _style.get(L, ("o", "-"))
        ax.errorbar(
            ALPHA_GRID[mask], b_means[i][mask], yerr=yerr,
            fmt=marker + ls, markersize=3, linewidth=1,
            color=colors[i], label=f"L={L}",
            capsize=2, elinewidth=0.7)

    # critical line: first alpha where L=64 b-value drops below 1.0
    if 64 in L_VALUES:
        idx64 = L_VALUES.index(64)
        b64 = b_means[idx64]
        below = (~np.isnan(b64)) & (b64 < 1.0)
        if below.any():
            alpha_crit = ALPHA_GRID[below][0]
            ax.axvline(alpha_crit, linestyle="--", color="dimgray",
                       linewidth=0.8, label="b = 1 (empirical G-R)")

    ax.set_xlabel(r"$\alpha_{\mathrm{OFC}}$")
    ax.set_ylabel("b-value (G-R)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


if __name__ == "__main__":
    b_means, b_stds = run_ofc_phase_diagram()
    plot_ofc_phase_diagram(b_means, b_stds)
