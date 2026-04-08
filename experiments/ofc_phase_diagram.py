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

# experiment parameters
ALPHA_GRID = np.linspace(0.05, 0.24, 12)
L_VALUES = [64, 128]
N_EVENTS = 15_000
N_SEEDS = 3


def run_ofc_phase_diagram():
    # grid search over (L, alpha_ofc), return mean b-value for each pair
    b_values = np.full((len(L_VALUES), len(ALPHA_GRID)), np.nan)

    total = len(L_VALUES) * len(ALPHA_GRID) * N_SEEDS
    with tqdm(total=total, desc="OFC phase diagram") as pbar:
        for i, L in enumerate(L_VALUES):
            for j, alpha_ofc in enumerate(ALPHA_GRID):
                bs = []
                for seed in range(N_SEEDS):
                    sizes = simulate_ofc(L, alpha_ofc, N_EVENTS, seed=seed)
                    sizes = sizes[sizes > 0]
                    if len(sizes) < 50:
                        pbar.update(1)
                        continue
                    mags = np.log10(sizes.astype(float))
                    try:
                        b = gutenberg_richter_b(mags)
                        bs.append(b)
                    except Exception:
                        pass
                    pbar.update(1)
                if bs:
                    b_values[i, j] = float(np.mean(bs))

    return b_values


def plot_ofc_phase_diagram(b_values):
    # plot b-value vs alpha_ofc for each L
    _ensure_figures_dir()
    _apply_style()

    save_path = os.path.join(FIGURES_DIR, "ofc_phase_diagram.pdf")

    import matplotlib.cm as cm
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    colors = cm.Oranges(np.linspace(0.4, 0.9, len(L_VALUES)))

    for i, L in enumerate(L_VALUES):
        mask = ~np.isnan(b_values[i])
        ax.plot(ALPHA_GRID[mask], b_values[i][mask], "o-",
                markersize=3, linewidth=1, color=colors[i], label=f"L={L}")

    ax.set_xlabel(r"$\alpha_\mathrm{OFC}$")
    ax.set_ylabel("b-value (G-R)")
    ax.set_title("OFC: b-value vs stress-transfer fraction")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


if __name__ == "__main__":
    b_values = run_ofc_phase_diagram()
    plot_ofc_phase_diagram(b_values)
