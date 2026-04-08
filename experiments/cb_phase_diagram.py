# CB phase diagram: sweep p and L, measure tail exponent alpha

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from models.cont_bouchaud import simulate_cb
from utils.powerlaw_fit import fit_powerlaw
from utils.plotting import plot_phase_diagram, FIGURES_DIR, _apply_style, _ensure_figures_dir

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# experiment parameters
P_GRID = np.linspace(0.40, 0.70, 15)
L_VALUES = [32, 64]
N_STEPS = 2_000
A = 0.01
N_SEEDS = 5
P_C = 0.593  # 2D bond percolation threshold


def run_cb_phase_diagram():
    # grid search over (L, p), return mean alpha for each pair
    exponents = np.full((len(L_VALUES), len(P_GRID)), np.nan)

    total = len(L_VALUES) * len(P_GRID) * N_SEEDS
    with tqdm(total=total, desc="CB phase diagram") as pbar:
        for i, L in enumerate(L_VALUES):
            for j, p in enumerate(P_GRID):
                alphas = []
                for seed in range(N_SEEDS):
                    returns = simulate_cb(L, p, A, N_STEPS, seed=seed)
                    abs_ret = np.abs(returns)
                    abs_ret = abs_ret[abs_ret > 0]
                    if len(abs_ret) < 50:
                        pbar.update(1)
                        continue
                    try:
                        res = fit_powerlaw(abs_ret)
                        alphas.append(res["alpha"])
                    except Exception:
                        pass
                    pbar.update(1)
                if alphas:
                    exponents[i, j] = float(np.mean(alphas))

    return exponents


def plot_cb_phase_diagram(exponents):
    # plot alpha vs p for each L, mark percolation threshold
    _ensure_figures_dir()
    _apply_style()

    save_path = os.path.join(FIGURES_DIR, "cb_phase_diagram.pdf")

    import matplotlib.cm as cm
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    colors = cm.Blues(np.linspace(0.4, 0.9, len(L_VALUES)))

    for i, L in enumerate(L_VALUES):
        mask = ~np.isnan(exponents[i])
        ax.plot(P_GRID[mask], exponents[i][mask], "o-",
                markersize=3, linewidth=1, color=colors[i], label=f"L={L}")

    ax.axvline(P_C, linestyle="--", color="gray", linewidth=0.8,
               label=rf"$p_c={P_C}$")
    ax.set_xlabel("$p$")
    ax.set_ylabel(r"$\alpha$ (tail exponent)")
    ax.set_title("CB: tail exponent vs percolation probability")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


if __name__ == "__main__":
    exponents = run_cb_phase_diagram()
    plot_cb_phase_diagram(exponents)
