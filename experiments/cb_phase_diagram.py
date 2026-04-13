# CB phase diagram: sweep p and L, measure tail exponent alpha

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from models.cont_bouchaud import simulate_cb
from utils.powerlaw_fit import fit_powerlaw
from utils.plotting import (
    plot_cb_phase_diagram as _plot_cb_phase_diagram,
    FIGURES_DIR, _ensure_figures_dir)

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
    # Delegates to utils.plotting.plot_cb_phase_diagram.
    save_path = os.path.join(FIGURES_DIR, "cb_phase_diagram.pdf")
    return _plot_cb_phase_diagram(exponents, P_GRID, L_VALUES, P_C, save_path)


if __name__ == "__main__":
    exponents = run_cb_phase_diagram()
    plot_cb_phase_diagram(exponents)
