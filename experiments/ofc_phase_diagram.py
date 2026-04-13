# OFC phase diagram: sweep alpha_ofc and L, measure G-R b-value

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from models.ofc import simulate_ofc
from utils.powerlaw_fit import gutenberg_richter_b
from utils.plotting import (
    plot_ofc_phase_diagram as _plot_ofc_phase_diagram,
    FIGURES_DIR, _ensure_figures_dir)

# experiment parameters
ALPHA_GRID = np.linspace(0.10, 0.24, 20)
L_VALUES = [32, 64, 128]
N_EVENTS = 50_000
N_SEEDS = 10


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
                        b = gutenberg_richter_b(
                            np.log10(sizes.astype(float)))
                        bs.append(b)
                    except Exception:
                        pass
                if bs:
                    b_means[i, j] = float(np.mean(bs))
                    b_stds[i, j]  = float(np.std(bs, ddof=1) if len(bs) > 1 else 0.0)

    return b_means, b_stds


def plot_ofc_phase_diagram(b_means, b_stds):
    # Delegates to utils.plotting.plot_ofc_phase_diagram.
    save_path = os.path.join(FIGURES_DIR, "ofc_phase_diagram.pdf")
    return _plot_ofc_phase_diagram(
        b_means, b_stds, ALPHA_GRID, L_VALUES, N_SEEDS, save_path)


if __name__ == "__main__":
    b_means, b_stds = run_ofc_phase_diagram()
    plot_ofc_phase_diagram(b_means, b_stds)
