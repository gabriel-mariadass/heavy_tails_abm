# OFC calibration: find alpha_ofc that matches empirical G-R b-value via MLE or ABC

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from models.ofc import simulate_ofc
from data.download_data import download_usgs_catalog
from utils.powerlaw_fit import gutenberg_richter_b
from utils.plotting import (
    plot_calibration_mle,
    plot_abc_posterior,
    FIGURES_DIR,
    _ensure_figures_dir,
)

# MLE grid search parameters
ALPHA_GRID_MLE = np.linspace(0.10, 0.24, 20)
L_OFC = 128
N_EVENTS = 50_000
N_SEEDS_MLE = 20

# ABC parameters
ALPHA_LOW, ALPHA_HIGH = 0.10, 0.24
ABC_POP_SIZE = 100
ABC_MAX_POP = 5


def _estimate_b_ofc(alpha_ofc, n_seeds=N_SEEDS_MLE):
    # run OFC at given alpha_ofc, return mean b-value over multiple seeds
    bs = []
    for seed in range(n_seeds):
        sizes = simulate_ofc(L_OFC, alpha_ofc, N_EVENTS, seed=seed)
        sizes = sizes[sizes > 0]
        if len(sizes) < 50:
            continue
        mags = np.log10(sizes.astype(float))
        try:
            b = gutenberg_richter_b(mags)
            bs.append(b)
        except Exception:
            pass
    return float(np.mean(bs)) if bs else float("nan")


def run_mle_calibration(b_emp):
    # grid search over alpha_ofc, find value closest to empirical b
    b_sim_grid = np.full(len(ALPHA_GRID_MLE), np.nan)

    for j, alpha_ofc in enumerate(tqdm(ALPHA_GRID_MLE, desc="OFC MLE calibration")):
        b_sim_grid[j] = _estimate_b_ofc(alpha_ofc)

    valid = ~np.isnan(b_sim_grid)
    alpha_star = ALPHA_GRID_MLE[valid][np.argmin(np.abs(b_sim_grid[valid] - b_emp))]

    save_path = os.path.join(FIGURES_DIR, "ofc_calibration_mle.pdf")
    plot_calibration_mle(
        param_grid=ALPHA_GRID_MLE,
        alpha_sim=b_sim_grid,
        alpha_emp=b_emp,
        param_name=r"\alpha_\mathrm{OFC}",
        emp_label="Empirical b (USGS)",
        title="OFC MLE calibration",
        save_path=save_path,
        p_star=alpha_star)
    print(f"  alpha_ofc* = {alpha_star:.4f}  |  Saved: {save_path}")
    return alpha_star, b_sim_grid


def run_abc_calibration(b_emp):
    # ABC-SMC to get posterior distribution over alpha_ofc
    import pyabc

    def model(params):
        alpha_ofc = params["alpha_ofc"]
        b_sim = _estimate_b_ofc(alpha_ofc, n_seeds=5)
        return {"b": b_sim}

    def distance(sim, obs):
        return abs(sim["b"] - obs["b"])

    prior = pyabc.Distribution(
        alpha_ofc=pyabc.RV("uniform", ALPHA_LOW, ALPHA_HIGH - ALPHA_LOW))

    abc = pyabc.ABCSMC(
        models=model,
        parameter_priors=prior,
        distance_function=distance,
        population_size=ABC_POP_SIZE)

    obs = {"b": b_emp}
    db_path = "sqlite:///ofc_abc.db"
    abc.new(db_path, obs)

    history = abc.run(
        minimum_epsilon=0.05,
        max_nr_populations=ABC_MAX_POP)

    df, w = history.get_distribution(m=0)
    samples = df["alpha_ofc"].values
    weights = w

    save_path = os.path.join(FIGURES_DIR, "ofc_calibration_abc.pdf")
    plot_abc_posterior(
        samples=samples,
        weights=weights,
        param_name=r"\alpha_\mathrm{OFC}",
        title="OFC ABC posterior",
        save_path=save_path)
    print(f"  Saved ABC posterior: {save_path}")

    # remove temp DB files
    import glob
    for f in glob.glob("ofc_abc*.db"):
        try:
            os.remove(f)
        except OSError:
            pass


def main():
    _ensure_figures_dir()

    print("Downloading USGS earthquake catalog …")
    catalog = download_usgs_catalog(
        min_magnitude=2.0,
        start="2000-01-01",
        end="2024-01-01")
    magnitudes = catalog["magnitude"].dropna().values
    print(f"  {len(magnitudes)} events, M in [{magnitudes.min():.1f}, {magnitudes.max():.1f}]")

    b_emp = gutenberg_richter_b(magnitudes)
    print(f"  Empirical b = {b_emp:.4f}")

    run_mle_calibration(b_emp)

    print("Running ABC calibration …")
    try:
        run_abc_calibration(b_emp)
    except Exception as e:
        print(f"  ABC failed: {e}")


if __name__ == "__main__":
    main()
