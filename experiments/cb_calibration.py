# CB calibration: find p that matches empirical tail exponent via MLE or ABC

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.cont_bouchaud import simulate_cb
from data.download_data import download_stock_returns
from utils.powerlaw_fit import fit_powerlaw
from utils.plotting import (
    plot_calibration_mle,
    plot_abc_posterior,
    FIGURES_DIR,
    _ensure_figures_dir)

# MLE grid search parameters
P_GRID_MLE = np.linspace(0.40, 0.70, 30)
L_CB = 128
A = 0.01
N_STEPS = 5_000
N_SEEDS_MLE = 20

# ABC parameters
P_LOW, P_HIGH = 0.40, 0.70
ABC_POP_SIZE = 100
ABC_MAX_POP = 5


def _estimate_alpha_cb(p, n_seeds=N_SEEDS_MLE):
    # run CB at given p, return mean tail exponent over multiple seeds
    alphas = []
    for seed in range(n_seeds):
        returns = simulate_cb(L_CB, p, A, N_STEPS, seed=seed)
        abs_ret = np.abs(returns[returns != 0])
        if len(abs_ret) < 50:
            continue
        try:
            res = fit_powerlaw(abs_ret)
            alphas.append(res["alpha"])
        except Exception:
            pass
    return float(np.mean(alphas)) if alphas else float("nan")


def run_mle_calibration(alpha_emp, ticker):
    # grid search over p, find p* closest to empirical alpha
    alpha_sim_grid = np.full(len(P_GRID_MLE), np.nan)

    for j, p in enumerate(tqdm(P_GRID_MLE, desc=f"MLE calibration ({ticker})")):
        alpha_sim_grid[j] = _estimate_alpha_cb(p)

    valid = ~np.isnan(alpha_sim_grid)
    p_star = P_GRID_MLE[valid][np.argmin(np.abs(alpha_sim_grid[valid] - alpha_emp))]

    save_path = os.path.join(FIGURES_DIR, f"cb_calibration_mle_{ticker.replace('^','')}.pdf")
    plot_calibration_mle(
        param_grid=P_GRID_MLE,
        alpha_sim=alpha_sim_grid,
        alpha_emp=alpha_emp,
        param_name="p",
        emp_label=f"Empirical ({ticker})",
        title=f"CB MLE calibration - {ticker}",
        save_path=save_path,
        p_star=p_star)
    print(f"  p* = {p_star:.4f}  -  Saved: {save_path}")
    return p_star, alpha_sim_grid


def run_abc_calibration(alpha_emp, ticker):
    # ABC-SMC to get posterior distribution over p
    import pyabc

    def model(params):
        p = params["p"]
        alpha_sim = _estimate_alpha_cb(p, n_seeds=5)
        return {"alpha": alpha_sim}

    def distance(sim, obs):
        return abs(sim["alpha"] - obs["alpha"])

    prior = pyabc.Distribution(p=pyabc.RV("uniform", P_LOW, P_HIGH - P_LOW))

    abc = pyabc.ABCSMC(
        models=model,
        parameter_priors=prior,
        distance_function=distance,
        population_size=ABC_POP_SIZE)

    obs = {"alpha": alpha_emp}
    db_path = f"sqlite:///cb_abc_{ticker.replace('^','')}.db"
    abc.new(db_path, obs)

    history = abc.run(
        minimum_epsilon=0.05,
        max_nr_populations=ABC_MAX_POP)

    df, w = history.get_distribution(m=0)
    samples = df["p"].values
    weights = w

    save_path = os.path.join(FIGURES_DIR, f"cb_calibration_abc_{ticker.replace('^','')}.pdf")
    plot_abc_posterior(
        samples=samples,
        weights=weights,
        param_name="p",
        title=f"CB ABC posterior - {ticker}",
        save_path=save_path)
    print(f"  Saved ABC posterior: {save_path}")

    # remove temp DB files
    import glob
    for f in glob.glob("cb_abc_*.db"):
        try:
            os.remove(f)
        except OSError:
            pass


def main():
    _ensure_figures_dir()

    print("Downloading stock data …")
    returns_dict = download_stock_returns(
        tickers=["^FCHI", "^GSPC"],
        start="2000-01-01",
        end="2024-01-01")

    for ticker, returns in returns_dict.items():
        print(f"\n=== {ticker} ===")
        abs_ret = np.abs(returns.dropna().values)
        abs_ret = abs_ret[abs_ret > 0]

        res = fit_powerlaw(abs_ret)
        alpha_emp = res["alpha"]
        print(f"  Empirical alpha = {alpha_emp:.4f} (xmin={res['xmin']:.6f})")

        run_mle_calibration(alpha_emp, ticker)

        print(f"  Running ABC calibration for {ticker} …")
        try:
            run_abc_calibration(alpha_emp, ticker)
        except Exception as e:
            print(f"  ABC failed: {e}")


if __name__ == "__main__":
    main()
