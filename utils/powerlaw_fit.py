# Power-law fitting using Clauset et al. (2009) method via the powerlaw package

import numpy as np
import powerlaw


def fit_powerlaw(data):
    # Fit a power law to data, compare against lognormal and exponential
    data = np.asarray(data, dtype=float)
    data = data[data > 0]
    if data.size == 0:
        raise ValueError("No positive values found in data.")

    fit = powerlaw.Fit(data, discrete=False, verbose=False)

    # Note: fit.power_law.KS() has a bug in some powerlaw versions (calls
    # compute_distance_metrics without self), so we read D directly. It is
    # computed during powerlaw.Fit() and is identical to what KS() returns.
    ks_stat = fit.power_law.D

    # bootstrap p-value, 100 samples to keep it fast
    try:
        p_val = fit.power_law.p_value(data, n_iter=100)
    except Exception:
        p_val = float("nan")

    # likelihood ratio tests vs alternatives
    R_ln, p_ln = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
    R_exp, p_exp = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)

    return {
        "alpha": fit.power_law.alpha,
        "xmin": fit.power_law.xmin,
        "sigma": fit.power_law.sigma,
        "KS_statistic": ks_stat,
        "p_value": p_val,
        "R_lognormal": R_ln,
        "p_lognormal": p_ln,
        "R_exponential": R_exp,
        "p_exponential": p_exp,
        "fit": fit}


def gutenberg_richter_b(magnitudes, m_min=None):
    # Estimate G-R b-value by MLE: b = log10(e) / (mean(M) - M_min)
    # Input must already be in magnitude units (Richter scale or log10(sizes)).
    magnitudes = np.asarray(magnitudes, dtype=float)
    magnitudes = magnitudes[np.isfinite(magnitudes)]
    if magnitudes.size == 0:
        raise ValueError("No valid magnitudes.")

    if m_min is None:
        m_min = magnitudes.min()
    magnitudes = magnitudes[magnitudes >= m_min]

    mean_m = magnitudes.mean()
    if mean_m <= m_min:
        raise ValueError("mean(M) must be greater than M_min for MLE b estimation.")

    b = np.log10(np.e) / (mean_m - m_min)
    return float(b)
