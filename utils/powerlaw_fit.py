# Power-law fitting using Clauset et al. (2009) method via the powerlaw package

import numpy as np
import powerlaw
from scipy import stats


def fit_powerlaw(data, xmin=None, discrete=False):
    # Fit a power law to data, compare against lognormal and exponential.
    # xmin: if provided, data is filtered to data >= xmin and xmin is fixed in the fit.
    # discrete: passed to powerlaw.Fit (use True for integer-valued data).
    data = np.asarray(data, dtype=float)
    data = data[data > 0]
    if data.size == 0:
        raise ValueError("No positive values found in data.")

    if xmin is not None:
        data = data[data >= xmin]
        fit = powerlaw.Fit(data, xmin=xmin, discrete=discrete, verbose=False)
    else:
        fit = powerlaw.Fit(data, discrete=discrete, verbose=False)

    xmin  = fit.power_law.xmin
    alpha = fit.power_law.alpha

    # Manual KS test against the theoretical power-law CDF on the tail.
    # fit.power_law.p_value() is broken in some powerlaw versions (returns nan),
    # so we use scipy.stats.kstest directly.
    data_tail = data[data >= xmin]
    ks_stat, p_val = stats.kstest(
        data_tail,
        lambda x: 1.0 - (xmin / x) ** (alpha - 1)
    )

    # Truncated power law parameters
    alpha_tpl  = fit.truncated_power_law.alpha
    lambda_tpl = fit.truncated_power_law.Lambda

    # likelihood ratio tests vs alternatives
    R_ln,  p_ln  = fit.distribution_compare("power_law", "lognormal",          normalized_ratio=True)
    R_exp, p_exp = fit.distribution_compare("power_law", "exponential",         normalized_ratio=True)
    R_pl_vs_tpl,  p_pl_vs_tpl  = fit.distribution_compare("power_law",          "truncated_power_law", normalized_ratio=True)
    R_tpl_vs_ln,  p_tpl_vs_ln  = fit.distribution_compare("truncated_power_law","lognormal",           normalized_ratio=True)

    return {
        "alpha": fit.power_law.alpha,
        "xmin": fit.power_law.xmin,
        "sigma": fit.power_law.sigma,
        "alpha_tpl": alpha_tpl,
        "lambda_tpl": lambda_tpl,
        "KS_statistic": ks_stat,
        "p_value": p_val,
        "R_lognormal": R_ln,
        "p_lognormal": p_ln,
        "R_exponential": R_exp,
        "p_exponential": p_exp,
        "R_pl_vs_tpl": R_pl_vs_tpl,
        "p_pl_vs_tpl": p_pl_vs_tpl,
        "R_tpl_vs_lognormal": R_tpl_vs_ln,
        "p_tpl_vs_lognormal": p_tpl_vs_ln,
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
