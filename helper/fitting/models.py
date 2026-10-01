"""Numerical models used for decay and time-of-flight analysis."""

import numpy as np
from scipy.optimize import curve_fit

from helper import physicalConstants as pC


def fit_exponential_decay(x_vals, y_vals):
    """Fit ``y = amplitude * exp(-x / tau) + offset``; return (popt, standard errors)."""
    x_vals, y_vals = _finite_arrays(x_vals, y_vals)
    _require_fit_points(x_vals)
    x_range = np.ptp(x_vals)
    if x_range == 0:
        raise ValueError("The chosen scan parameter does not vary; exponential decay fitting is not possible.")
    p0 = [max(max(y_vals) - min(y_vals), 1e-6), min(y_vals), max(x_range, 1e-6)]
    popt, pcov = curve_fit(lambda t, amp, offset, tau: amp * np.exp(-t / tau) + offset, x_vals, y_vals, p0=p0, bounds=([0, -np.inf, 1e-12], [np.inf, np.inf, np.inf]), maxfev=100000)
    return popt, np.sqrt(np.diag(pcov))


def fit_exponential_decay_no_offset(x_vals, y_vals):
    """Fit a zero-offset exponential lifetime model; return (popt, standard errors)."""
    x_vals, y_vals = _finite_arrays(x_vals, y_vals)
    _require_fit_points(x_vals)
    x_range = np.ptp(x_vals)
    if x_range == 0:
        raise ValueError("The chosen scan parameter does not vary; exponential decay fitting is not possible.")
    popt, pcov = curve_fit(lambda t, amp, tau: amp * np.exp(-t / tau), x_vals, y_vals, p0=[max(max(y_vals), 1e-6), max(x_range, 1e-6)], bounds=([0, 1e-12], [np.inf, np.inf]), maxfev=100000)
    return popt, np.sqrt(np.diag(pcov))


def tof_para(t, temperature, sigma0, constants, atom):
    """Calculate cloud width during time of flight for a temperature and atom."""
    return np.sqrt(sigma0**2 + 2 * constants["k_B"] * temperature / atom["m"] * (t / 1000) ** 2)


def fit_tof_temperature(x_vals, sigma_vals):
    """Estimate temperature and initial width (with standard errors) from sigma^2 vs t^2."""
    x_vals, sigma_vals = _finite_arrays(x_vals, sigma_vals)
    _require_fit_points(x_vals, minimum=3)
    (slope, intercept), covariance = np.polyfit((x_vals / 1000) ** 2, sigma_vals**2, 1, cov=True)
    if slope <= 0:
        raise ValueError("Fitted TOF slope is non-positive.")
    slope_err, intercept_err = np.sqrt(np.diag(covariance))

    sigma0 = np.sqrt(max(intercept, 0))
    # Error propagation through sqrt diverges at sigma0 = 0, so fall back to 0 there.
    sigma0_err = intercept_err / (2 * sigma0) if sigma0 > 0 else 0.0

    scale = pC.atom["K41"]["m"] / (2 * pC.const["k_B"])
    temperature = slope * scale
    temperature_err = slope_err * scale

    return np.array([temperature, sigma0]), np.array([temperature_err, sigma0_err])


def _finite_arrays(first, second):
    """Return paired numeric arrays with all non-finite entries removed."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    mask = np.isfinite(first) & np.isfinite(second)
    return first[mask], second[mask]


def _require_fit_points(values, minimum=3):
    """Raise an informative error when a fit has too few data points."""
    if len(values) < minimum:
        raise ValueError(f"At least {minimum} points are required for an exponential fit.")
