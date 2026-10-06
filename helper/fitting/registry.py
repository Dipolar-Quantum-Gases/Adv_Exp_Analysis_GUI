"""
Fit function registry
======================

Every "Fit type" dropdown in the GUI (the 1D plot panel and the workspace
plot panel) is built from FIT_DEFINITIONS below, so adding a fit here makes
it available everywhere without touching gui.py.

How to add a new fit
---------------------
1. In helper/fitting/models.py, write a `fit(x, y) -> (popt, perr)` function
   that returns the fitted parameters and their standard errors (typically
   via scipy.optimize.curve_fit's covariance matrix).
2. Write a `model(x, *popt) -> y` callable (a plain function or a lambda)
   that evaluates the fitted curve at new x-values.
3. Append a FitDefinition to FIT_DEFINITIONS below, for example:

    FitDefinition(
        name="Linear fit",                       # shown in the dropdowns
        slug="linear",                            # prefix for result-table rows
        fit=my_linear_fit,
        model=lambda x, slope, intercept: slope * x + intercept,
        param_names=("slope", "intercept"),
        label=lambda source, popt, perr: f"{source}: slope={format_with_uncertainty(popt[0], perr[0])}",
    )

That's it. The plotting code looks up the definition by name, calls `fit`,
evaluates `model` to draw the curve, uses `label` for the legend text, and
records every parameter (with its standard error) into the collected-results
table automatically.
"""

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from helper import physicalConstants as pC
from helper.fitting.models import (
    fit_exponential_decay,
    fit_exponential_decay_no_offset,
    fit_tof_temperature,
    get_tof_atom,
    tof_para,
)
from helper.formatting import format_with_uncertainty


@dataclass
class FitDefinition:
    """Describes one selectable fit type: how to fit it, draw it, and report it."""

    name: str
    slug: str
    fit: Callable[[np.ndarray, np.ndarray], tuple]
    model: Callable[..., np.ndarray]
    param_names: Sequence[str]
    label: Callable[[str, Sequence[float], Sequence[float]], str]
    units: Sequence[str] = field(default_factory=tuple)

    def record_rows(self, source, popt, perr):
        """Build one result row per parameter, each carrying its standard error."""
        units = self.units or [""] * len(self.param_names)
        return [
            {"source": source, "quantity": f"{self.slug}_{param_name}", "value": value, "error": error, "unit": unit, "origin": "1D fit"}
            for param_name, value, error, unit in zip(self.param_names, popt, perr, units)
        ]


FIT_DEFINITIONS = [
    FitDefinition(
        name="Exponential decay (with offset)",
        slug="decay",
        fit=fit_exponential_decay,
        model=lambda x, amplitude, offset, tau: amplitude * np.exp(-x / tau) + offset,
        param_names=("amplitude", "offset", "tau"),
        label=lambda source, popt, perr: f"{source} fit: τ={format_with_uncertainty(popt[2], perr[2])}",
    ),
    FitDefinition(
        name="Exponential decay (lifetime)",
        slug="lifetime",
        fit=fit_exponential_decay_no_offset,
        model=lambda x, amplitude, tau: amplitude * np.exp(-x / tau),
        param_names=("amplitude", "tau"),
        label=lambda source, popt, perr: f"{source} lifetime: τ={format_with_uncertainty(popt[1], perr[1])}",
    ),
    FitDefinition(
        name="TOF temperature",
        slug="tof",
        fit=fit_tof_temperature,
        model=lambda x, temperature, sigma0: tof_para(x, temperature, sigma0, pC.const, pC.atom[get_tof_atom()]),
        param_names=("temperature", "sigma0"),
        units=("K", "m"),
        label=lambda source, popt, perr: f"{source} TOF: T={format_with_uncertainty(popt[0] * 1e6, perr[0] * 1e6)} µK",
    ),
]

FIT_NAMES = ["No fit"] + [definition.name for definition in FIT_DEFINITIONS]


def get_fit_definition(name):
    """Look up a FitDefinition by its dropdown name, or None for 'No fit'/unknown names."""
    return next((definition for definition in FIT_DEFINITIONS if definition.name == name), None)
