"""
Value +/- uncertainty formatting
=================================

`format_with_uncertainty` renders a value together with its standard error
using the common physics "concise notation", e.g. ``2.343(3)`` meaning
``2.343 +/- 0.003``.

Rounding rule used here (the standard simplified convention):
  - Round the error to 1 significant figure, or 2 if its leading digit is 1.
  - Round the value to the same decimal place as the rounded error.
  - Show the error's significant digit(s) in parentheses after the value.

Worked example: value=2.342532423, error=0.0032312.
The error's leading digit is 3 (not 1), so round to 1 significant figure:
0.003. The value rounded to the same (3rd) decimal place is 2.343. The
digit "3" in parentheses stands for +/-0.003 in the last shown decimal
place, giving "2.343(3)".

When the error is at or above the units place, showing it in parentheses
would be ambiguous (parentheses only make sense for digits after the
decimal point), so an explicit "value +/- error" string is used instead.
"""

import math

import numpy as np


def round_to_significant_error(value, error):
    """Round value/error to the error's significant figure(s); return (value, error, decimals)."""
    error = abs(float(error))
    exponent = math.floor(math.log10(error) + 1e-12)
    leading_digit = int(error / 10**exponent + 1e-9)
    sig_figs = 2 if leading_digit == 1 else 1
    decimals = -(exponent - (sig_figs - 1))
    rounded_error = round(error, decimals)

    # Rounding can push the error into the next decade (e.g. 0.96 -> 1.0); redo once if so.
    if rounded_error > 0:
        new_exponent = math.floor(math.log10(rounded_error) + 1e-9)
        if new_exponent != exponent:
            exponent = new_exponent
            leading_digit = int(rounded_error / 10**exponent + 1e-9)
            sig_figs = 2 if leading_digit == 1 else 1
            decimals = -(exponent - (sig_figs - 1))
            rounded_error = round(error, decimals)

    rounded_value = round(float(value), decimals)
    return rounded_value, rounded_error, decimals


def format_with_uncertainty(value, error):
    """Format ``value`` with its standard error as 'value(error)', e.g. '2.343(3)'."""
    if value is None or not np.isfinite(value):
        return str(value)
    if error is None or not np.isfinite(error) or error == 0:
        return f"{value:.6g}"

    rounded_value, rounded_error, decimals = round_to_significant_error(value, error)

    if decimals > 0:
        error_digits = int(round(rounded_error * 10**decimals))
        return f"{rounded_value:.{decimals}f}({error_digits})"
    # The uncertainty is at or above the units place; parentheses would be misleading there.
    return f"{rounded_value:.0f} \u00b1 {rounded_error:.0f}"


def mean_with_sem(values):
    """Return (mean, standard error of the mean) over the finite entries of ``values``."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(finite))
    sem = float(np.std(finite, ddof=1) / np.sqrt(finite.size)) if finite.size > 1 else 0.0
    return mean, sem
