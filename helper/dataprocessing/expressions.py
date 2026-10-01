"""
Workspace expression templates and evaluation
==============================================

This module powers the GUI's "Create variable from function" dialog, which
lets you derive a new workspace variable/array from existing ones.

How to add a new template
---------------------------
Append an (name, expression) pair to EXPRESSION_TEMPLATES below. The name is
shown in the dialog's "Template" dropdown; selecting it fills the expression
field. Use placeholder identifiers (X, Y, N, ...) that the user is expected
to replace with their own workspace variable names, for example:

    EXPRESSION_TEMPLATES.append(("Difference (A - B)", "A - B"))

How expressions are evaluated
------------------------------
`evaluate_expression` runs the text with `eval()` against a namespace built
from:
  - every current workspace variable, bound by its own name
  - `np`             NumPy
  - `const`, `atom`  dictionaries from helper.physicalConstants
  - a few safe builtins: abs, min, max, sum, len, range

Arbitrary indexing such as `f[0]`, `f[1]`, `f[2]` works because workspace
variables are plain NumPy arrays.
"""

import numpy as np

from helper import physicalConstants as pC

EXPRESSION_TEMPLATES = [
    ("Custom expression", ""),
    ("Mean of X", "np.mean(X)"),
    ("Standard deviation of X", "np.std(X)"),
    ("Sum of X", "np.sum(X)"),
    ("Normalize X (X / max(X))", "X / np.max(X)"),
    ("Gaussian cloud density N / V", "N / ((2 * np.pi * const['k_B'] * T / const['m']) ** 1.5 * sx * sy * sz)"),
    ("Trap frequency omega from sigma and T", "np.sqrt(const['k_B'] * T / (const['m'] * sigma ** 2))"),
]

SAFE_BUILTINS = {"abs": abs, "min": min, "max": max, "sum": sum, "len": len, "range": range}


def evaluate_expression(expression, workspace_variables):
    """Evaluate a user-written expression against workspace variables and physical constants."""
    namespace = {name: variable["value"] for name, variable in workspace_variables.items()}
    namespace.update({"np": np, "const": pC.const, "atom": pC.atom})
    result = eval(expression, {"__builtins__": SAFE_BUILTINS}, namespace)
    return np.asarray(result, dtype=float)
