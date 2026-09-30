import os

import numpy as np
import pandas as pd

from helper.config import GEN_PARAMS, VAL_NAME
from helper.physDataproc import load_multiple_files


def detect_scan_parameters(file_paths):
    for file_path in file_paths:
        try:
            data = pd.read_csv(file_path, encoding="utf-8", skipinitialspace=True)
        except Exception:
            continue
        if data.empty:
            continue
        params = data.drop(columns=VAL_NAME, errors="ignore")
        params_diff = params.loc[:, params.nunique() > 1]
        scan_params = params_diff.drop(columns=[col for col in GEN_PARAMS if col in params_diff.columns], errors="ignore")
        if not scan_params.empty:
            return scan_params.columns.tolist()
    return []


def _numeric_or_text(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def prepare_filtered_data(filedata, main_name=None, other_name=None, filter_mode="All points", other_value=None, value_column="atom_number_fit"):
    if filedata is None or filedata["data"].empty:
        return np.array([]), np.array([]), np.array([]), "Sample index", "Sample value"

    data = filedata["data"].copy()
    if value_column in data.columns:
        y = data[value_column].to_numpy(dtype=float)
        y_label = value_column
    else:
        y = np.arange(len(data), dtype=float)
        y_label = "Sample value"

    if main_name is not None and main_name in data.columns:
        x = data[main_name].to_numpy(dtype=float)
        x_label = main_name
    else:
        x = np.arange(len(data), dtype=float)
        x_label = "Sample index"

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if other_name is not None and other_name in data.columns and filter_mode != "All points":
        other_values = data.loc[mask, other_name].to_numpy()
        if filter_mode == "Average over other scan parameter":
            grouped = pd.DataFrame({"x": x, "y": y, "other": other_values}).groupby("x", as_index=False).agg(
                y_mean=("y", "mean"),
                y_err=("y", lambda values: values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0),
            )
            return grouped["x"].to_numpy(), grouped["y_mean"].to_numpy(), grouped["y_err"].to_numpy(), x_label, y_label
        if filter_mode == "Specific value":
            if other_value is None:
                raise ValueError("Please choose the value of the other scan parameter to analyse.")
            target_value = _numeric_or_text(other_value)
            match = np.asarray([_numeric_or_text(value) == target_value for value in other_values], dtype=bool)
            if not np.any(match):
                raise ValueError(f"No data points were found for {other_name} = {other_value}.")
            return x[match], y[match], np.zeros(np.sum(match)), x_label, y_label

    return x, y, np.zeros_like(y), x_label, y_label


def load_selected_data(file_paths, scan_var_names=None):
    if not file_paths:
        return None
    directories = [os.path.dirname(path) for path in file_paths]
    filenames = [os.path.splitext(os.path.basename(path))[0] for path in file_paths]
    selected_scan_names = [name for name in (scan_var_names or []) if name]
    selected_scan_names = selected_scan_names[:2]
    return load_multiple_files(
        directories,
        filenames,
        VAL_NAME,
        GEN_PARAMS,
        scanAuto=False,
        b2D=len(selected_scan_names) >= 2,
        scanVarNames=selected_scan_names or None,
    )
