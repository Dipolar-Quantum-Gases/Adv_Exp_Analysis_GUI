# -*- coding: utf-8 -*-
"""
Created on Tue May 20 16:58:38 2025

@author: klaus

Heavily use of integrated VS Code AI to create this analysis GUI for 2D experimental data.
"""
################################################
#### Importing libraries
################################################

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from helper.config import DEFAULT_CSV_DIR, GEN_PARAMS, SAMPLE, VAL_NAME
from helper.dataprocessing.analysis import (
    detect_scan_parameters,
    load_selected_data,
    prepare_filtered_data as _prepare_filtered_data,
)
from helper.fitting.models import (
    fit_exponential_decay,
    fit_exponential_decay_no_offset,
    fit_tof_temperature,
    tof_para,
)
from helper import physicalConstants as pC
from helper.physDataproc import get_ExpObserv
from helper.physPlot import plot_2Ddata_heatmap, plot_2Ddata_scatter
from helper.plotting.analysis import plot_loaded_data

plt.close("all")

################################################
#### Analysis defaults
################################################

################################################
#### GUI helper functions
################################################


def _legacy_detect_scan_parameters(file_paths):
    """Look at the first valid CSV file and extract the available scan-parameter columns."""
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


def _legacy_prepare_filtered_data(filedata, main_name=None, other_name=None, filter_mode="All points", other_value=None, value_column="atom_number_fit"):
    """Select the x and y arrays for a 1D analysis, optionally averaging over or slicing by the second scan parameter."""
    if filedata is None:
        return np.array([]), np.array([]), np.array([]), "Sample index"

    data = filedata["data"].copy()
    if data.empty:
        return np.array([]), np.array([]), np.array([]), "Sample index"

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
    x = x[mask]
    y = y[mask]

    if other_name is not None and other_name in data.columns and filter_mode != "All points":
        other_values = data.loc[mask, other_name].to_numpy()
        if filter_mode == "Average over other scan parameter":
            df = pd.DataFrame({"x": x, "y": y, "other": other_values})
            grouped = df.groupby("x", as_index=False).agg(y_mean=("y", "mean"), y_err=("y", lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0))
            return grouped["x"].to_numpy(dtype=float), grouped["y_mean"].to_numpy(dtype=float), grouped["y_err"].to_numpy(dtype=float), x_label, y_label

        if filter_mode == "Specific value":
            if other_value is None:
                raise ValueError("Please choose the value of the other scan parameter to analyse.")
            target_value = _numeric_or_text(other_value)
            match = np.asarray([_numeric_or_text(v) == target_value for v in other_values], dtype=bool)
            if not np.any(match):
                raise ValueError(f"No data points were found for {other_name} = {other_value}.")
            return x[match], y[match], np.zeros(np.sum(match), dtype=float), x_label, y_label

    return x, y, np.zeros_like(y, dtype=float), x_label, y_label

def _legacy_load_selected_data(file_paths, scan_var_names=None):
    """Load selected CSV files using the same logic as the static script."""
    if not file_paths:
        return None

    directories = []
    filenames = []
    for file_path in file_paths:
        directories.append(os.path.dirname(file_path))
        filenames.append(os.path.splitext(os.path.basename(file_path))[0])

    selected_scan_names = [] if scan_var_names is None else [name for name in scan_var_names if name]
    b2d = len(selected_scan_names) >= 2
    if len(selected_scan_names) > 2:
        selected_scan_names = selected_scan_names[:2]

    filedata = load_multiple_files(
        directories,
        filenames,
        VAL_NAME,
        GEN_PARAMS,
        scanAuto=False,
        b2D=b2d,
        scanVarNames=selected_scan_names if selected_scan_names else None,
    )
    return filedata


def _legacy_plot_loaded_data(filedata, scan_var_names=None, plot_kind="1d"):
    """Generate the requested plots from the loaded data."""
    if filedata is None:
        return

    atomnumber, sigma_x, sigma_y, x0, y0 = get_ExpObserv(filedata["values"], sample_factor=SAMPLE)
    plot_names = [] if scan_var_names is None else [name for name in scan_var_names if name]

    if plot_kind == "1d":
        if len(plot_names) >= 1 and filedata.get("scanVar1") is not None:
            x_vals = np.asarray(filedata["scanVar1"], dtype=float)
            x_label = plot_names[0]
        else:
            x_vals = np.arange(len(atomnumber), dtype=float)
            x_label = "Sample index"

        fig = plt.figure()
        if len(plot_names) >= 1 and filedata.get("scanVar1") is not None:
            x_vals = np.asarray(filedata["scanVar1"], dtype=float)
            x_label = plot_names[0]
        else:
            x_vals = np.arange(len(atomnumber), dtype=float)
            x_label = "Sample index"

        fig = plt.figure()
        if config["mode"] == "Average over other scan parameter" and np.any(y_err > 0):
            plt.errorbar(x_vals, y_vals, yerr=y_err, fmt="o", capsize=4, label="Averaged data")
        else:
            plt.scatter(x_vals, y_vals, s=18)
        plt.xlabel(x_label)
        plt.ylabel("Atom Number")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return

    if len(plot_names) < 2 or filedata.get("scanVar2") is None:
        raise ValueError("Two scan parameters are required for 2D analysis.")

    scan_var_1 = np.asarray(filedata["scanVar1"], dtype=float)
    scan_var_2 = np.asarray(filedata["scanVar2"], dtype=float)
    scan_param_name_1 = plot_names[0]
    scan_param_name_2 = plot_names[1]

    if plot_kind == "2d_scatter":
        plot_2Ddata_scatter(
            scan_var_1,
            scan_var_2,
            atomnumber,
            paramName1=scan_param_name_1,
            paramName2=scan_param_name_2,
            ylabel="Atom Number",
            cMap="plasma",
            grid=True,
            legend=True,
            legendLoc="outside",
            show=False,
            bAvg=True,
        )
        plt.show()
        return

    if plot_kind == "2d_heatmap":
        plot_2Ddata_heatmap(
            scan_var_1,
            scan_var_2,
            atomnumber,
            paramName1=scan_param_name_1,
            paramName2=scan_param_name_2,
            paramNameC="Atom Number",
            cMap="plasma",
            grid=False,
            show=False,
        )
        plt.show()
        return

    raise ValueError(f"Unknown plot kind: {plot_kind}")


def _legacy_fit_exponential_decay(x_vals, y_vals):
    """Fit y = A * exp(-x / tau) + offset."""
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    finite = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[finite]
    y_vals = y_vals[finite]

    if len(x_vals) < 3:
        raise ValueError("At least three points are required for an exponential fit.")

    x_range = np.ptp(x_vals)
    if x_range == 0:
        raise ValueError("The chosen scan parameter does not vary; exponential decay fitting is not possible.")

    y_amp = max(y_vals) - min(y_vals)
    y_offset = min(y_vals)
    tau_guess = max(x_range, 1e-6)
    p0 = [max(y_amp, 1e-6), y_offset, tau_guess]
    lower_bounds = [0.0, -np.inf, 1e-12]
    upper_bounds = [np.inf, np.inf, np.inf]

    popt, _ = curve_fit(
        lambda t, amp, offset, tau: amp * np.exp(-t / tau) + offset,
        x_vals,
        y_vals,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=100000,
    )
    return popt


def _legacy_fit_exponential_decay_no_offset(x_vals, y_vals):
    """Fit y = A * exp(-x / tau) (standard lifetime, no offset)."""
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    finite = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[finite]
    y_vals = y_vals[finite]

    if len(x_vals) < 3:
        raise ValueError("At least three points are required for an exponential fit.")

    x_range = np.ptp(x_vals)
    if x_range == 0:
        raise ValueError("The chosen scan parameter does not vary; exponential decay fitting is not possible.")

    y_amp = max(y_vals)
    tau_guess = max(x_range, 1e-6)
    p0 = [max(y_amp, 1e-6), tau_guess]
    lower_bounds = [0.0, 1e-12]
    upper_bounds = [np.inf, np.inf]

    popt, _ = curve_fit(
        lambda t, amp, tau: amp * np.exp(-t / tau),
        x_vals,
        y_vals,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=100000,
    )
    return popt

def _legacy_tof_para(t, T, sigma0, pC, atom):
    sigma = np.sqrt(sigma0 ** 2 + (2 * pC["k_B"] * T / atom["m"] * (t / 1000) ** 2))
    return sigma

def _legacy_fit_tof_temperature(x_vals, sigma_vals):
    """Fit TOF broadening:
    
        sigma^2 = sigma0^2 + (2*k_B*T/m) * t^2

    x_vals are assumed to be in ms.
    sigma_vals are assumed to be in metres.
    """
    x_vals = np.asarray(x_vals, dtype=float)
    sigma_vals = np.asarray(sigma_vals, dtype=float)

    finite = np.isfinite(x_vals) & np.isfinite(sigma_vals)
    x_vals = x_vals[finite]
    sigma_vals = sigma_vals[finite]

    if len(x_vals) < 3:
        raise ValueError("At least three points are required.")

    pCon = pC.const
    atom = pC.atom["K41"]

    # Convert ms -> seconds
    t = x_vals / 1000.0

    # Fit y = intercept + slope*x
    x = t**2
    y = sigma_vals**2

    slope, intercept = np.polyfit(x, y, 1)

    if slope <= 0:
        raise ValueError("Fitted TOF slope is non-positive.")

    if intercept < 0:
        # Physically sigma0^2 cannot be negative
        sigma0 = 0.0
    else:
        sigma0 = np.sqrt(intercept)

    T = slope * atom["m"] / (2 * pCon["k_B"])

    return np.array([T, sigma0])

class DataAnalysisGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_files = []
        self.setWindowTitle("Experiment Data Analysis")
        self.resize(1200, 700)

        self.file_metadata = {}
        self.current_file_path = None
        self._custom_file_selection = []

        # Some constants/variables
        self.yparams = ["pixel_sum",
                        "i0f","x0","y0",
                        "sig_x","sig_y",
                        "foffset",
                        "theta",
                        "sig_xx", "sig_yy",
                        "icount",
                        "atom_number_fit"]

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(12)

        splitter = QSplitter(self)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter)

        left_widget = QWidget(self)
        middle_widget = QWidget(self)
        right_widget = QWidget(self)
        splitter.addWidget(left_widget)
        splitter.addWidget(middle_widget)
        splitter.addWidget(right_widget)

        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        middle_layout = QVBoxLayout(middle_widget)
        middle_layout.setContentsMargins(8, 8, 8, 8)
        middle_layout.setSpacing(10)

        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        left_group = QGroupBox("Load data")
        left_group_layout = QVBoxLayout(left_group)
        left_group_layout.setSpacing(8)
        left_layout.addWidget(left_group)

        self.status_label = QLabel("No CSV files selected.")
        left_group_layout.addWidget(self.status_label)

        self.load_button = QPushButton("Select CSV files")
        self.load_button.clicked.connect(self.select_csv_files)
        left_group_layout.addWidget(self.load_button)

        self.remove_file_button = QPushButton("Remove selected CSV")
        self.remove_file_button.clicked.connect(self.remove_selected_csv_file)
        left_group_layout.addWidget(self.remove_file_button)

        self.file_list = QListWidget(self)
        self.file_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.file_list.itemClicked.connect(self.on_file_list_clicked)
        left_group_layout.addWidget(self.file_list)

        self.file_selector = QComboBox(self)
        self.file_selector.addItem("All selected files")
        self.file_selector.addItem("Selected file(s) in list")
        left_group_layout.addWidget(self.file_selector)

        self.metadata_row = QHBoxLayout()
        self.set_name_button = QPushButton("Set file label")
        self.set_name_button.clicked.connect(self.set_selected_file_name)
        self.metadata_row.addWidget(self.set_name_button)

        self.set_color_button = QPushButton("Set file color")
        self.set_color_button.clicked.connect(self.set_selected_file_color)
        self.metadata_row.addWidget(self.set_color_button)
        left_group_layout.addLayout(self.metadata_row)

        self.use_selected_files_button = QPushButton("Use selected files for plot")
        self.use_selected_files_button.clicked.connect(self.use_selected_files_for_plot)
        left_group_layout.addWidget(self.use_selected_files_button)

        middle_group = QGroupBox("Plot settings")
        middle_group_layout = QVBoxLayout(middle_group)
        middle_group_layout.setSpacing(8)
        middle_layout.addWidget(middle_group)

        # Plot type selection
        # Select if you do a 1D or a 2D plot, the rest of the options should then update accordingly
        # Default: 1D Plot
        self.plot_kind_combo = QComboBox(self)
        self.plot_kind_combo.addItems([
            "1D plot",
            "2D scatter",
            "2D heatmap",
        ])
        middle_group_layout.addWidget(QLabel("Plot type: "))
        middle_group_layout.addWidget(self.plot_kind_combo)

        ## Create all widgets
        # Scan parameter group
        self.param_group = QHBoxLayout(self)

        # First scan parameter selection
        self.main_param_layout = QVBoxLayout(self)
        self.main_param_combo = QComboBox(self)
        self.main_param_combo.addItem("No scan parameter selected")
        self.main_param_combo.setVisible(False)
        self.main_param_layout.addWidget(QLabel("Main scan parameter: "))
        self.main_param_layout.addWidget(self.main_param_combo)
        self.param_group.addLayout(self.main_param_layout)

        # Second scan parameter selection (optional)
        self.other_param_layout = QVBoxLayout(self)
        self.other_param_combo = QComboBox(self)
        self.other_param_combo.addItem("Not used")
        self.other_param_combo.setVisible(False)
        self.other_param_layout.addWidget(QLabel("Other scan parameter: "))
        self.other_param_layout.addWidget(self.other_param_combo)
        self.param_group.addLayout(self.other_param_layout)

        middle_group_layout.addLayout(self.param_group)

        ## Y axis
        self.y_param_layout = QVBoxLayout(self)
        self.y_param_combo = QComboBox(self)
        self.y_param_combo.addItem("Not selected")
        self.y_param_combo.setVisible(False)
        self.y_param_layout.addWidget(QLabel("Y Parameter: "))
        self.y_param_layout.addWidget(self.y_param_combo)
        middle_group_layout.addLayout(self.y_param_layout)

        # Display/filtering mode selection
        self.display_mode_combo = QComboBox(self)
        self.display_mode_combo.addItems([
            "All points",
            "Average over other scan parameter",
        ])
        # If a second scan parameter is selected, also allow to use to filter for specific values
        if self.other_param_combo.currentText() != "Not used":
            self.display_mode_combo.addItems(["Specific value"])
        self.display_mode_combo.setVisible(False)
        # Add widget to layout
        middle_group_layout.addWidget(QLabel("Display mode: "))
        middle_group_layout.addWidget(self.display_mode_combo)

        # If specific value is used, we have to choose which value
        self.filter_value_combo = QComboBox(self)
        self.filter_value_combo.addItem("Select a value")
        self.filter_value_combo.setVisible(False)
        middle_group_layout.addWidget(QLabel("Filter value: "))
        middle_group_layout.addWidget(self.filter_value_combo)

        # Choose fit type (optional)
        self.fit_type_combo = QComboBox(self)
        self.fit_type_combo.addItems([
            "No fit",
            "Exponential decay (with offset)",
            "Exponential decay (lifetime)",
            "TOF temperature",
        ])
        self.fit_type_combo.setVisible(False)
        middle_group_layout.addWidget(QLabel("Fit type: "))
        middle_group_layout.addWidget(self.fit_type_combo)

        ## If 1D plot is selected, show the following options -- probably not necessary since it is anyways handled by the update function
        if self.plot_kind_combo.currentText() == "1D plot":

            self.main_param_combo.setVisible(True) # First scan parameter selection
            self.other_param_combo.setVisible(True) # Second scan parameter selection (optional)
            self.y_param_combo.setVisible(True)
            self.display_mode_combo.setVisible(True) # Display Mode
            if self.display_mode_combo.currentText == "Specific Value":
                self.filter_value_combo.setVisible(True)    # Value choice
            self.fit_type_combo.setVisible(True)

        # And the 2D analysis
        else:
            # ToDo!!!!
            self.main_param_combo.setVisible(True) # First scan parameter selection
            self.other_param_combo.setVisible(True) # Second scan parameter selection (optional)
            self.y_param_combo.setVisible(True)
            self.display_mode_combo.setVisible(True) # Display Mode
            if self.display_mode_combo.currentText == "Specific Value":
                self.filter_value_combo.setVisible(True)    # Value choice
            self.fit_type_combo.setVisible(True)
            
        # Connect everything, that is needed to update the layout, to the refresh function
        self.main_param_combo.currentIndexChanged.connect(self._refresh_analysis_controls)
        self.other_param_combo.currentIndexChanged.connect(self._refresh_analysis_controls)
        self.display_mode_combo.currentIndexChanged.connect(self._refresh_analysis_controls)
        self.y_param_combo.currentIndexChanged.connect(self._refresh_analysis_controls)

        self.generate_plot_button = QPushButton("Generate plot")
        self.generate_plot_button.clicked.connect(self.run_selected_plot)
        middle_group_layout.addWidget(self.generate_plot_button)

        middle_layout.addStretch()

        right_group = QGroupBox("Selected file output")
        right_group_layout = QVBoxLayout(right_group)
        right_group_layout.setSpacing(8)
        right_layout.addWidget(right_group)

        self.terminal_output = QPlainTextEdit(self)
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setPlaceholderText("Selected file output will appear here...")
        self.terminal_output.setMaximumHeight(250)
        right_group_layout.addWidget(self.terminal_output)

        right_layout.addStretch()

        self._refresh_analysis_controls()

    def _refresh_analysis_controls(self):
        ## Many of this steps should probably be set up in the specific listeners of the different drop down menus
        
        # Get all changing parameters of the file and to choose from
        params = detect_scan_parameters(self.selected_files)
        # Set default if nothing is selected
        current_main = self.main_param_combo.currentText() if self.main_param_combo.count() else "No scan parameter selected"
        current_other = self.other_param_combo.currentText() if self.other_param_combo.count() else "Not used"
        current_displayMode = self.display_mode_combo.currentText if self.display_mode_combo.count() else "All points"
        current_yparam = self.y_param_combo.currentText() if self.y_param_combo.count() else "Not selected"

        # Get all choosable main parameters
        self.main_param_combo.blockSignals(True)
        self.main_param_combo.clear()
        self.main_param_combo.addItem("No scan parameter selected")
        self.main_param_combo.addItems(params)
        if current_main in params:
            self.main_param_combo.setCurrentText(current_main)
        else:
            self.main_param_combo.setCurrentIndex(0)
        self.main_param_combo.blockSignals(False)

        # Get all choosable second parameters
        self.other_param_combo.blockSignals(True)
        self.other_param_combo.clear()
        self.other_param_combo.addItem("Not used")
        for name in params:
            if name != self.main_param_combo.currentText():
                self.other_param_combo.addItem(name)
        if current_other in params and current_other != self.main_param_combo.currentText():
            self.other_param_combo.setCurrentText(current_other)
        else:
            self.other_param_combo.setCurrentIndex(0)
        self.other_param_combo.blockSignals(False)

        # Get all selectable y params
        self.y_param_combo.blockSignals(True)
        self.y_param_combo.clear()
        self.y_param_combo.addItem("Not selected")
        self.y_param_combo.addItems(self.yparams)
        if current_yparam in self.yparams:
            self.y_param_combo.setCurrentText(current_yparam)
        else:
            self.y_param_combo.setCurrentIndex(0)
        self.y_param_combo.blockSignals(False)

        # Display mode
        if current_other != "Not used":
            if self.display_mode_combo.findText("Specific value") == -1:
                self.display_mode_combo.addItem("Specific value")
        else:
            index = self.display_mode_combo.findText("Specific value")
            if index != -1:
                self.display_mode_combo.removeItem(index)

        selected_mode = self.display_mode_combo.currentText()
        other_name = self._selected_name(self.other_param_combo)
        is_specific = selected_mode == "Specific value"
        self.filter_value_combo.setVisible(is_specific)

        if not is_specific or other_name is None:
            self.filter_value_combo.clear()
            self.filter_value_combo.addItem("Select a value")
            return

        if is_specific:
            values = []
            for file_path in self.selected_files:
                try:
                    data = pd.read_csv(file_path, encoding="utf-8", skipinitialspace=True)
                    if other_name in data.columns:
                        for value in pd.unique(data[other_name].dropna()):
                            if value not in values:
                                values.append(value)
                except Exception:
                    QMessageBox.warning(self, "No data attainable", "The data for the specific scan parameter choice cannot be attained!")
                    continue

            self.filter_value_combo.blockSignals(True)
            self.filter_value_combo.clear()
            if not values:
                self.filter_value_combo.addItem("No values found")
            else:
                for value in values:
                    self.filter_value_combo.addItem(str(value))
            self.filter_value_combo.blockSignals(False)

    def _selected_name(self, combo):
        value = combo.currentText()
        if value in ("No scan parameter selected", "Not used", "Select a value"):
            return None
        return value

    def ask_for_analysis_settings(self):
        params = detect_scan_parameters(self.selected_files)
        if not params:
            return {"main_name": None, "other_name": None, "mode": "All points", "other_value": None, "fit_type": self.fit_type_combo.currentText()}

        main_name = self._selected_name(self.main_param_combo)
        other_name = self._selected_name(self.other_param_combo)
        y_param = self._selected_name(self.y_param_combo)
        mode = self.display_mode_combo.currentText()
        fit_type = self.fit_type_combo.currentText()
        filter_value = None

        if mode == "Specific value":
            if other_name is None:
                QMessageBox.warning(self, "Invalid choice", "Please choose the other scan parameter first.")
                return None
            filter_value = self.filter_value_combo.currentText()
            if filter_value in ("Select a value", "No values found") or filter_value == "":
                QMessageBox.warning(self, "Invalid choice", "Please select a valid value for the other scan parameter.")
                return None

        if mode == "Average over other scan parameter" and other_name is None:
            other_name = None

        return {
            "main_name": main_name,
            "other_name": other_name if mode != "All points" else None,
            "y_name": y_param,
            "mode": mode,
            "other_value": filter_value,
            "fit_type": fit_type,
        }

    def update_file_list(self):
        self.file_list.clear()
        self.file_selector.clear()
        self.file_selector.addItem("All selected files")
        self.file_selector.addItem("Selected file(s) in list")
        for file_path in self.selected_files:
            self._ensure_metadata(file_path)
            label = self.display_label_for_file(file_path)
            color = self.color_for_file(file_path)
            item = QListWidgetItem(f"{label}   [{color.name()}]")
            item.setForeground(color)
            item.setData(Qt.ItemDataRole.UserRole, file_path)
            self.file_list.addItem(item)
            self.file_selector.addItem(label)
        self.file_selector.setEnabled(len(self.selected_files) > 0)

    def load_data_for_analysis(self, file_paths, selected_scan_names):
        return load_selected_data(file_paths, selected_scan_names)

    def _default_color_for_index(self, index):
        palette = [
            QColor("#1f77b4"), QColor("#ff7f0e"), QColor("#2ca02c"), QColor("#d62728"),
            QColor("#9467bd"), QColor("#8c564b"), QColor("#e377c2"), QColor("#7f7f7f"),
            QColor("#bcbd22"), QColor("#17becf"), QColor("#ff1493"), QColor("#00bfff"),
        ]
        return palette[index % len(palette)]

    def _ensure_metadata(self, file_path):
        if file_path not in self.file_metadata:
            self.file_metadata[file_path] = {
                "label": self._default_label_for_file(file_path),
                "color": self._default_color_for_index(len(self.file_metadata)),
            }

    def _selected_file_from_list(self):
        current_item = self.file_list.currentItem()
        if current_item is not None:
            file_path = current_item.data(Qt.ItemDataRole.UserRole)
            if file_path in self.selected_files:
                return file_path

        selected_items = self.file_list.selectedItems()
        if selected_items:
            file_path = selected_items[0].data(Qt.ItemDataRole.UserRole)
            if file_path in self.selected_files:
                return file_path
        return None

    def on_file_list_clicked(self, item):
        if item is None:
            return
        if item.isSelected():
            item.setSelected(False)
            self.file_list.setCurrentItem(None)
            self._set_current_file_from_selection()
            return

        self.file_list.setCurrentItem(item)
        self._set_current_file_from_selection()
        selected_name = item.text().split("   [")[0]
        for file_path in self.selected_files:
            if self.display_label_for_file(file_path) == selected_name:
                self.file_selector.setCurrentText(selected_name)
                return

    def _set_current_file_from_selection(self):
        file_path = self._selected_file_from_list()
        if file_path is None:
            self.current_file_path = None
            self.terminal_output.clear()
            self.terminal_output.setPlainText("Selected file output will appear here...")
            return

        self.current_file_path = file_path
        self._update_terminal_for_file(file_path)

    def _update_terminal_for_file(self, file_path):
        if file_path is None:
            return

        try:
            data = pd.read_csv(file_path, encoding="utf-8", skipinitialspace=True)
            if data.empty:
                self.terminal_output.setPlainText(f"{os.path.basename(file_path)}\nNo data in file.")
                return

            label = self.display_label_for_file(file_path)
            scan_params = detect_scan_parameters([file_path])
            main_name = scan_params[0] if scan_params else "N/A"
            atomnumber, sigma_x, sigma_y, x0, y0 = get_ExpObserv(data[VAL_NAME], sample_factor=SAMPLE)
            if len(atomnumber) > 0:
                atom_mean = float(np.nanmean(atomnumber))
                sigma_x_mean = float(np.nanmean(sigma_x))
                sigma_y_mean = float(np.nanmean(sigma_y))
                x0_mean = float(np.nanmean(x0))
                y0_mean = float(np.nanmean(y0))
            else:
                atom_mean = np.nan
                sigma_x_mean = np.nan
                sigma_y_mean = np.nan
                x0_mean = np.nan
                y0_mean = np.nan

            if main_name != "N/A" and main_name in data.columns and data[main_name].nunique() > 1:
                x_vals = data[main_name].to_numpy(dtype=float)
                y_vals = atomnumber
                finite = np.isfinite(x_vals) & np.isfinite(y_vals)
                x_vals = x_vals[finite]
                y_vals = y_vals[finite]
                if len(x_vals) >= 3:
                    popt = fit_exponential_decay(x_vals, y_vals)
                    fit_summary = f"Exponential fit: A={popt[0]:.3g}, tau={popt[2]:.3g}, offset={popt[1]:.3g}"
                else:
                    fit_summary = "Exponential fit: insufficient points"
            else:
                fit_summary = "Exponential fit: not available"

            if main_name != "N/A" and main_name in data.columns and data[main_name].nunique() > 1:
                x_vals = data[main_name].to_numpy(dtype=float)
                sigma_vals = sigma_x
                finite = np.isfinite(x_vals) & np.isfinite(sigma_vals)
                x_vals = x_vals[finite]
                sigma_vals = sigma_vals[finite]
                if len(x_vals) >= 3:
                    tof_popt = fit_tof_temperature(x_vals, sigma_vals)
                    tof_summary = f"TOF T_x = {tof_popt[0] * 1e6:.3g} uK, sigma0_x = {tof_popt[1] * 1e6:.3g} um"
                else:
                    tof_summary = "TOF fit: insufficient points"
            else:
                tof_summary = "TOF fit: not available"

            text = (
                f"File: {label}\n"
                f"Path: {file_path}\n"
                f"Detected scan parameter: {main_name}\n"
                f"Mean atom number: {atom_mean:.6g}\n"
                f"Mean sigma_x: {sigma_x_mean:.6g}\n"
                f"Mean sigma_y: {sigma_y_mean:.6g}\n"
                f"Mean x0: {x0_mean:.6g}\n"
                f"Mean y0: {y0_mean:.6g}\n"
                f"{fit_summary}\n"
                f"{tof_summary}"
            )
            self.terminal_output.setPlainText(text)
        except Exception as exc:
            self.terminal_output.setPlainText(f"{os.path.basename(file_path)}\nError while reading file summary:\n{exc}")

    def set_selected_file_name(self):
        file_path = self._selected_file_from_list()
        if file_path is None:
            QMessageBox.warning(self, "No file selected", "Click a loaded CSV file first, then set its label.")
            return
        self._ensure_metadata(file_path)
        new_label, ok = QInputDialog.getText(self, "File label", "Enter a legend label for this file:", text=self.file_metadata[file_path]["label"])
        if ok and new_label.strip():
            self.file_metadata[file_path]["label"] = new_label.strip()
            self.update_file_list()
            self._set_current_file_from_selection()

    def set_selected_file_color(self):
        file_path = self._selected_file_from_list()
        if file_path is None:
            QMessageBox.warning(self, "No file selected", "Click a loaded CSV file first, then set its color.")
            return
        self._ensure_metadata(file_path)
        current = self.file_metadata[file_path]["color"]
        if current is None:
            current = QColor("C0")
        color = QColorDialog.getColor(current, self, "Select file color")
        if color.isValid():
            self.file_metadata[file_path]["color"] = color
            self._set_current_file_from_selection()

    def _default_label_for_file(self, file_path):
        return os.path.splitext(os.path.basename(file_path))[0]

    def _ensure_metadata(self, file_path):
        if file_path not in self.file_metadata:
            self.file_metadata[file_path] = {
                "label": self._default_label_for_file(file_path),
                "color": self._default_color_for_index(len(self.file_metadata)),
            }

    def display_label_for_file(self, file_path):
        self._ensure_metadata(file_path)
        return self.file_metadata[file_path]["label"]

    def color_for_file(self, file_path):
        self._ensure_metadata(file_path)
        color = self.file_metadata[file_path]["color"]
        return QColor(color) if color is not None else None

    def select_csv_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select experiment CSV files",
            DEFAULT_CSV_DIR,
            "CSV files (*.csv)",
        )
        if not files:
            return

        existing = set(self.selected_files)
        for file_path in sorted(files):
            if file_path not in existing:
                self.selected_files.append(file_path)
                self._ensure_metadata(file_path)
                existing.add(file_path)

        self.selected_files = sorted(self.selected_files)
        self.update_file_list()
        self.status_label.setText(f"Selected {len(self.selected_files)} CSV files")
        self._refresh_analysis_controls()

    def remove_selected_csv_file(self):
        selected_name = self.file_selector.currentText()
        if not self.selected_files:
            return

        if selected_name in ("All selected files", "Selected file(s) in list"):
            selected_items = self.file_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "No file selected", "Click a loaded CSV file first, then remove it.")
                return
            selected_labels = {item.text().split("   [")[0] for item in selected_items}
            files_to_remove = [
                path for path in self.selected_files
                if self.display_label_for_file(path) in selected_labels
            ]
        else:
            files_to_remove = [
                path for path in self.selected_files
                if self.display_label_for_file(path) == selected_name
            ]

        for file_path in files_to_remove:
            if file_path in self.selected_files:
                self.selected_files.remove(file_path)
                self.file_metadata.pop(file_path, None)

        if hasattr(self, "_custom_file_selection"):
            self._custom_file_selection = [
                path for path in self.selected_files if path in getattr(self, "_custom_file_selection", [])
            ]

        self.update_file_list()
        self.status_label.setText(f"Selected {len(self.selected_files)} CSV files")
        if not self.selected_files:
            self.status_label.setText("No CSV files selected.")
        self._refresh_analysis_controls()

    def use_selected_files_for_plot(self):
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            self.file_selector.setCurrentText("All selected files")
            self._custom_file_selection = list(self.selected_files)
            return

        selected_labels = {item.text().split("   [")[0] for item in selected_items}
        self.file_selector.setCurrentText("Selected file(s) in list")
        self._custom_file_selection = [
            file_path for file_path in self.selected_files
            if self.display_label_for_file(file_path) in selected_labels
        ]

        if not self._custom_file_selection:
            self._custom_file_selection = list(self.selected_files)

    def selected_target_files(self):
        if not self.selected_files:
            return []

        selector_value = self.file_selector.currentText()
        if selector_value == "All selected files":
            return list(self.selected_files)
        if selector_value == "Selected file(s) in list":
            if hasattr(self, "_custom_file_selection") and self._custom_file_selection:
                return list(self._custom_file_selection)
            selected_labels = {item.text().split("   [")[0] for item in self.file_list.selectedItems()}
            return [
                path for path in self.selected_files
                if self.display_label_for_file(path) in selected_labels
            ]
        for file_path in self.selected_files:
            if self.display_label_for_file(file_path) == selector_value:
                return [file_path]
        return list(self.selected_files)

    def run_selected_plot(self):
        if not self.selected_files:
            QMessageBox.warning(self, "No files selected", "Please select at least one CSV file first.")
            return

        config = self.ask_for_analysis_settings()
        if config is None:
            return

        plot_kind = self.plot_kind_combo.currentText()
        if plot_kind == "1D plot":
            self.plot_1d_analysis()
            return
        if plot_kind == "1D exponential decay":
            self.plot_exponential_decay_analysis()
            return
        if plot_kind == "TOF temperature":
            self.plot_tof_temperature_analysis()
            return
        if plot_kind == "2D scatter":
            self.plot_2d_scatter_analysis()
            return
        if plot_kind == "2D heatmap":
            self.plot_2d_heatmap_analysis()
            return

    def plot_1d_analysis(self):
        if not self.selected_files:
            QMessageBox.warning(self, "No files selected", "Please select at least one CSV file.")
            return

        config = self.ask_for_analysis_settings()
        if config is None:
            return

        main_name = config["main_name"]
        selected_scan_names = [main_name] if main_name else []
        target_files = self.selected_target_files()
        try:
            fig = plt.figure()
            for file_path in target_files:
                filedata = self.load_data_for_analysis([file_path], selected_scan_names)
                x_vals, y_vals, y_err, x_label, y_label = _prepare_filtered_data(
                    filedata,
                    main_name,
                    config["other_name"],
                    config["mode"],
                    config["other_value"],
                    value_column=config["y_name"],
                )
                color = self.color_for_file(file_path)
                color_value = color.name() if color is not None else None
                if config["mode"] == "Average over other scan parameter" and np.any(y_err > 0):
                    plt.errorbar(
                        x_vals,
                        y_vals,
                        yerr=y_err,
                        fmt="o",
                        capsize=4,
                        color=color_value,
                        label=self.display_label_for_file(file_path),
                    )
                else:
                    plt.scatter(
                        x_vals,
                        y_vals,
                        s=18,
                        color=color_value,
                        label=self.display_label_for_file(file_path),
                    )

                # Apply fit if requested
                fit_type = config.get("fit_type", "No fit")
                if fit_type == "Exponential decay (with offset)" and len(x_vals) > 2:
                    try:
                        popt = fit_exponential_decay(x_vals, y_vals)
                        x_smooth = np.linspace(x_vals.min(), x_vals.max(), 100)
                        y_fit = popt[0] * np.exp(-x_smooth / popt[2]) + popt[1]
                        fit_label = f"Decay fit: τ={popt[2]:.3f}"
                        plt.plot(x_smooth, y_fit, "--", color=color_value, linewidth=2, label=fit_label, alpha=0.8)
                    except Exception as e:
                        print(f"Exponential decay fit error: {e}")
                elif fit_type == "Exponential decay (lifetime)" and len(x_vals) > 2:
                    try:
                        popt = fit_exponential_decay_no_offset(x_vals, y_vals)
                        x_smooth = np.linspace(x_vals.min(), x_vals.max(), 100)
                        y_fit = popt[0] * np.exp(-x_smooth / popt[1])
                        fit_label = f"Lifetime: τ={popt[1]:.3f}"
                        plt.plot(x_smooth, y_fit, "--", color=color_value, linewidth=2, label=fit_label, alpha=0.8)
                    except Exception as e:
                        print(f"Exponential decay (no offset) fit error: {e}")
                elif fit_type == "TOF temperature" and len(x_vals) > 2:
                    try:
                        popt = fit_tof_temperature(x_vals, y_vals)
                        x_smooth = np.linspace(x_vals.min(), x_vals.max(), 100)
                        print(["TOF fit: ", popt])
                        pCon = pC.const
                        atom = pC.atom["K41"]
                        y_fit = tof_para(x_smooth, popt[0], popt[1], pCon, atom)
                        fit_label = f"TOF fit: T={popt[0]*1e6:.3f} / Sigma_0={popt[1]*1e6:.3f}"
                        plt.plot(x_smooth, y_fit, "--", color=color_value, linewidth=2, label=fit_label, alpha=0.8)
                    except Exception as e:
                        print(f"TOF fit error: {e}")

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(True)
            if target_files:
                plt.legend()
            plt.tight_layout()
            plt.show()
            self.status_label.setText(f"Loaded {len(target_files)} CSV file(s) and created the 1D plot.")
        except Exception as exc:
            QMessageBox.critical(self, "Analysis failed", f"A problem occurred while creating the 1D plot:\n{exc}")

    def plot_exponential_decay_analysis(self):
        if not self.selected_files:
            QMessageBox.warning(self, "No files selected", "Please select at least one CSV file.")
            return

        config = self.ask_for_analysis_settings()
        if config is None:
            return

        main_name = config["main_name"]
        selected_scan_names = [main_name] if main_name else []
        target_files = self.selected_target_files()
        try:
            fig, ax = plt.subplots()
            for file_path in target_files:
                filedata = self.load_data_for_analysis([file_path], selected_scan_names)
                x_vals, y_vals, y_err, x_label, y_label = _prepare_filtered_data(
                    filedata,
                    main_name,
                    config["other_name"],
                    config["mode"],
                    config["other_value"],
                    value_column="atom_number_fit",
                )
                popt = fit_exponential_decay(x_vals, y_vals)
                fit_x = np.linspace(np.min(x_vals), np.max(x_vals), 300)
                fit_y = popt[0] * np.exp(-fit_x / popt[2]) + popt[1]
                color = self.color_for_file(file_path)
                color_value = color.name() if color is not None else None
                if config["mode"] == "Average over other scan parameter" and np.any(y_err > 0):
                    ax.errorbar(
                        x_vals,
                        y_vals,
                        yerr=y_err,
                        fmt="o",
                        capsize=4,
                        color=color_value,
                        label=self.display_label_for_file(file_path),
                    )
                else:
                    ax.scatter(x_vals, y_vals, s=18, c=color_value, label=self.display_label_for_file(file_path))
                ax.plot(fit_x, fit_y, "--", color=color_value, linewidth=2, label=f"Decay fit (τ={popt[2]:.3f})")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True)
            if target_files:
                ax.legend()
            plt.tight_layout()
            plt.show()
            self.status_label.setText(f"Loaded {len(target_files)} CSV file(s) and created the exponential decay fit.")
        except Exception as exc:
            QMessageBox.critical(self, "Analysis failed", f"A problem occurred while fitting the exponential decay:\n{exc}")

    def plot_tof_temperature_analysis(self):
        if not self.selected_files:
            QMessageBox.warning(self, "No files selected", "Please select at least one CSV file.")
            return

        config = self.ask_for_analysis_settings()
        if config is None:
            return

        main_name = config["main_name"]
        selected_scan_names = [main_name] if main_name else []
        target_files = self.selected_target_files()
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for file_path in target_files:
                filedata = self.load_data_for_analysis([file_path], selected_scan_names)
                atomnumber, sigma_x, sigma_y, x0, y0 = get_ExpObserv(filedata["values"], sample_factor=SAMPLE)
                x_vals, y_vals, y_err, x_label, y_label = _prepare_filtered_data(
                    filedata,
                    main_name,
                    config["other_name"],
                    config["mode"],
                    config["other_value"],
                    value_column="atom_number_fit",
                )

                if main_name is None:
                    x_vals = np.arange(len(atomnumber), dtype=float)
                    x_label = "Sample index"

                sigma_x_plot = sigma_x[: len(x_vals)] if len(sigma_x) >= len(x_vals) else sigma_x
                sigma_y_plot = sigma_y[: len(x_vals)] if len(sigma_y) >= len(x_vals) else sigma_y
                if len(x_vals) != len(sigma_x_plot):
                    x_vals = np.asarray(x_vals, dtype=float)
                    sigma_x_plot = np.asarray(sigma_x_plot, dtype=float)
                    sigma_y_plot = np.asarray(sigma_y_plot, dtype=float)
                    mask = np.isfinite(x_vals) & np.isfinite(sigma_x_plot) & np.isfinite(sigma_y_plot)
                    x_vals = x_vals[mask]
                    sigma_x_plot = sigma_x_plot[mask]
                    sigma_y_plot = sigma_y_plot[mask]

                sigma_x_fit = fit_tof_temperature(x_vals, sigma_x_plot)
                sigma_y_fit = fit_tof_temperature(x_vals, sigma_y_plot)
                fit_x = np.linspace(np.min(x_vals), np.max(x_vals), 300)
                fit_sigma_x = np.sqrt((sigma_x_fit[1] ** 2) + (2 * pC.const["k_B"] * sigma_x_fit[0] / pC.atom["K41"]["m"]) * ((fit_x / 1000) ** 2))
                fit_sigma_y = np.sqrt((sigma_y_fit[1] ** 2) + (2 * pC.const["k_B"] * sigma_y_fit[0] / pC.atom["K41"]["m"]) * ((fit_x / 1000) ** 2))
                color = self.color_for_file(file_path)
                color_value = color.name() if color is not None else None

                axes[0].scatter(x_vals, sigma_x_plot, s=18, color=color_value, label=self.display_label_for_file(file_path))
                axes[0].plot(fit_x, fit_sigma_x, "--", color=color_value, linewidth=2, label=f"TOF fit σ_x (T={sigma_x_fit[0]:.3f})")
                axes[1].scatter(x_vals, sigma_y_plot, s=18, color=color_value, label=self.display_label_for_file(file_path))
                axes[1].plot(fit_x, fit_sigma_y, "--", color=color_value, linewidth=2, label=f"TOF fit σ_y (T={sigma_y_fit[0]:.3f})")

            axes[0].set_xlabel(x_label)
            axes[0].set_ylabel("sigma_x")
            axes[0].grid(True)
            axes[0].legend()
            axes[1].set_xlabel(x_label)
            axes[1].set_ylabel("sigma_y")
            axes[1].grid(True)
            axes[1].legend()
            plt.tight_layout()
            plt.show()
            self.status_label.setText(f"Loaded {len(target_files)} CSV file(s) and created the TOF fit.")
        except Exception as exc:
            QMessageBox.critical(self, "Analysis failed", f"A problem occurred while fitting the TOF temperature:\n{exc}")

    def plot_2d_scatter_analysis(self):
        if not self.selected_files:
            QMessageBox.warning(self, "No files selected", "Please select at least one CSV file.")
            return

        config = self.ask_for_analysis_settings()
        if config is None:
            return

        main_name = config["main_name"]
        other_name = config["other_name"]
        if main_name is None or other_name is None:
            QMessageBox.warning(self, "Invalid choice", "Please select two different scan parameters for the 2D analysis.")
            return

        try:
            filedata = self.load_data_for_analysis(self.selected_target_files(), [main_name, other_name])
            plot_loaded_data(filedata, [main_name, other_name], plot_kind="2d_scatter")
            self.status_label.setText(f"Loaded {len(self.selected_target_files())} CSV file(s) and created the 2D scatter plot.")
        except Exception as exc:
            QMessageBox.critical(self, "Analysis failed", f"A problem occurred while creating the 2D scatter plot:\n{exc}")

    def plot_2d_heatmap_analysis(self):
        if not self.selected_files:
            QMessageBox.warning(self, "No files selected", "Please select at least one CSV file.")
            return

        config = self.ask_for_analysis_settings()
        if config is None:
            return

        main_name = config["main_name"]
        other_name = config["other_name"]
        if main_name is None or other_name is None:
            QMessageBox.warning(self, "Invalid choice", "Please select two different scan parameters for the 2D heatmap.")
            return

        try:
            filedata = self.load_data_for_analysis(self.selected_target_files(), [main_name, other_name])
            plot_loaded_data(filedata, [main_name, other_name], plot_kind="2d_heatmap")
            self.status_label.setText(f"Loaded {len(self.selected_target_files())} CSV file(s) and created the 2D heatmap.")
        except Exception as exc:
            QMessageBox.critical(self, "Analysis failed", f"A problem occurred while creating the 2D heatmap:\n{exc}")


