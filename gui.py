# -*- coding: utf-8 -*-
"""
Created on Tue May 20 16:58:38 2025

@author: klaus

Heavily use of integrated VS Code AI to create this analysis GUI for 2D experimental data.
"""
################################################
#### Importing libraries
################################################

import ast
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QColorDialog,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from helper.config import DEFAULT_CSV_DIR, SAMPLE, VAL_NAME
from helper.dataprocessing.analysis import (
    detect_scan_parameters,
    load_selected_data,
    prepare_filtered_data as _prepare_filtered_data,
    split_filtered_data_by_other,
)
from helper.fitting.registry import FIT_NAMES, get_fit_definition
from helper.formatting import format_with_uncertainty, mean_with_sem
from helper import physicalConstants as pC
from helper.physDataproc import get_ExpObserv
from helper.plotting.analysis import plot_loaded_data
from helper.dataprocessing.expressions import EXPRESSION_TEMPLATES, evaluate_expression

plt.close("all")


class CollapsibleSection(QWidget):
    """A titled panel that collapses to a single header row to save vertical space."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.toggle_button = QToolButton(self)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.toggle_button.clicked.connect(self._on_toggled)

        self.content_area = QWidget(self)
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(4, 0, 4, 4)
        self.content_area.setVisible(False)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        outer_layout.addWidget(self.toggle_button)
        outer_layout.addWidget(self.content_area)

    def _on_toggled(self):
        """Show or hide the content area and flip the header arrow."""
        expanded = self.toggle_button.isChecked()
        self.content_area.setVisible(expanded)
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)


class DataAnalysisGUI(QWidget):
    def __init__(self):
        """Create the file-selection, analysis-control, and output panels."""
        super().__init__()

        # Application state is kept on the widget so every control uses the same selection.
        self.selected_files = []
        self.setWindowTitle("Experiment Data Analysis")
        self.resize(1450, 800)

        self.file_metadata = {}
        self.current_file_path = None
        self._custom_file_selection = []
        self.result_rows = []
        self.workspace_variables = {}

        # The value columns are shared with the data-processing helpers.
        self.yparams = list(VAL_NAME)

        # The window has three columns: file selection, analysis controls, and output.
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

        # File selection panel.
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

        # Plot and fit controls panel.
        middle_group = QGroupBox("Plot settings")
        middle_group_layout = QVBoxLayout(middle_group)
        middle_group_layout.setSpacing(8)
        middle_layout.addWidget(middle_group)

        self.plot_kind_combo = QComboBox(self)
        self.plot_kind_combo.addItems([
            "1D plot",
            "2D scatter",
            "2D heatmap",
        ])
        middle_group_layout.addWidget(QLabel("Plot type: "))
        middle_group_layout.addWidget(self.plot_kind_combo)

        self.param_group = QHBoxLayout(self)

        self.main_param_layout = QVBoxLayout(self)
        self.main_param_combo = QComboBox(self)
        self.main_param_combo.addItem("No scan parameter selected")
        self.main_param_combo.setVisible(False)
        self._make_searchable(self.main_param_combo)
        self.main_param_layout.addWidget(QLabel("Main scan parameter: "))
        self.main_param_layout.addWidget(self.main_param_combo)
        self.param_group.addLayout(self.main_param_layout)

        self.other_param_layout = QVBoxLayout(self)
        self.other_param_combo = QComboBox(self)
        self.other_param_combo.addItem("Not used")
        self.other_param_combo.setVisible(False)
        self._make_searchable(self.other_param_combo)
        self.other_param_layout.addWidget(QLabel("Other scan parameter: "))
        self.other_param_layout.addWidget(self.other_param_combo)
        self.param_group.addLayout(self.other_param_layout)

        middle_group_layout.addLayout(self.param_group)

        self.y_param_layout = QVBoxLayout(self)
        self.y_param_combo = QComboBox(self)
        self.y_param_combo.addItem("Not selected")
        self.y_param_combo.setVisible(False)
        self._make_searchable(self.y_param_combo)
        self.y_param_layout.addWidget(QLabel("Y Parameter (type to search): "))
        self.y_param_layout.addWidget(self.y_param_combo)
        middle_group_layout.addLayout(self.y_param_layout)

        self.display_mode_combo = QComboBox(self)
        self.display_mode_combo.addItems([
            "All points",
            "Average over other scan parameter",
        ])
        self.display_mode_combo.setVisible(False)
        middle_group_layout.addWidget(QLabel("Display mode: "))
        middle_group_layout.addWidget(self.display_mode_combo)

        self.filter_value_combo = QComboBox(self)
        self.filter_value_combo.addItem("Select a value")
        self.filter_value_combo.setVisible(False)
        middle_group_layout.addWidget(QLabel("Filter value: "))
        middle_group_layout.addWidget(self.filter_value_combo)

        self.fit_type_combo = QComboBox(self)
        self.fit_type_combo.addItems(FIT_NAMES)
        self.fit_type_combo.setVisible(False)
        middle_group_layout.addWidget(QLabel("Fit type: "))
        middle_group_layout.addWidget(self.fit_type_combo)

        for widget in (
            self.main_param_combo,
            self.other_param_combo,
            self.y_param_combo,
            self.display_mode_combo,
            self.fit_type_combo,
        ):
            widget.setVisible(True)

        self.main_param_combo.currentIndexChanged.connect(self._refresh_analysis_controls)
        self.other_param_combo.currentIndexChanged.connect(self._refresh_analysis_controls)
        self.display_mode_combo.currentIndexChanged.connect(self._refresh_analysis_controls)
        self.y_param_combo.currentIndexChanged.connect(self._refresh_analysis_controls)

        self.generate_plot_button = QPushButton("Generate plot")
        self.generate_plot_button.clicked.connect(self.run_selected_plot)
        middle_group_layout.addWidget(self.generate_plot_button)

        middle_layout.addStretch()

        output_section = CollapsibleSection("Selected file output")
        right_layout.addWidget(output_section)

        self.terminal_output = QPlainTextEdit(self)
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setPlaceholderText("Selected file output will appear here...")
        self.terminal_output.setMaximumHeight(250)
        output_section.content_layout.addWidget(self.terminal_output)

        # Results workspace for fit parameters, manual values, and constants.
        self._build_results_workspace(right_layout)

        # Keep the output panel compact while allowing the analysis controls to expand.
        right_layout.addStretch()

        self._refresh_analysis_controls()

    def _build_results_workspace(self, parent_layout):
        """Create collapsible sections for collected results and physical constants."""
        results_section = CollapsibleSection("Collected results")
        results_layout = results_section.content_layout

        self.results_search_field = QLineEdit()
        self.results_search_field.setPlaceholderText("Search quantity / source / origin...")
        self.results_search_field.textChanged.connect(self._apply_results_filter)
        results_layout.addWidget(self.results_search_field)

        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels(["Keep", "Quantity", "Value", "Source", "Unit", "Origin"])
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.results_table)

        results_buttons = QHBoxLayout()
        add_button = QPushButton("Add data")
        add_button.clicked.connect(self.add_manual_data)
        results_buttons.addWidget(add_button)
        transfer_button = QPushButton("Transfer kept to workspace")
        transfer_button.clicked.connect(self.transfer_kept_to_workspace)
        results_buttons.addWidget(transfer_button)
        save_button = QPushButton("Save kept data")
        save_button.clicked.connect(self.save_collected_data)
        results_buttons.addWidget(save_button)
        results_layout.addLayout(results_buttons)
        parent_layout.addWidget(results_section)

        self._build_workspace_panel(parent_layout)
        self._build_workspace_plot_panel(parent_layout)

        constants_section = CollapsibleSection("Physical constants")
        constants_layout = constants_section.content_layout

        species_row = QHBoxLayout()
        species_row.addWidget(QLabel("Species:"))
        self.species_combo = QComboBox()
        self.species_combo.addItem("All")
        self.species_combo.addItems(sorted(pC.atom.keys()))
        self.species_combo.currentTextChanged.connect(self._on_species_changed)
        species_row.addWidget(self.species_combo)
        constants_layout.addLayout(species_row)

        self.constants_table = QTableWidget(0, 2)
        self.constants_table.setHorizontalHeaderLabels(["Name", "Value"])
        self.constants_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.constants_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.constants_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.constants_table.horizontalHeader().setStretchLastSection(True)
        self._populate_constants_table()
        constants_layout.addWidget(self.constants_table)

        transfer_constants_button = QPushButton("Add selected to workspace")
        transfer_constants_button.clicked.connect(self.transfer_constants_to_workspace)
        constants_layout.addWidget(transfer_constants_button)
        parent_layout.addWidget(constants_section)

    def _build_workspace_panel(self, parent_layout):
        """Create the panel holding transferred and custom workspace variables."""
        workspace_section = CollapsibleSection("Workspace variables")
        workspace_layout = workspace_section.content_layout

        self.workspace_table = QTableWidget(0, 4)
        self.workspace_table.setHorizontalHeaderLabels(["Name", "Shape", "Value", "Source"])
        self.workspace_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.workspace_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.workspace_table.horizontalHeader().setStretchLastSection(True)
        workspace_layout.addWidget(self.workspace_table)

        workspace_buttons = QHBoxLayout()
        add_variable_button = QPushButton("Add custom variable")
        add_variable_button.clicked.connect(self.add_custom_variable)
        workspace_buttons.addWidget(add_variable_button)
        remove_variable_button = QPushButton("Remove selected variable")
        remove_variable_button.clicked.connect(self.remove_selected_variable)
        workspace_buttons.addWidget(remove_variable_button)
        function_button = QPushButton("Create from function")
        function_button.clicked.connect(self.create_variable_from_function)
        workspace_buttons.addWidget(function_button)
        workspace_layout.addLayout(workspace_buttons)

        persistence_buttons = QHBoxLayout()
        save_workspace_button = QPushButton("Save workspace")
        save_workspace_button.clicked.connect(self.save_workspace)
        persistence_buttons.addWidget(save_workspace_button)
        load_workspace_button = QPushButton("Load workspace")
        load_workspace_button.clicked.connect(self.load_workspace)
        persistence_buttons.addWidget(load_workspace_button)
        workspace_layout.addLayout(persistence_buttons)

        parent_layout.addWidget(workspace_section)

    def _build_workspace_plot_panel(self, parent_layout):
        """Create controls for plotting workspace arrays/variables against each other."""
        plot_section = CollapsibleSection("Plot from workspace")
        plot_layout = plot_section.content_layout

        self.workspace_x_combo = QComboBox()
        self._make_searchable(self.workspace_x_combo)
        plot_layout.addWidget(QLabel("X variable:"))
        plot_layout.addWidget(self.workspace_x_combo)

        self.workspace_y_combo = QComboBox()
        self._make_searchable(self.workspace_y_combo)
        plot_layout.addWidget(QLabel("Y variable:"))
        plot_layout.addWidget(self.workspace_y_combo)

        self.workspace_fit_combo = QComboBox()
        self.workspace_fit_combo.addItems(FIT_NAMES)
        plot_layout.addWidget(QLabel("Fit type:"))
        plot_layout.addWidget(self.workspace_fit_combo)

        plot_button = QPushButton("Plot workspace variables")
        plot_button.clicked.connect(self.plot_workspace_variables)
        plot_layout.addWidget(plot_button)
        parent_layout.addWidget(plot_section)

    def _refresh_workspace_plot_controls(self):
        """Keep the workspace X/Y pickers in sync with the current variable names."""
        names = sorted(self.workspace_variables)
        for combo in (self.workspace_x_combo, self.workspace_y_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            if current in names:
                combo.setCurrentText(current)
            combo.blockSignals(False)

    def plot_workspace_variables(self):
        """Plot one workspace variable against another, with an optional fit."""
        x_name = self.workspace_x_combo.currentText()
        y_name = self.workspace_y_combo.currentText()
        if x_name not in self.workspace_variables or y_name not in self.workspace_variables:
            QMessageBox.warning(self, "Invalid selection", "Choose an X and a Y variable from the workspace.")
            return

        x_vals = np.ravel(self.workspace_variables[x_name]["value"])
        y_vals = np.ravel(self.workspace_variables[y_name]["value"])
        if x_vals.size != y_vals.size:
            QMessageBox.warning(self, "Shape mismatch", f"'{x_name}' has {x_vals.size} values but '{y_name}' has {y_vals.size}.")
            return

        try:
            plt.figure()
            self._plot_series_with_fit(
                x_vals, y_vals, np.zeros_like(y_vals), None, f"{y_name} vs {x_name}",
                self.workspace_fit_combo.currentText(), use_errorbar=False,
            )
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as exc:
            QMessageBox.critical(self, "Plot failed", f"A problem occurred while plotting workspace variables:\n{exc}")

    def _parse_array_literal(self, text):
        """Parse scalars, comma lists, or nested lists into a NumPy array of any shape."""
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            # ast.literal_eval rejects bare nan/inf tokens, so evaluate those as a fallback.
            parsed = eval(text, {"__builtins__": {}, "nan": float("nan"), "inf": float("inf")})
        return np.array(parsed, dtype=float)

    def _coerce_to_array(self, value):
        """Convert a stored result value (already numeric, or a typed literal) into an array."""
        if isinstance(value, (list, tuple, np.ndarray, int, float, np.floating, np.integer)):
            return np.asarray(value, dtype=float)
        return self._parse_array_literal(str(value))

    def _set_workspace_variable(self, name, array, source):
        """Store or overwrite a workspace variable under the given name."""
        self.workspace_variables[name] = {"value": array, "source": source}
        self._refresh_workspace_table()

    def _refresh_workspace_table(self):
        """Render workspace variables with their shape and a truncated value preview."""
        names = sorted(self.workspace_variables)
        self.workspace_table.setRowCount(len(names))
        for row_index, name in enumerate(names):
            variable = self.workspace_variables[name]
            array = variable["value"]
            preview = np.array2string(array, threshold=8, max_line_width=60)
            self.workspace_table.setItem(row_index, 0, QTableWidgetItem(name))
            self.workspace_table.setItem(row_index, 1, QTableWidgetItem(str(array.shape)))
            self.workspace_table.setItem(row_index, 2, QTableWidgetItem(preview))
            self.workspace_table.setItem(row_index, 3, QTableWidgetItem(variable["source"]))
        self.workspace_table.resizeColumnsToContents()
        self._refresh_workspace_plot_controls()

    def transfer_kept_to_workspace(self):
        """Copy each kept result into the workspace as a named scalar variable."""
        results = self._kept_results()
        if not results:
            QMessageBox.warning(self, "No data selected", "Select at least one kept result before transferring.")
            return

        for row in results:
            name = f"{row['source']}_{row['quantity']}".replace(" ", "_")
            try:
                array = self._coerce_to_array(row["value"])
            except (ValueError, SyntaxError, TypeError, NameError):
                continue
            self._set_workspace_variable(name, array, source=row["source"])

    def add_custom_variable(self):
        """Open a dialog for adding a scalar/array variable, or combining existing ones."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add custom variable")
        form = QFormLayout(dialog)
        name_field = QLineEdit()
        value_field = QLineEdit()
        form.addRow("Name:", name_field)
        form.addRow("Value (e.g. 3.4 or 1,2,3 or [[1,2],[3,4]]):", value_field)

        existing_list = QListWidget()
        existing_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        existing_list.addItems(sorted(self.workspace_variables))
        form.addRow("Or combine existing variables (ordered selection):", existing_list)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name = name_field.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid name", "Please enter a variable name.")
            return

        selected_names = [item.text() for item in existing_list.selectedItems()]
        if selected_names:
            array = np.concatenate([np.atleast_1d(self.workspace_variables[n]["value"]) for n in selected_names])
            source = "combined: " + ", ".join(selected_names)
        else:
            try:
                array = self._parse_array_literal(value_field.text().strip())
            except (ValueError, SyntaxError) as exc:
                QMessageBox.warning(self, "Invalid value", f"Could not parse the value:\n{exc}")
                return
            source = "manual"
        self._set_workspace_variable(name, array, source=source)

    def remove_selected_variable(self):
        """Remove the workspace variable selected in the workspace table."""
        selected_items = self.workspace_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No variable selected", "Select a variable in the workspace table first.")
            return
        name = self.workspace_table.item(selected_items[0].row(), 0).text()
        self.workspace_variables.pop(name, None)
        self._refresh_workspace_table()

    def save_workspace(self):
        """Save every workspace variable (values and source) to a JSON file."""
        if not self.workspace_variables:
            QMessageBox.warning(self, "Nothing to save", "The workspace has no variables to save yet.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save workspace variables", "workspace.json", "JSON files (*.json)")
        if not file_path:
            return

        payload = {
            name: {"value": variable["value"].tolist(), "source": variable["source"]}
            for name, variable in self.workspace_variables.items()
        }
        try:
            with open(file_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save the workspace:\n{exc}")
            return
        self.status_label.setText(f"Saved {len(payload)} workspace variable(s) to {os.path.basename(file_path)}")

    def load_workspace(self):
        """Load workspace variables from a previously saved JSON file, merging with existing ones."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load workspace variables", "", "JSON files (*.json)")
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Load failed", f"Could not load the workspace:\n{exc}")
            return

        loaded = 0
        for name, entry in payload.items():
            try:
                array = np.asarray(entry["value"], dtype=float)
            except (KeyError, ValueError, TypeError):
                continue
            self._set_workspace_variable(name, array, source=entry.get("source", "loaded"))
            loaded += 1
        self.status_label.setText(f"Loaded {loaded} workspace variable(s) from {os.path.basename(file_path)}")

    def create_variable_from_function(self):
        """Open a dialog that evaluates an expression over workspace variables and constants."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Create variable from function")
        form = QFormLayout(dialog)

        name_field = QLineEdit()
        form.addRow("New variable name:", name_field)

        # Selecting a template fills the expression field; see helper/dataprocessing/expressions.py to add more.
        templates = dict(EXPRESSION_TEMPLATES)
        template_combo = QComboBox()
        template_combo.addItems(templates.keys())
        form.addRow("Template:", template_combo)

        expression_field = QLineEdit()
        template_combo.currentTextChanged.connect(lambda text: expression_field.setText(templates[text]))
        form.addRow("Expression:", expression_field)

        available = ", ".join(sorted(self.workspace_variables)) or "(no workspace variables yet)"
        help_label = QLabel(f"Available: {available}\nAlso usable: const[...], atom[...][...], np, and indexing like f[0]")
        help_label.setWordWrap(True)
        form.addRow(help_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        name = name_field.text().strip()
        expression = expression_field.text().strip()
        if not name or not expression:
            QMessageBox.warning(self, "Invalid input", "Please provide both a variable name and an expression.")
            return

        try:
            array = evaluate_expression(expression, self.workspace_variables)
        except Exception as exc:
            QMessageBox.warning(self, "Evaluation failed", f"Could not evaluate the expression:\n{exc}")
            return

        self._set_workspace_variable(name, array, source=f"function: {expression}")

    def _populate_constants_table(self, species="All"):
        """Display physical constants, optionally filtered to a single atomic species."""
        if species == "All":
            rows = [(f"const.{name}", value) for name, value in pC.const.items()]
            atoms_to_show = pC.atom.items()
        else:
            rows = []
            atoms_to_show = [(species, pC.atom[species])]

        for atom_name, parameters in atoms_to_show:
            for parameter_name, value in parameters.items():
                if isinstance(value, dict):
                    for nested_name, nested_value in value.items():
                        rows.append((f"atom.{atom_name}.{parameter_name}.{nested_name}", nested_value))
                else:
                    rows.append((f"atom.{atom_name}.{parameter_name}", value))

        self.constants_table.setRowCount(len(rows))
        for row_index, (name, value) in enumerate(rows):
            self.constants_table.setItem(row_index, 0, QTableWidgetItem(name))
            self.constants_table.setItem(row_index, 1, QTableWidgetItem(str(value)))

    def _on_species_changed(self, species):
        """Refresh the constants table to show only the chosen atomic species."""
        self._populate_constants_table(species)

    def transfer_constants_to_workspace(self):
        """Copy the selected physical constants into the workspace as named variables."""
        selected_rows = sorted({item.row() for item in self.constants_table.selectedItems()})
        if not selected_rows:
            QMessageBox.warning(self, "No constants selected", "Select one or more rows in the constants table first.")
            return
        for row in selected_rows:
            name = self.constants_table.item(row, 0).text().replace(".", "_")
            value_text = self.constants_table.item(row, 1).text()
            try:
                array = self._parse_array_literal(value_text)
            except (ValueError, SyntaxError):
                continue
            self._set_workspace_variable(name, array, source="physical constant")

    def _format_result_value(self, row):
        """Render a result's value, combined with its standard error when available."""
        value = row.get("value")
        error = row.get("error")
        if error is not None and np.isfinite(error) and isinstance(value, (int, float, np.floating, np.integer)):
            return format_with_uncertainty(value, error)
        return str(value)

    def _refresh_results_table(self):
        """Render collected rows while preserving each row's keep selection."""
        self.results_table.setRowCount(len(self.result_rows))
        for row_index, row in enumerate(self.result_rows):
            keep_item = QTableWidgetItem()
            keep_item.setFlags(keep_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            keep_item.setCheckState(Qt.CheckState.Checked if row.get("keep", False) else Qt.CheckState.Unchecked)
            keep_item.setData(Qt.ItemDataRole.UserRole, row)
            self.results_table.setItem(row_index, 0, keep_item)
            display_values = {
                "quantity": row.get("quantity", ""),
                "value": self._format_result_value(row),
                "source": row.get("source", ""),
                "unit": row.get("unit", ""),
                "origin": row.get("origin", ""),
            }
            for column, key in enumerate(("quantity", "value", "source", "unit", "origin"), start=1):
                self.results_table.setItem(row_index, column, QTableWidgetItem(str(display_values[key])))
        self.results_table.resizeColumnsToContents()
        self._apply_results_filter()

    def _apply_results_filter(self):
        """Hide result rows whose quantity, source, or origin don't match the search text."""
        filter_text = self.results_search_field.text().strip().lower()
        for row_index, row in enumerate(self.result_rows):
            haystack = f"{row.get('quantity', '')} {row.get('source', '')} {row.get('origin', '')}".lower()
            self.results_table.setRowHidden(row_index, bool(filter_text) and filter_text not in haystack)

    def _replace_file_results(self, file_path, rows):
        """Replace automatic rows for one file without disturbing manual rows."""
        existing = {row["quantity"]: row for row in self.result_rows if row.get("file_path") == file_path and row.get("automatic")}
        self.result_rows = [
            row for row in self.result_rows
            if not (row.get("file_path") == file_path and row.get("automatic"))
        ]
        for row in rows:
            row["keep"] = existing.get(row["quantity"], {}).get("keep", False)
            row["automatic"] = True
            row["file_path"] = file_path
            self.result_rows.append(row)
        self._refresh_results_table()

    def _kept_results(self):
        """Return checked result rows, synchronizing checkbox states first."""
        for row_index, row in enumerate(self.result_rows):
            item = self.results_table.item(row_index, 0)
            row["keep"] = item.checkState() == Qt.CheckState.Checked if item else row.get("keep", False)
        return [row for row in self.result_rows if row.get("keep", False)]

    def _record_result(self, source, quantity, value, unit, origin, error=None):
        """Add a newly computed fit or summary value (with optional standard error)."""
        self.result_rows.append({
            "source": source, "quantity": quantity, "value": value, "error": error, "unit": unit,
            "origin": origin, "keep": False, "automatic": False,
        })
        self._refresh_results_table()

    def _record_average_results(self, source, x_label, x_vals, y_label, y_vals, y_err):
        """Add each averaged point's value and its standard error to the collected results."""
        for x, y, err in zip(x_vals, y_vals, y_err):
            quantity = f"{y_label}_avg_at_{x_label}={x:.6g}"
            self.result_rows.append({
                "source": source, "quantity": quantity, "value": float(y), "error": float(err),
                "unit": "", "origin": "average", "keep": False, "automatic": False,
            })
        self._refresh_results_table()

    def _record_file_arrays(self, file_path, label, data):
        """Add each scan parameter's and value column's full array, skipping ones already recorded."""
        columns = [(name, "scan parameter array") for name in detect_scan_parameters([file_path]) if name in data.columns]
        columns += [(name, "value array") for name in VAL_NAME if name in data.columns]

        recorded = {
            row["quantity"] for row in self.result_rows
            if row.get("file_path") == file_path and row.get("origin") in ("scan parameter array", "value array")
        }
        added = False
        for name, origin in columns:
            if name in recorded:
                continue
            self.result_rows.append({
                "source": label, "quantity": name, "value": data[name].tolist(), "unit": "",
                "origin": origin, "keep": False, "automatic": True, "file_path": file_path,
            })
            added = True
        if added:
            self._refresh_results_table()

    def add_manual_data(self):
        """Open a small form for adding a user-defined result row."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add data")
        form = QFormLayout(dialog)
        source = QLineEdit("Manual")
        # Offer every detected scan parameter and fit observable, searchable, but still editable for custom names.
        quantity = QComboBox()
        quantity.setEditable(True)
        quantity.addItems(list(dict.fromkeys(detect_scan_parameters(self.selected_files) + self.yparams)))
        self._make_searchable(quantity)
        value = QLineEdit()
        unit = QLineEdit()
        form.addRow("Source:", source)
        form.addRow("Quantity (type to search):", quantity)
        form.addRow("Value:", value)
        form.addRow("Unit:", unit)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted and quantity.currentText().strip() and value.text().strip():
            self.result_rows.append({
                "source": source.text().strip() or "Manual",
                "quantity": quantity.currentText().strip(),
                "value": value.text().strip(),
                "unit": unit.text().strip(),
                "origin": "manual",
                "keep": True,
                "automatic": False,
            })
            self._refresh_results_table()

    def save_collected_data(self):
        """Save checked results and all displayed physical constants to a CSV file."""
        results = self._kept_results()
        if not results:
            QMessageBox.warning(self, "No data selected", "Select or add at least one result before saving.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save collected data", "analysis_results.csv", "CSV files (*.csv)")
        if not file_path:
            return

        rows = [
            {
                "source": row.get("source", ""),
                "quantity": row.get("quantity", ""),
                "value": self._format_result_value(row),
                "error": row.get("error", ""),
                "unit": row.get("unit", ""),
                "origin": row.get("origin", ""),
            }
            for row in results
        ]
        output = pd.DataFrame(rows, columns=["source", "quantity", "value", "error", "unit", "origin"])
        constants = pd.DataFrame(
            [{"source": "physical constants", "quantity": self.constants_table.item(row, 0).text(), "value": self.constants_table.item(row, 1).text(), "error": "", "unit": "", "origin": "constant"} for row in range(self.constants_table.rowCount())]
        )
        pd.concat([output, constants], ignore_index=True).to_csv(file_path, index=False)
        self.status_label.setText(f"Saved {len(results)} collected result(s) to {os.path.basename(file_path)}")

    def _make_searchable(self, combo):
        """Let the user type to filter this combo box's items instead of scrolling."""
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        completer = QCompleter(combo.model(), combo)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        combo.setCompleter(completer)

    def _refresh_analysis_controls(self):
        """Refresh scan-parameter choices after the selected files change."""
        params = detect_scan_parameters(self.selected_files)
        current_main = self.main_param_combo.currentText() if self.main_param_combo.count() else "No scan parameter selected"
        current_other = self.other_param_combo.currentText() if self.other_param_combo.count() else "Not used"
        current_yparam = self.y_param_combo.currentText() if self.y_param_combo.count() else "Not selected"

        self.main_param_combo.blockSignals(True)
        self.main_param_combo.clear()
        self.main_param_combo.addItem("No scan parameter selected")
        self.main_param_combo.addItems(params)
        if current_main in params:
            self.main_param_combo.setCurrentText(current_main)
        else:
            self.main_param_combo.setCurrentIndex(0)
        self.main_param_combo.blockSignals(False)

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

        self.y_param_combo.blockSignals(True)
        self.y_param_combo.clear()
        self.y_param_combo.addItem("Not selected")
        self.y_param_combo.addItems(self.yparams)
        if current_yparam in self.yparams:
            self.y_param_combo.setCurrentText(current_yparam)
        else:
            self.y_param_combo.setCurrentIndex(0)
        self.y_param_combo.blockSignals(False)

        # "Specific value" and the per-value split only make sense with a second parameter.
        extra_modes = ("Specific value", "Split by other scan parameter")
        if current_other != "Not used":
            for extra_mode in extra_modes:
                if self.display_mode_combo.findText(extra_mode) == -1:
                    self.display_mode_combo.addItem(extra_mode)
        else:
            for extra_mode in extra_modes:
                index = self.display_mode_combo.findText(extra_mode)
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

        # Specific-value filtering is populated from the currently selected files.
        values = []
        for file_path in self.selected_files:
            try:
                data = pd.read_csv(file_path, encoding="utf-8", skipinitialspace=True)
                if other_name in data.columns:
                    values.extend(value for value in pd.unique(data[other_name].dropna()) if value not in values)
            except Exception:
                QMessageBox.warning(self, "No data attainable", "The data for the specific scan parameter choice cannot be attained!")

        self.filter_value_combo.blockSignals(True)
        self.filter_value_combo.clear()
        self.filter_value_combo.addItems([str(value) for value in values] or ["No values found"])
        self.filter_value_combo.blockSignals(False)

    def _selected_name(self, combo):
        """Return a meaningful combo-box value, or None for placeholder entries."""
        value = combo.currentText()
        if value in ("No scan parameter selected", "Not used", "Select a value"):
            return None
        return value

    def ask_for_analysis_settings(self):
        """Read the current controls and validate settings needed for analysis."""
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

        if mode == "Split by other scan parameter" and other_name is None:
            QMessageBox.warning(self, "Invalid choice", "Please choose the other scan parameter first.")
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
        """Rebuild the visible file list and preserve each file's metadata."""
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
        """Delegate selected-file loading to the data-processing helper."""
        return load_selected_data(file_paths, selected_scan_names)

    def _default_color_for_index(self, index):
        """Return a repeatable display color for a file-list position."""
        palette = [
            QColor("#1f77b4"), QColor("#ff7f0e"), QColor("#2ca02c"), QColor("#d62728"),
            QColor("#9467bd"), QColor("#8c564b"), QColor("#e377c2"), QColor("#7f7f7f"),
            QColor("#bcbd22"), QColor("#17becf"), QColor("#ff1493"), QColor("#00bfff"),
        ]
        return palette[index % len(palette)]

    def _ensure_metadata(self, file_path):
        """Create default label and color metadata for a new file path."""
        if file_path not in self.file_metadata:
            self.file_metadata[file_path] = {
                "label": self._default_label_for_file(file_path),
                "color": self._default_color_for_index(len(self.file_metadata)),
            }

    def _selected_file_from_list(self):
        """Return the current valid file-list selection, if one exists."""
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
        """Update the active file and selector when a list item is clicked."""
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
        """Refresh the output summary for the current file-list selection."""
        file_path = self._selected_file_from_list()
        if file_path is None:
            self.current_file_path = None
            self.terminal_output.clear()
            self.terminal_output.setPlainText("Selected file output will appear here...")
            return

        self.current_file_path = file_path
        self._update_terminal_for_file(file_path)

    def _update_terminal_for_file(self, file_path):
        """Show summary statistics and quick fit results for one selected file."""
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
            atom_mean, atom_sem = mean_with_sem(atomnumber)
            sigma_x_mean, sigma_x_sem = mean_with_sem(sigma_x)
            sigma_y_mean, sigma_y_sem = mean_with_sem(sigma_y)
            x0_mean, x0_sem = mean_with_sem(x0)
            y0_mean, y0_sem = mean_with_sem(y0)

            # Quick-look fits reuse the same registry as the plotting panels.
            decay_fit = get_fit_definition("Exponential decay (with offset)")
            tof_fit = get_fit_definition("TOF temperature")
            has_varying_main = main_name != "N/A" and main_name in data.columns and data[main_name].nunique() > 1
            popt = perr = tof_popt = tof_perr = None

            if has_varying_main:
                x_vals = data[main_name].to_numpy(dtype=float)
                finite = np.isfinite(x_vals) & np.isfinite(atomnumber)
                if np.count_nonzero(finite) >= 3:
                    popt, perr = decay_fit.fit(x_vals[finite], atomnumber[finite])
                    fit_summary = (
                        f"Exponential fit: A={format_with_uncertainty(popt[0], perr[0])}, "
                        f"tau={format_with_uncertainty(popt[2], perr[2])}, "
                        f"offset={format_with_uncertainty(popt[1], perr[1])}"
                    )
                else:
                    fit_summary = "Exponential fit: insufficient points"

                finite = np.isfinite(x_vals) & np.isfinite(sigma_x)
                if np.count_nonzero(finite) >= 3:
                    tof_popt, tof_perr = tof_fit.fit(x_vals[finite], sigma_x[finite])
                    tof_summary = (
                        f"TOF T_x = {format_with_uncertainty(tof_popt[0] * 1e6, tof_perr[0] * 1e6)} uK, "
                        f"sigma0_x = {format_with_uncertainty(tof_popt[1] * 1e6, tof_perr[1] * 1e6)} um"
                    )
                else:
                    tof_summary = "TOF fit: insufficient points"
            else:
                fit_summary = "Exponential fit: not available"
                tof_summary = "TOF fit: not available"

            # Store reusable scalar results separately from the human-readable summary.
            result_rows = [
                {"source": label, "quantity": "atom_number_mean", "value": atom_mean, "error": atom_sem, "unit": "", "origin": "summary"},
                {"source": label, "quantity": "sigma_x_mean", "value": sigma_x_mean, "error": sigma_x_sem, "unit": "", "origin": "summary"},
                {"source": label, "quantity": "sigma_y_mean", "value": sigma_y_mean, "error": sigma_y_sem, "unit": "", "origin": "summary"},
                {"source": label, "quantity": "x0_mean", "value": x0_mean, "error": x0_sem, "unit": "", "origin": "summary"},
                {"source": label, "quantity": "y0_mean", "value": y0_mean, "error": y0_sem, "unit": "", "origin": "summary"},
            ]
            # Keep the full data column for every scan parameter and fit observable available.
            for name in scan_params:
                if name in data.columns:
                    result_rows.append({"source": label, "quantity": name, "value": data[name].tolist(), "unit": "", "origin": "scan parameter array"})
            for name in VAL_NAME:
                if name in data.columns:
                    result_rows.append({"source": label, "quantity": name, "value": data[name].tolist(), "unit": "", "origin": "value array"})
            if popt is not None:
                result_rows.extend(decay_fit.record_rows(label, popt, perr))
            if tof_popt is not None:
                result_rows.extend(tof_fit.record_rows(label, tof_popt, tof_perr))
            self._replace_file_results(file_path, result_rows)

            text = (
                f"File: {label}\n"
                f"Path: {file_path}\n"
                f"Detected scan parameter: {main_name}\n"
                f"Mean atom number: {format_with_uncertainty(atom_mean, atom_sem)}\n"
                f"Mean sigma_x: {format_with_uncertainty(sigma_x_mean, sigma_x_sem)}\n"
                f"Mean sigma_y: {format_with_uncertainty(sigma_y_mean, sigma_y_sem)}\n"
                f"Mean x0: {format_with_uncertainty(x0_mean, x0_sem)}\n"
                f"Mean y0: {format_with_uncertainty(y0_mean, y0_sem)}\n"
                f"{fit_summary}\n"
                f"{tof_summary}"
            )
            self.terminal_output.setPlainText(text)
        except Exception as exc:
            self.terminal_output.setPlainText(f"{os.path.basename(file_path)}\nError while reading file summary:\n{exc}")

    def set_selected_file_name(self):
        """Let the user assign a custom legend label to the selected file."""
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
        """Let the user choose a plot color for the selected file."""
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
        """Use the filename without its extension as the default label."""
        return os.path.splitext(os.path.basename(file_path))[0]

    def display_label_for_file(self, file_path):
        """Return the configured display label for a file."""
        self._ensure_metadata(file_path)
        return self.file_metadata[file_path]["label"]

    def color_for_file(self, file_path):
        """Return the configured Qt color for a file."""
        self._ensure_metadata(file_path)
        color = self.file_metadata[file_path]["color"]
        return QColor(color) if color is not None else None

    def select_csv_files(self):
        """Add CSV files to the selection without duplicating existing entries."""
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
        """Remove the files selected by the current file-scope control."""
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
        """Store the currently highlighted files as the plot subset."""
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
        """Return the files selected by the file-scope control."""
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
        """Dispatch the selected plot type to its analysis method."""
        if not self.selected_files:
            QMessageBox.warning(self, "No files selected", "Please select at least one CSV file first.")
            return

        handlers = {
            "1D plot": self.plot_1d_analysis,
            "2D scatter": self.plot_2d_scatter_analysis,
            "2D heatmap": self.plot_2d_heatmap_analysis,
        }
        handler = handlers.get(self.plot_kind_combo.currentText())
        if handler is not None:
            handler()

    def _plot_series_with_fit(self, x_vals, y_vals, y_err, color_value, label, fit_type, use_errorbar):
        """Plot one data series and overlay the requested fit using a shared color."""
        if use_errorbar and np.any(y_err > 0):
            handle = plt.errorbar(x_vals, y_vals, yerr=y_err, fmt="o", capsize=4, color=color_value, label=label)
            resolved_color = color_value if color_value is not None else handle.lines[0].get_color()
        else:
            handle = plt.scatter(x_vals, y_vals, s=18, color=color_value, label=label)
            resolved_color = color_value if color_value is not None else handle.get_facecolor()[0]

        if fit_type == "No fit" or len(x_vals) <= 2:
            return
        fit_definition = get_fit_definition(fit_type)
        if fit_definition is None:
            return
        try:
            popt, perr = fit_definition.fit(x_vals, y_vals)
            x_smooth = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_fit = fit_definition.model(x_smooth, *popt)
            for row in fit_definition.record_rows(label, popt, perr):
                self._record_result(**row)
            plt.plot(x_smooth, y_fit, "--", color=resolved_color, linewidth=2, label=fit_definition.label(label, popt, perr), alpha=0.8)
        except Exception as exc:
            print(f"{fit_type} fit error for {label}: {exc}")

    def plot_1d_analysis(self):
        """Plot the selected value for each file and optionally overlay a fit."""
        if not self.selected_files:
            QMessageBox.warning(self, "No files selected", "Please select at least one CSV file.")
            return

        config = self.ask_for_analysis_settings()
        if config is None:
            return

        main_name = config["main_name"]
        other_name = config["other_name"]
        fit_type = config.get("fit_type", "No fit")
        is_split_mode = config["mode"] == "Split by other scan parameter"
        selected_scan_names = [main_name] if main_name else []
        target_files = self.selected_target_files()
        x_label, y_label = "Sample index", "Sample value"
        try:
            plt.figure()
            for file_path in target_files:
                filedata = self.load_data_for_analysis([file_path], selected_scan_names)
                file_label = self.display_label_for_file(file_path)
                self._record_file_arrays(file_path, file_label, filedata["data"])

                if is_split_mode:
                    # Each value of the other scan parameter becomes its own series and fit.
                    series, x_label, y_label = split_filtered_data_by_other(
                        filedata, main_name, other_name, value_column=config["y_name"]
                    )
                    for other_value, x_vals, y_vals in series:
                        if len(x_vals) == 0:
                            continue
                        series_label = f"{file_label} ({other_name}={other_value})"
                        self._plot_series_with_fit(
                            x_vals, y_vals, np.zeros_like(y_vals), None, series_label, fit_type, use_errorbar=False
                        )
                    continue

                x_vals, y_vals, y_err, x_label, y_label = _prepare_filtered_data(
                    filedata,
                    main_name,
                    other_name,
                    config["mode"],
                    config["other_value"],
                    value_column=config["y_name"],
                )
                color = self.color_for_file(file_path)
                color_value = color.name() if color is not None else None
                use_errorbar = config["mode"] == "Average over other scan parameter"
                self._plot_series_with_fit(x_vals, y_vals, y_err, color_value, file_label, fit_type, use_errorbar)
                if use_errorbar and np.any(y_err > 0):
                    self._record_average_results(file_label, x_label, x_vals, config["y_name"] or y_label, y_vals, y_err)

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

    def plot_2d_scatter_analysis(self):
        """Create a 2D scatter plot using the two selected scan parameters."""
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
        """Create a 2D heatmap using the two selected scan parameters."""
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


