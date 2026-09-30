"""Low-level Matplotlib functions for two-dimensional scan data."""

import matplotlib.pyplot as plt
import numpy as np


def plot_2Ddata_scatter(
    scanVar1,
    scanVar2,
    y,
    paramName1="x",
    paramName2="y",
    title=None,
    xlabel=None,
    ylabel=None,
    cMap="viridis",
    logx=False,
    logy=False,
    grid=False,
    legend=True,
    legendLoc=None,
    show=True,
    bAvg=False,
):
    """Plot a two-parameter scan, optionally averaging repeated x values."""
    unique_scan_var_1 = np.unique(scanVar1)
    color_map = plt.get_cmap(cMap)
    colors = color_map(np.linspace(0, 1, len(unique_scan_var_1)))

    fig = plt.figure()
    for index, category in enumerate(unique_scan_var_1):
        category_mask = scanVar1 == category
        category_scan_var_2 = scanVar2[category_mask]
        category_y = y[category_mask]

        if bAvg:
            unique_scan_var_2 = np.unique(category_scan_var_2)
            y_average = np.zeros_like(unique_scan_var_2, dtype=float)
            y_error = np.zeros_like(unique_scan_var_2, dtype=float)
            for value_index, value in enumerate(unique_scan_var_2):
                value_mask = category_scan_var_2 == value
                values = category_y[value_mask]
                y_average[value_index] = np.mean(values)
                y_error[value_index] = np.std(values) / np.sqrt(len(values))
            category_scan_var_2 = unique_scan_var_2
            category_y = y_average
            label = f"{paramName1}={category:.2f} (avg)"
            plt.errorbar(
                category_scan_var_2,
                category_y,
                yerr=y_error,
                fmt="o",
                label=label,
                color=colors[index],
            )
        else:
            label = f"{paramName1}={category:.2f}"
            plt.scatter(category_scan_var_2, category_y, label=label, color=colors[index])

    plt.xlabel(xlabel if xlabel else paramName2)
    plt.ylabel(ylabel if ylabel else "y")
    if legend:
        plt.legend(loc="best" if legendLoc is None else legendLoc)
        if legendLoc == "outside":
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.subplots_adjust(right=0.75)
    if title:
        plt.title(title)
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    if grid:
        plt.grid()
    if show:
        plt.show()
    return fig


def plot_2Ddata_heatmap(
    scanVar1,
    scanVar2,
    y,
    paramName1="x",
    paramName2="y",
    paramNameC="z",
    title=None,
    xlabel=None,
    ylabel=None,
    cMap="viridis",
    logx=False,
    logy=False,
    grid=False,
    show=True,
):
    """Plot averaged two-parameter scan values as a heatmap."""
    unique_scan_var_1 = np.unique(scanVar1)
    unique_scan_var_2 = np.unique(scanVar2)
    heatmap = np.full((len(unique_scan_var_2), len(unique_scan_var_1)), np.nan)

    delta_scan_var_1 = np.mean(np.diff(unique_scan_var_1)) / 2
    delta_scan_var_2 = np.mean(np.diff(unique_scan_var_2)) / 2
    for index_1, value_1 in enumerate(unique_scan_var_1):
        for index_2, value_2 in enumerate(unique_scan_var_2):
            mask = (scanVar1 == value_1) & (scanVar2 == value_2)
            if np.any(mask):
                heatmap[index_2, index_1] = np.mean(y[mask])

    fig, ax = plt.subplots()
    image = ax.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        cmap=cMap,
        extent=[
            unique_scan_var_1[0] - delta_scan_var_1,
            unique_scan_var_1[-1] + delta_scan_var_1,
            unique_scan_var_2[0] - delta_scan_var_2,
            unique_scan_var_2[-1] + delta_scan_var_2,
        ],
    )
    if len(unique_scan_var_1) <= 15:
        ax.set_xticks(unique_scan_var_1)
    if len(unique_scan_var_2) <= 15:
        ax.set_yticks(unique_scan_var_2)
    ax.set_xlabel(xlabel if xlabel else paramName1)
    ax.set_ylabel(ylabel if ylabel else paramName2)
    if title:
        ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(paramNameC)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if grid:
        ax.grid(visible=True, which="both", color="gray", linestyle="--", linewidth=0.5)
    if show:
        plt.show()
    return fig
