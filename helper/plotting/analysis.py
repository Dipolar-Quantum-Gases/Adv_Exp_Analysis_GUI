import matplotlib.pyplot as plt
import numpy as np

from helper.config import SAMPLE
from helper.physDataproc import get_ExpObserv
from helper.physPlot import plot_2Ddata_heatmap, plot_2Ddata_scatter


def plot_loaded_data(filedata, scan_var_names=None, plot_kind="1d"):
    if filedata is None:
        return
    atomnumber, _, _, _, _ = get_ExpObserv(filedata["values"], sample_factor=SAMPLE)
    plot_names = [name for name in (scan_var_names or []) if name]
    if plot_kind == "1d":
        x_vals = np.asarray(filedata["scanVar1"], dtype=float) if plot_names and filedata.get("scanVar1") is not None else np.arange(len(atomnumber))
        plt.figure()
        plt.scatter(x_vals, atomnumber, s=18)
        plt.xlabel(plot_names[0] if plot_names else "Sample index")
        plt.ylabel("Atom Number")
        plt.grid(True)
        plt.tight_layout()
    elif len(plot_names) >= 2 and filedata.get("scanVar2") is not None:
        args = (np.asarray(filedata["scanVar1"], dtype=float), np.asarray(filedata["scanVar2"], dtype=float), atomnumber)
        if plot_kind == "2d_scatter":
            plot_2Ddata_scatter(*args, paramName1=plot_names[0], paramName2=plot_names[1], ylabel="Atom Number", cMap="plasma", grid=True, legend=True, legendLoc="outside", show=False, bAvg=True)
        elif plot_kind == "2d_heatmap":
            plot_2Ddata_heatmap(*args, paramName1=plot_names[0], paramName2=plot_names[1], paramNameC="Atom Number", cMap="plasma", show=False)
        else:
            raise ValueError(f"Unknown plot kind: {plot_kind}")
    else:
        raise ValueError("Two scan parameters are required for 2D analysis.")
    plt.show()
