"""Shared analysis settings used by the GUI and helper modules."""

# Columns containing fitted observables rather than scan coordinates.
VAL_NAME = [
    "pixel_sum", "i0f", "x0", "y0", "sig_x", "sig_y", "foffset",
    "theta", "sig_xx", "sig_yy", "icount", "atom_number_fit",
]
# Metadata columns excluded when detecting scan parameters.
GEN_PARAMS = ["run", "time", "listiterationnumber", "sequenceduration", "run_path", "IterationNum"]
# Multipliers and default locations used by the current analysis workflow.
SAMPLE = 1
ATOMNUM_FACTOR = 4.34
DEFAULT_CSV_DIR = r"Z:\SmithGroup\data"
