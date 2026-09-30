import numpy as np
import pandas as pd


def load_csv_data(path, filename, valName, genParams, b2D=False):
    data = pd.read_csv(f"{path}\\{filename}.csv", encoding="utf-8", skipinitialspace=True)
    if data.empty:
        raise ValueError("The loaded data is empty. Please check the file path and name.")

    values = data[valName]
    params = data.drop(columns=valName)
    params_diff = params.loc[:, params.nunique() > 1]
    scan_param = params_diff.drop(columns=[col for col in genParams if col in params_diff.columns])

    return {
        "data": data,
        "values": values,
        "scanParam": scan_param,
        "scanParamName": scan_param.columns.tolist(),
    }


def load_multiple_files(paths, filenames, valName, genParams, b2D=False, filetype="csv", scanAuto=True, **kwargs):
    data_list = []
    values_list = []
    scan_param_list = []
    scan_param_name_list = []
    scan_var_1 = []
    scan_var_2 = []
    scan_var_names = []

    for path, filename in zip(paths, filenames):
        if filetype.lower() != "csv":
            raise ValueError("Currently only 'csv' filetype is supported.")

        file_data = load_csv_data(path, filename, valName, genParams, b2D)
        scan_param = file_data["scanParam"]
        scan_param_names = file_data["scanParamName"]
        data_list.append(file_data["data"])
        values_list.append(file_data["values"])
        scan_param_list.append(scan_param)
        scan_param_name_list.append(scan_param_names)

        number_of_scan_vars = 2 if b2D else 1
        if scanAuto:
            scan_var_names = scan_param_names[:number_of_scan_vars]
        else:
            scan_var_names = kwargs.get("scanVarNames")
            if scan_var_names and len(scan_var_names) != number_of_scan_vars:
                raise ValueError(
                    f"Length of scanVarNames must be {number_of_scan_vars} for the specified b2D value."
                )

        if number_of_scan_vars == 1 and scan_var_names:
            scan_var_1.append(scan_param[scan_var_names[0]].to_numpy())
            scan_var_2 = None
        elif number_of_scan_vars == 2 and scan_var_names:
            scan_var_1.append(scan_param[scan_var_names[0]].to_numpy())
            scan_var_2.append(scan_param[scan_var_names[1]].to_numpy())

    return {
        "data": pd.concat(data_list, ignore_index=True) if data_list else pd.DataFrame(),
        "values": pd.concat(values_list, ignore_index=True) if values_list else pd.DataFrame(),
        "scanParam": pd.concat(scan_param_list, ignore_index=True) if scan_param_list else pd.DataFrame(),
        "scanParamName": scan_param_name_list[0] if scan_param_name_list else [],
        "scanVar1": np.concatenate(scan_var_1).flatten() if scan_var_1 else None,
        "scanVar2": np.concatenate(scan_var_2).flatten() if scan_var_2 else None,
        "scanVarNames": scan_var_names,
    }


def get_ExpObserv(values, sample_factor=1):
    return (
        values["atom_number_fit"].to_numpy() * sample_factor,
        values["sig_xx"].to_numpy() * sample_factor,
        values["sig_yy"].to_numpy() * sample_factor,
        values["x0"].to_numpy() * sample_factor,
        values["y0"].to_numpy() * sample_factor,
    )
