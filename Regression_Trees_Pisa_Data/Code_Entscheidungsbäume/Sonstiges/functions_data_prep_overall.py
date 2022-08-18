import itertools
from collections import Counter
from pathlib import Path
from shutil import copyfile

import country_converter as coco
import pandas as pd
from tqdm import tqdm

converter = coco.CountryConverter()


def copy_relevant_raw_data_to_prep(key, folder_raw, folder_prep, replace=False):
    """create copy of each data set that includes values and add to prepare folder"""
    path_raw = Path(folder_raw) / f"{key}.csv"
    path_prep = Path(folder_prep) / f"{key}.csv"
    empty_content = "No Results Found - No Results Found"
    server_error = "Internal Server Error"

    if not path_prep.parent.is_dir():
        path_prep.parent.mkdir()

    if path_prep.is_file() and not replace:
        return

    else:
        with open(path_raw, "r", encoding="utf-8") as file:
            content = file.read()
        if empty_content in content or server_error in content:
            return

    copyfile(path_raw, path_prep)


def delete_empty_rows(key, folder):
    """delete all empty rows in the dataset of key folder: prep folder, NOT raw folder"""
    path = Path(folder) / f"{key}.csv"

    with open(path, "r+", encoding="utf-8") as file:
        lines = file.readlines()
        non_empty_lines = [x for x in lines if x.strip()]
        file.seek(0)
        file.truncate()
        file.writelines(non_empty_lines)

    return


def prepare_data_in_csv(key, folder_prep):
    """calls all preperation folders required so far - if you want to overright currently prepared data: replace=True"""
    path_prep = Path(folder_prep) / f"{key}.csv"
    if path_prep.is_file():
        delete_empty_rows(key, folder_prep)


#######################Functions in PD ##########################


def analyse_columns(keys, foldername):
    """identify all remaining column names of all files"""
    col_names = []
    path = Path(foldername)
    pbar = tqdm(keys)

    for key in pbar:
        pbar.set_description(key)
        file_key = path / f"{key}.csv"
        df = pd.read_csv(file_key, nrows=1)
        key_col_names = df.columns.to_list()
        col_names.extend(key_col_names)

    col_count = Counter(col_names)
    return pd.DataFrame.from_dict(col_count, orient="index").reset_index()


# Findings:
# Flag Codes and Flags appears in every Set - empty => can be deleted
# Value column contains data we want -> has to be renamed
# Power Code and Power Code Code are redundant - Power Code is the unit of the value
# All UPPERCASE columns are redundant in any way => can be deleted
# All columns ending with "code" are redundant


def delete_entire_columns(dataframe, column_names_to_delete=None):
    """give df and a list of column names and the function drops all columns, returns df"""

    if not column_names_to_delete:
        column_names_to_delete = []

    if not isinstance(column_names_to_delete, list):
        column_names_to_delete = [column_names_to_delete]

    dataframe_new = dataframe.drop(labels=column_names_to_delete, axis=1)
    dataframe_new = dataframe_new.dropna(axis=1, how="all")

    return dataframe_new


def remove_all_upper_case_columns(dataframe):
    """Drops all columns with only UPPERCASE column names as they are redundant. returns df"""
    col_list = dataframe.columns.to_list()
    col_lower = [x.lower() for x in col_list]
    col_names_to_drop = []

    for name in col_lower:
        if name.upper() in col_list:
            col_names_to_drop.append(name.upper())

    return delete_entire_columns(dataframe, col_names_to_drop)


def delete_columns_with_identical_content(dataframe, list_columns_suspected_equal=None):
    if not list_columns_suspected_equal:
        list_columns_suspected_equal = dataframe.columns.to_list()

    if not isinstance(list_columns_suspected_equal, list):
        list_columns_suspected_equal = [list_columns_suspected_equal]

    list_columns_suspected_equal = [x for x in list_columns_suspected_equal if x != "unit" and
                                    x != "unit coefficient" and x != "measure" and x != "indicator"]

    iteration = list(itertools.combinations(list_columns_suspected_equal, 2))
    to_drop = set()

    for t in iteration:
        unique_values_col1 = dataframe[t[0]].nunique()
        unique_values_col2 = dataframe[t[1]].nunique()
        if unique_values_col1 == unique_values_col2:
            dataframe["temp"] = dataframe[t[0]].astype(str) + dataframe[t[1]].astype(str)
            test_col = dataframe["temp"].nunique()
            dataframe = dataframe.drop("temp", axis=1)
            if test_col == dataframe[t[0]].nunique():
                to_drop.add(t[0])

    dataframe_new = delete_entire_columns(dataframe, list(to_drop))

    return dataframe_new


def drop_columns_ending_on_code(dataframe):
    """Drops all columns where col name ends with "code" as they are redundant. returns df"""
    col_list = dataframe.columns.to_list()
    columns_to_drop = []

    for name in col_list:
        if name.lower().endswith("code") or name.lower().endswith("codes"):
            columns_to_drop.append(name)

    return delete_entire_columns(dataframe, columns_to_drop)


def delete_unnecessary_columns(dataframe, list_to_leave_out_delete_code_upper_single_identical=None):
    """deletes all unneeded columns (UPPERCASE, ending in code, empty columns, returns df"""
    all_functions = ["delete", "code", "upper", "single", "identical"]

    if not list_to_leave_out_delete_code_upper_single_identical:
        list_to_leave_out_delete_code_upper_single_identical = []

    if not isinstance(list_to_leave_out_delete_code_upper_single_identical, list):
        list_to_leave_out_delete_code_upper_single_identical = [list_to_leave_out_delete_code_upper_single_identical]

    requested_functions = [x for x in all_functions if x not in list_to_leave_out_delete_code_upper_single_identical]

    if "delete" in requested_functions:
        columns_to_delete = ["flags", "unnamed: 0"]
        delete_cols = [x for x in columns_to_delete if x in dataframe.columns]

        dataframe = delete_entire_columns(dataframe, delete_cols)

    if "code" in requested_functions:
        dataframe = drop_columns_ending_on_code(dataframe)

    if "upper" in requested_functions:
        dataframe = remove_all_upper_case_columns(dataframe)

    if "single" in requested_functions:
        dataframe = check_and_delete_columns_containing_single_value(dataframe)

    if "identical" in requested_functions:
        dataframe = delete_columns_with_identical_content(dataframe)

    return dataframe


def delete_entries(dataframe, columns_to_consider, nan=False, condition=None):
    """input df and parameters to drop certain rows in data accorinding to params. return df"""
    if not isinstance(columns_to_consider, list):
        columns_to_consider = [columns_to_consider]

    if nan:
        if columns_to_consider == ["value"]:
            dataframe_new = delete_empty_entries(dataframe)
        dataframe_new = dataframe_new.dropna(subset=columns_to_consider)

    else:
        dataframe_new = dataframe.copy()
        for col in columns_to_consider:
            dataframe_new = dataframe_new[dataframe_new[col] == condition]

    return dataframe_new


def delete_empty_entries(dataframe):
    """All entries that habe a 0 in the "value" column or no data in the "value" column are irrelevant """

    dataframe["value"] = pd.to_numeric(dataframe["value"])
    dataframe_new = dataframe.loc[dataframe["value"] != 0, :]

    return dataframe_new


def check_and_delete_columns_containing_single_value(dataframe):
    """drops columns that only include a single value for all rows. reurns df"""
    single_value_cols = []

    for col in dataframe.columns:
        values = dataframe[col].unique()
        if len(values) <= 1:
            single_value_cols.append(col)

    single_value_cols = [x for x in single_value_cols if x != "unit" and
                         x != "unit coefficient" and
                         x != "measure" and
                         x != "indicator"]
    return delete_entire_columns(dataframe, single_value_cols)


def prepare_special_column_names(dataframe):
    special_names = ["Value", "Frequency", "Unnamed: 0", "Flags", "Country", "Unit", "Measure", "Indicator",
                     "Reference Period", "Country", "Declaring country", "Reporter country", "Donor",
                     "Reporting country", "Economy", "Reporter Country", "Reporter"]
    names_required_changing = [x for x in special_names if x in dataframe.columns]

    if "PowerCode" in dataframe.columns:
        dataframe = dataframe.rename(columns={"PowerCode": "unit coefficient"})

    for col in names_required_changing:
        dataframe = dataframe.rename(columns={col: col.lower()})

    return dataframe
