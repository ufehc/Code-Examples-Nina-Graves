from functions_data_prep_overall import *


def bring_data_to_desired_structure_and_save(dataframe, key, folder_for_final_data, overwrite=False):

    if "year" not in dataframe.columns:
        return "No year found"

    if "country" not in dataframe.columns:
        return "No country found"

    dataframe_final = categorise_values_in_dataset(dataframe, key)

    store_dataset_in_final_structure_folder(dataframe_final, key, folder_for_final_data, overwrite)

    return "Data transformed"


def categorise_values_in_dataset(dataframe, key):
    """transform units to [units], create new columns and add values to appropriate columns"""
    unit = ["unit coefficient", "unit"]
    new_unit_colname = "UNIT"
    explanation = ["indicator", "measure"]
    new_expl_col_name = "EXPLANATION"
    new_content_col_name = "CONTENT"
    prepared_category_name = "CATEGORY"

    u_nit = [x for x in unit if x in dataframe.columns]
    dataframe = merge_col_content_and_delete_old_cols(dataframe, u_nit, new_unit_colname, in_brackets=True)

    expl = [x for x in explanation if x in dataframe.columns]
    dataframe = merge_col_content_and_delete_old_cols(dataframe, expl, new_expl_col_name, key=key, in_brackets=False)

    column_names_not_to_merge = ["country", "year", "value", new_unit_colname, new_expl_col_name]
    col_merges = [x for x in dataframe.columns if x not in column_names_not_to_merge]
    dataframe = merge_col_content_and_delete_old_cols(dataframe, col_merges, new_content_col_name, merge_symbol="+")

    merge_to_single_col = [new_expl_col_name, new_content_col_name, new_unit_colname]
    dataframe = merge_by_colon_then_space(dataframe, merge_to_single_col, prepared_category_name)

    index_merge = ["country", "year"]
    dataframe = merge_col_content_and_delete_old_cols(dataframe, index_merge, "Index", "-")
    dataframe = dataframe.set_index("Index")

    dataframe_final = dataframe.pivot_table(columns=prepared_category_name, values="value", index="Index")

    return dataframe_final


def merge_col_content_and_delete_old_cols(dataframe, list_of_cols_to_merge, new_col_name, merge_symbol=None, key=None,
                                          in_brackets=False):
    if merge_symbol:
        if merge_symbol == "," or merge_symbol == ":":
            merge_symbol = f"{merge_symbol} "
        if merge_symbol == "_" or merge_symbol == "-":
            merge_symbol = f"{merge_symbol}"
        else:
            merge_symbol = f" {merge_symbol} "
    else:
        merge_symbol = " "

    dataframe[new_col_name] = dataframe[list_of_cols_to_merge].astype(str).agg(merge_symbol.join, axis=1)

    if key:
        dataframe[new_col_name] = key + " - " + dataframe[new_col_name]

    if in_brackets:
        dataframe[new_col_name] = "[" + dataframe[new_col_name] + "]"

    dataframe = delete_entire_columns(dataframe, list_of_cols_to_merge)

    return dataframe


def merge_by_colon_then_space(dataframe, list_of_col_names, new_col_name):
    colon_merge_col_name = "EPL_CAT"

    if len(list_of_col_names) == 3:
        colon_merge = list_of_col_names[:2]
        dataframe = merge_col_content_and_delete_old_cols(dataframe, colon_merge, colon_merge_col_name,
                                                          merge_symbol=":")
        space_merge = [colon_merge_col_name, list_of_col_names[2]]
        dataframe = merge_col_content_and_delete_old_cols(dataframe, space_merge, new_col_name)

    return dataframe


def store_dataset_in_final_structure_folder(dataframe, key, folder_for_final_data, overwrite=False):
    path_final = Path(folder_for_final_data) / f"{key}.csv"

    if not path_final.parent.is_dir():
        path_final.parent.mkdir()

    if path_final.is_file() and not overwrite:  # if already exists, do nothing
        return

    return dataframe.to_csv(path_final)
