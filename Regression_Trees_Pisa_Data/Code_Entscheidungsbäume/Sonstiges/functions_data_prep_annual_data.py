from functions_data_prep_country_data import *


def analyse_annual_data(dataframe, start_year=1998, end_year=2018):
    """returns list of colnames with only yearly data and colnames including yearly data"""
    years = [str(x) for x in range(start_year, end_year + 1)]
    relevant_columns = [x for x in dataframe.columns if x != "value"]
    columns_with_years = []
    columns_with_only_years = []

    for col in relevant_columns:
        values = dataframe[col].unique()
        # checks if a year is contained in any of the unique values
        if any(str(x) in years for x in values):
            columns_with_years.append(col)
            # checks whether all values in the columns only contain years
            if all(str(y) in years for y in values):
                columns_with_only_years.append(col)

    return columns_with_years, columns_with_only_years


def frequency_in_annual_data(dataframe):
    """Check whether there is a column called frequency and if so only use annual data and delete column"""

    if "frequency" in dataframe.columns:
        dataframe["frequency"] = dataframe["frequency"].str.lower()

        if "annual" in dataframe["frequency"].unique():
            dataframe_new = dataframe[dataframe["frequency"] == "annual"]
            dataframe_new = delete_entire_columns(dataframe_new, "frequency")

            status_frequency = "Frequency: Filtered to annual data in 1 column"
            return dataframe_new, status_frequency

        else:
            status_frequency = "Frequency: Different term for annual data in column frequency"
    else:
        status_frequency = "Frequency: No column 'frequency'"

    return dataframe, status_frequency


def check_and_edit_annual_data(dataframe, start_year, end_year):
    """prepares annual data as far as possible, returns df and information on annual data"""

    columns_with_years, columns_with_only_years = analyse_annual_data(dataframe, start_year, end_year)

    if len(columns_with_only_years) == 1 and len(columns_with_years) == 1:
        dataframe_new = dataframe.rename({columns_with_only_years[0]: "year"}, axis=1)
        status = "As required - One column with annual data"
        return dataframe_new, status

    if len(columns_with_years) == 0:
        status = "No annual data available"
        return dataframe, status

    dataframe, status_frequency = frequency_in_annual_data(dataframe)

    status = f"{status_frequency}, " \
                 f"Including years: {len(columns_with_years)}, " \
                 f"Only year-entries: {len(columns_with_only_years)}"

    if status_frequency == "Frequency: Filtered to annual data in 1 column":
        status = "Rerun"
    elif status_frequency == "Frequency: No column 'frequency'" and len(columns_with_years) == 1:
        years = [str(x) for x in range(start_year, end_year + 1)]
        dataframe = dataframe[dataframe[columns_with_years[0]].isin(years)]
        dataframe = dataframe.rename({columns_with_years[0]: "year"}, axis=1)
        status = "Rerun"
    elif status_frequency == "Frequency: No column 'frequency'" and len(columns_with_years) > len(columns_with_only_years):
        if "reference period" in columns_with_years and "reference period" not in columns_with_only_years:
            dataframe = delete_entire_columns(dataframe, "reference period")
            status = "Rerun"

    if "reference period" in dataframe.columns and "reference period" not in columns_with_years:
        dataframe = delete_entire_columns(dataframe, "reference period")

    return dataframe, status


def prepare_annual_data(dataframe, start_year=1998, end_year=2018):
    """Prepares annual data including a rerun after something has changed to offer accurate status"""
    dataframe_new, annual_status = check_and_edit_annual_data(dataframe, start_year, end_year)

    if annual_status == "Rerun":
        dataframe_new, annual_status = check_and_edit_annual_data(dataframe_new, start_year, end_year)

    return dataframe_new, annual_status


def check_key_names_for_year(list_of_datasets_without_year, folder_prep_datasets, start_year=1998, end_year=2018):
    if not list_of_datasets_without_year:
        list_of_datasets_without_year = []

    if not isinstance(list_of_datasets_without_year, list):
        list_of_datasets_without_year = [list_of_datasets_without_year]

    years = [str(x) for x in range(start_year, end_year + 1)]
    year_found = []
    path = Path(folder_prep_datasets)

    for dataset in list_of_datasets_without_year:
        for year in years:
            if year in dataset:
                file_key = path / f"{dataset}.csv"
                temp = pd.read_csv(file_key, encoding="utf-8")
                if "Unnamed: 0" in temp.columns:
                    temp = delete_entire_columns(temp, ["Unnamed: 0"])
                temp["year"] = year
                temp.to_csv(file_key)
                year_found.append(dataset)
                break

    datasets_remaining_without_year = [x for x in list_of_datasets_without_year if x not in year_found]

    return year_found, datasets_remaining_without_year


def check_raw_data_for_upper_year_column_name(list_of_datasets_without_year,
                            folder_raw_datasets,
                            folder_prep_datasets,
                            country_list,
                            start_year=1998, end_year=2018):

    if not list_of_datasets_without_year:
        list_of_datasets_without_year = []

    if not isinstance(list_of_datasets_without_year, list):
        list_of_datasets_without_year = [list_of_datasets_without_year]

    path_prep = Path(folder_prep_datasets)
    year_found = []
    country_status_dict = dict()
    annual_status_dict = dict()

    for dataset in list_of_datasets_without_year:
        copy_relevant_raw_data_to_prep(dataset, folder_raw_datasets, folder_prep_datasets, True)
        prepare_data_in_csv(dataset, folder_prep_datasets)

        file = path_prep / f"{dataset}.csv"
        dataframe = pd.read_csv(file, encoding="utf-8")

        dataframe = prepare_special_column_names(dataframe)
        dataframe = delete_unnecessary_columns(dataframe, ["upper"])

        dataframe = delete_entries(dataframe, "value", True)

        dataframe, annual_status = prepare_annual_data(dataframe, start_year, end_year)
        annual_status_dict[dataset] = annual_status
        if annual_status == "As required - One column with annual data":
            year_found.append(dataset)

        dataframe, country_status = prepare_country_data(dataframe, country_list)
        country_status_dict[dataset] = country_status

        dataframe.to_csv(file)

    datasets_remaining_without_year = [x for x in list_of_datasets_without_year if x not in year_found]
    return datasets_remaining_without_year, annual_status_dict, country_status_dict