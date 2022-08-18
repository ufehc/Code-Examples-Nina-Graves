from pathlib import Path
import numpy as np
import pandas as pd
from Global_configuartions import *
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def create_data_in_pisa_structure_only_pisa_years_no_aggregation(pisa_data, list_of_keys_of_datasets,
                                  minimum_amount_of_entries=None, min_amount_in_percent=False):
    """merge data from all datsets of the years and countries included in PISA. Chose min number of entries per
    column"""
    path_final_data = Path(FOLDERNAME_FINAL_DATA)
    relevant_entries = list(pisa_data.index)
    merged_dataframe = pisa_data.copy()

    if (not minimum_amount_of_entries) or minimum_amount_of_entries==0:
        minimum_amount_of_entries = 1

    if min_amount_in_percent:
        threshold = int(np.ceil((minimum_amount_of_entries/100)*len(relevant_entries)))
    else:
        threshold = int(np.ceil(minimum_amount_of_entries))

    pbar = tqdm(list_of_keys_of_datasets)
    for dataset in pbar:
        pbar.set_description(dataset)
        data = pd.read_csv(path_final_data / f"{dataset}.csv")
        data = data.set_index("Index")

        irrelevant_entries = [x for x in data.index if x not in relevant_entries]
        data = data.drop(index=irrelevant_entries)
        data = data.dropna(axis=1, thresh=threshold)

        merged_dataframe = merged_dataframe.merge(data, how="left", on="Index")

    return merged_dataframe

def aggregate_annual_data_according_to_pisa_in_dataset(data_dataset, years_relevant, interval):
    """merge data for the years relevant to data aggregating the results of the past 3 years"""
    min_thresh = years_relevant[0]-interval

    data_dataset["YEAR"] = data_dataset.index.str[-4:].astype(int)
    data_dataset["COUNTRY"] = data_dataset.index.str[:3]
    rows_to_drop = list(data_dataset.loc[data_dataset["YEAR"] <= min_thresh].index)
    data_dataset = data_dataset.drop(rows_to_drop, axis=0)

    data_dataset["GROUP_YEAR"] = data_dataset.loc[:, "YEAR"].apply(lambda y: min(int(x) for x in years_relevant if x >= y))
    data_dataset["Weight"] = 3-(data_dataset["GROUP_YEAR"]-data_dataset["YEAR"])
    data_dataset["GROUP"] = data_dataset[["COUNTRY", "GROUP_YEAR"]].astype(str).agg("-".join, axis=1)
    col_list = [x for x in data_dataset.columns if x != "Weight" and x != "COUNTRY" and x != "GROUP"]
    data_dataset.loc[:, col_list] = data_dataset.loc[:, col_list].apply(lambda x: x*data_dataset.loc[x.index, "Weight"])

    data_dataset = data_dataset.groupby("GROUP", axis=0).sum(min_count=1)
    data_dataset.loc[:, col_list] = data_dataset.loc[:, col_list].apply(lambda x: x/data_dataset.loc[x.index, "Weight"])
    data_dataset = data_dataset.drop(columns=["YEAR", "GROUP_YEAR", "Weight"])

    return data_dataset

def analyse_pisa_data_concerning_annual_data(pisa_dataframe):
    start_year = min(pisa_dataframe["YEAR"])
    end_year = max(pisa_dataframe["YEAR"])
    number_of_entries = len(pisa_dataframe["YEAR"].unique())
    interval = (end_year-start_year)/(number_of_entries-1)

    years_relevant = np.arange(start_year, end_year+1, interval).tolist()

    return years_relevant, interval


def create_data_in_pisa_structure_aggregate_years(pisa_data, list_of_keys_of_datasets,
                                  minimum_amount_of_entries=None, min_amount_in_percent=False):
    """merge the data of all datasets of the aggregated annual data to the pisa-years. Chose min number of entries per
    column"""
    path_final_data = Path(FOLDERNAME_FINAL_DATA)
    relevant_entries = list(pisa_data.index)
    merged_dataframe = pisa_data.copy()

    years_relevant, interval = analyse_pisa_data_concerning_annual_data(pisa_data)

    if (not minimum_amount_of_entries) or minimum_amount_of_entries==0:
        minimum_amount_of_entries = 1

    if min_amount_in_percent:
        threshold = int(np.ceil((minimum_amount_of_entries/100)*len(relevant_entries)))
    else:
        threshold = int(np.ceil(minimum_amount_of_entries))

    pbar = tqdm(list_of_keys_of_datasets)
    for dataset in pbar:
        pbar.set_description(dataset)
        data = pd.read_csv(path_final_data / f"{dataset}.csv")
        data = data.set_index("Index")
        data = data.apply(pd.to_numeric)

        data_new = aggregate_annual_data_according_to_pisa_in_dataset(data, years_relevant, interval)

        merged_dataframe = merged_dataframe.merge(data_new, how="left", left_index=True, right_index=True)

        merged_dataframe = merged_dataframe.dropna(axis=1, thresh=threshold)

    return merged_dataframe


def remove_PISA_data_from_data(model_data):
    pisa_columns = ["FAMILY - Country mean average score in mathematics, by sex :  Female [Units Number]",
                    "FAMILY - Country mean average score in mathematics, by sex :  Male [Units Number]",
                    "FAMILY - Country mean average score in reading, by sex :  Female [Units Number]",
                    "FAMILY - Country mean average score in reading, by sex :  Male [Units Number]",
                    "FAMILY - Country mean average score in science, by sex :  Female [Units Number]",
                    "FAMILY - Country mean average score in science, by sex :  Male [Units Number]",
                    "GFG -  :  PISA: Educational achievement - Maths [Units]",
                    "GFG -  :  PISA: Educational achievement - Reading [Units]",
                    "GFG -  :  PISA: Educational achievement - Sciences [Units]",
                    "GFG -  :  PISA: Educational achievement [Units]",
                    "GFG -  :  PISA: Influence of socio-eco and cultural background on student reading perf. [Units]",
                    "GFG -  :  PISA: Variance of educational achievement - Maths [Units]",
                    "GFG -  :  PISA: Variance of educational achievement - Reading [Units]",
                    "GFG -  :  PISA: Variance of educational achievement - Sciences [Units]",
                    "GFG -  :  PISA: Variance of educational achievement [Units]",
                    "HSL - Students with low skills :  Deprivation + Current Well-being + Total population + Total population + Total population []",
                    "HSL - Student skills (science) :  Gap between top and bottom performers (vertical inequality) + Current Well-being + Total population + Total population + Total population []",
                    "HSL - Student skills (science) :  Gap between groups (horizontal inequality) + Current Well-being + Women + Total population + Total population []",
                    "HSL - Student skills (science) :  Gap between groups (horizontal inequality) + Current Well-being + Total population + Total population + Tertiary []",
                    "HSL - Student skills (science) :  Gap between groups (horizontal inequality) + Current Well-being + Total population + Total population + Secondary []",
                    "HSL - Student skills (science) :  Gap between groups (horizontal inequality) + Current Well-being + Total population + Total population + Primary []",
                    "HSL - Student skills (science) :  Gap between groups (horizontal inequality) + Current Well-being + Men + Total population + Total population []",
                    "HSL - Student skills (science) :  Average + Current Well-being + Total population + Total population + Total population []",
                    "HSL - Student skills (reading) :  Gap between top and bottom performers (vertical inequality) + Current Well-being + Total population + Total population + Total population []",
                    "HSL - Student skills (reading) :  Gap between groups (horizontal inequality) + Current Well-being + Women + Total population + Total population []",
                    "HSL - Student skills (reading) :  Gap between groups (horizontal inequality) + Current Well-being + Total population + Total population + Tertiary []",
                    "HSL - Student skills (reading) :  Gap between groups (horizontal inequality) + Current Well-being + Total population + Total population + Secondary []",
                    "HSL - Student skills (reading) :  Gap between groups (horizontal inequality) + Current Well-being + Total population + Total population + Primary []",
                    "HSL - Student skills (reading) :  Gap between groups (horizontal inequality) + Current Well-being + Men + Total population + Total population []",
                    "HSL - Student skills (reading) :  Average + Current Well-being + Total population + Total population + Total population []",
                    "HSL - Student skills (maths) :  Gap between top and bottom performers (vertical inequality) + Current Well-being + Total population + Total population + Total population []",
                    "HSL - Student skills (maths) :  Gap between groups (horizontal inequality) + Current Well-being + Women + Total population + Total population []",
                    "HSL - Student skills (maths) :  Gap between groups (horizontal inequality) + Current Well-being + Total population + Total population + Tertiary []",
                    "HSL - Student skills (maths) :  Gap between groups (horizontal inequality) + Current Well-being + Total population + Total population + Secondary []",
                    "HSL - Student skills (maths) :  Gap between groups (horizontal inequality) + Current Well-being + Total population + Total population + Primary []",
                    "HSL - Student skills (maths) :  Gap between groups (horizontal inequality) + Current Well-being + Men + Total population + Total population []",
                    "HSL - Student skills (maths) :  Average + Current Well-being + Total population + Total population + Total population []"]

    pisa_cols_in_data = [x for x in pisa_columns if x in model_data.columns]

    model_data = model_data.drop(pisa_cols_in_data, axis=1)

    return model_data


def interpolate_nan_values_or_set_mean_if_not_possible(model_data):
    """Substitutes all nan data in data for modelling. For data not aquired for single years in data a linear
    interpolation is used to estimate nan data. For countries where no interpolation is possible the mean of the year
    in question is added as estimaton. Returns the data as df and another dataframe with information on the columns in
    regard to the preparation"""
    na_values = model_data.isna().sum()
    column_information = na_values.to_frame("NA-Values")
    total_number = len(model_data)
    column_information["NA-Values [%]"] = (column_information["NA-Values"] / total_number)*100

    model_data = model_data.set_index(["COUNTRY", "YEAR"])

    model_data_interpolate = model_data.groupby([pd.Grouper(level="COUNTRY"),
                                          pd.Grouper(level="YEAR")]).mean()

    model_data_interpolate = model_data_interpolate.groupby(level=[0]).apply(lambda g: g.interpolate(limit_direction="both",
                                                                                                     method='linear'))

    column_information["Annual Mean"] = model_data_interpolate.isna().sum()
    column_information["Annual Mean [%]"] = (column_information["Annual Mean"] / total_number) * 100
    column_information["Inpolated"] = column_information["NA-Values"] - column_information["Annual Mean"]
    column_information["Inpolated [%]"] = (column_information["Inpolated"] / total_number) * 100
    column_information = column_information.reindex(columns=["NA-Values", "NA-Values [%]", "Inpolated", "Inpolated [%]",
                                                             "Annual Mean", "Annual Mean [%]"])

    model_data_translation_missing_values = model_data_interpolate.groupby(level=[1]).mean()
    model_data_interpolate = model_data_interpolate.fillna(model_data_translation_missing_values)

    model_data_interpolate = model_data_interpolate.reset_index(level='YEAR')
    model_data_interpolate = model_data_interpolate.reset_index(level='COUNTRY')
    model_data_interpolate["Index"] = model_data_interpolate[["COUNTRY", "YEAR"]].astype(str).agg("-".join, axis=1)
    model_data_interpolate = model_data_interpolate.set_index("Index")

    return model_data_interpolate, column_information




