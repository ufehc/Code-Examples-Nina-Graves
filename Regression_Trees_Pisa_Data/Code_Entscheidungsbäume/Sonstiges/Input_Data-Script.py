from functions_input_data import *
from pathlib import Path
import itertools
from tqdm import tqdm

# Definition of several parameters
Input_Data = ["aggregated-years", "pisa-years"]
Pisa_Type = ["MATH", "READ", "SCIENCE"]
path_0 = Path(FOLDERNAME_MODELLING_DATA)
path_input = Path(path_0/"Input Data")

# location of pisa data
path_pisa = Path(FOLDERNAME_PISA)
path_final_data = Path(FOLDERNAME_FINAL_DATA)

# percentages of non-na Data we want to compare
Iterations = np.arange(50, 105, 5)

# remove for script - overview table of feature information
index = itertools.product(Input_Data, Pisa_Type, Iterations)
overview_input_data = pd.DataFrame(columns=["Inpu_Data_Type", "Pisa-Type", "Percentage_non_NA", "Number_Features",
                                            "Average_Percentage_NAs", "Average_Percentage_apprximated_Interpolation",
                                            "Average_Percentage_approximated_Annual_Mean"], index=pd.MultiIndex.from_tuples(index))

# For each type of input Data
InputData = tqdm(Input_Data)
for input_data in InputData:
    InputData.set_description(f"{input_data}")

    # create Folders - one Folder per Input type
    path_0.mkdir(exist_ok=True)
    path_input.mkdir(exist_ok=True)
    path_data_type = Path(path_input / input_data)
    path_data_type.mkdir(exist_ok=True)

    # for every Pisa type seperately, as they start in different years
    pbar0 = tqdm(Pisa_Type)
    for pisa_type in pbar0:
        pbar0.set_description(f"{input_data}-{pisa_type}")
        pisa_type_lower = str.lower(pisa_type)

        # load Pisa Data
        pisa_file_name = f"20-11-05_PISA_{pisa_type_lower}_total.csv"
        pisa_data = pd.read_csv(path_pisa / pisa_file_name)

        pisa_data = pisa_data.rename(columns={"value": "PISA-SCORE"})
        pisa_data = pisa_data.set_index("Index")

        # load dataset_keys for all data sets that have been prepared up to the final state
        datasets_keys = [x.name.replace(".csv", "") for x in path_final_data.iterdir() if x.is_file()]

        # differentiation between different percentages of non-na values
        pbar = tqdm(Iterations)
        for iteration in pbar:
            pbar.set_description(f"{input_data}-{pisa_type}-{iteration}")

           # destinguish between input data types
            if input_data == "aggregated_years":
                data_for_model = create_data_in_pisa_structure_aggregate_years(pisa_data, datasets_keys, iteration, True)
            else:
                data_for_model = create_data_in_pisa_structure_only_pisa_years_no_aggregation(pisa_data, datasets_keys,
                                                                                              iteration, True)
            # remove Pisa entries
            data_for_model = remove_PISA_data_from_data(data_for_model)

            # replace nan values and save csv
            data_for_model, column_information = interpolate_nan_values_or_set_mean_if_not_possible(data_for_model)
            data_for_model.to_csv(Path(path_input/input_data)/f"{pisa_type}-{iteration}.csv")

            # remove for script - overview of feature Information
            ind = (input_data, pisa_type, iteration)
            overview_input_data.loc[ind, "Inpu_Data_Type"] = input_data
            overview_input_data.loc[ind, "Pisa-Type"] = pisa_type
            overview_input_data.loc[ind, "Percentage_non_NA"] = iteration
            overview_input_data.loc[ind, "Number_Features"] = len(column_information)-3
            overview_input_data.loc[ind, "Average_Percentage_NAs"] = column_information.iloc[3:, 1].mean()
            overview_input_data.loc[ind, "Average_Percentage_apprximated_Interpolation"] = column_information.iloc[3:, 3].mean()
            overview_input_data.loc[ind, "Average_Percentage_approximated_Annual_Mean"] = column_information.iloc[3:, 5].mean()


            # store column information - only needed for iterations = 50, as it includes all features that are also
            # included in the other iterations
            if iteration == 50:
                feature_columns = [x for x in data_for_model.columns if x != "PISA-SCORE" and x != "COUNTRY"]
                column_information.to_csv(Path(path_input/input_data) / f"{pisa_type}-Column_information.csv")

# remove from script
overview_input_data.to_csv(path_input / "Overview_Input_Data_Types.csv")