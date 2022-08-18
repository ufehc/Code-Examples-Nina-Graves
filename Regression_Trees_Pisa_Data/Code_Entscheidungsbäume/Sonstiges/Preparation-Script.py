from Global_configuartions import *
from bring_data_into_shape import *
from functions_prepare_annual_data import *
from functions_prepare_country_data import *

path_raw = Path(FOLDERNAME_RAW)
path_prep = Path(FOLDERNAME_PREP)
country_csv = Path(FOLDERNAME_PISA) / "PISA_Länder.csv"
with open(country_csv, "r", encoding="utf-8-sig") as file:
    pisa_countries = file.read().splitlines()

keys_downloaded = [x.name.rstrip(".csv") for x in path_raw.iterdir() if x.is_file()]

pbar = tqdm(keys_downloaded)
for key in pbar:
    pbar.set_description(key)
    copy_relevant_raw_data_to_prep(key, FOLDERNAME_RAW, FOLDERNAME_PREP)
    prepare_data_in_csv(key, FOLDERNAME_PREP)

keys_for_prep = [x.name.rstrip(".csv") for x in path_prep.iterdir() if x.is_file()]

overview_df_columns = ["Number of columns", "Number of entries", "Status annual data",
                       "Status country data", "Status transformation"]
overview_keys = pd.DataFrame(index=keys_for_prep, columns=overview_df_columns)

pbar2 = tqdm(keys_for_prep)
for key in pbar2:
    pbar2.set_description(key)

    if not key.startswith("FFS_"):
        file_key = path_prep / f"{key}.csv"
        df = pd.read_csv(file_key, encoding="utf-8")

        df = prepare_special_column_names(df)
        df = delete_unnecessary_columns(df)

        df = delete_entries(df, "value", True)

        df, overview_keys.loc[key, "Status annual data"] = prepare_annual_data(df, STARTYEAR, ENDYEAR)

        df, overview_keys.loc[key, "Status country data"] = prepare_country_data(df, pisa_countries)

        overview_keys.loc[key, "Number of columns"] = len(df.columns)
        overview_keys.loc[key, "Number of entries"] = len(df)

        df.to_csv(file_key, index=False)

        if overview_keys.loc[key, "Status annual data"] == "As required - One column with annual data":
            if overview_keys.loc[key, "Status country data"] == "Country data identified, renamed and reduced":
                try:
                    overview_keys.loc[key, "Status transformation"] = bring_data_to_desired_structure_and_save(df, key, FOLDERNAME_FINAL_DATA)

                except (Exception,RuntimeError) as e:
                    overview_keys.loc[key, "Status transformation"] = "Failed: " + type(e).__name__ + "(" + str(e) + ")"

        overview_keys.to_csv("Overview.csv")



overview = pd.read_csv("Overview.csv", encoding="utf-8")
if "Unnamed: 0" in overview.columns:
    overview = overview.rename(columns={"Unnamed: 0": "key"})
overview = overview.set_index("key")

delete = [x for x in overview.index if x.startswith("FFS_")]
overview = overview.drop(delete)

delete_country = overview[overview["Status country data"] != "Country data identified, renamed and reduced"].index
overview = overview.drop(delete_country)

annual_issues = overview[overview["Status annual data"] != "As required - One column with annual data"]
country_csv = Path("PISA-Daten") / "PISA_Länder.csv"
with open(country_csv, "r", encoding="utf-8-sig") as file:
    pisa_countries = file.read().splitlines()

missing_annual = list(annual_issues[annual_issues["Status annual data"] == "No annual data available"].index)
year_found, missing_annual = check_key_names_for_year(missing_annual, FOLDERNAME_PREP)

overview.loc[year_found, ["Status annual data"]] = "As required - One column with annual data"

missing_annual, annual_status_dict, country_status_dict = check_raw_data_for_upper_year_column_name(missing_annual,
                                                                                                    FOLDERNAME_RAW,
                                                                                                    FOLDERNAME_PREP,
                                                                                                    pisa_countries)

for key in annual_status_dict:
    overview.loc[key, ["Status annual data"]] = annual_status_dict[key]

for key in country_status_dict:
    overview.loc[key, ["Status country data"]] = country_status_dict[key]

missing_annual = overview[overview["Status annual data"] != "As required - One column with annual data"].index
overview = overview.drop(missing_annual)

condition_transformation = overview[overview["Status transformation"] != "Data transformed"]
condition_country = overview[overview["Status country data"] == "Country data identified, renamed and reduced"]
condition_year = overview[overview["Status annual data"] == "As required - One column with annual data"]
ready_for_transformation = list(overview[condition_transformation and condition_country and condition_year].index)

pbar_final = tqdm(ready_for_transformation)
for key in pbar_final:
    try:
        pbar_final.set_description(key)
        file_key = path_prep / f"{key}.csv"
        dataframe = pd.read_csv(file_key, encoding="utf-8")

        dataframe = prepare_special_column_names(dataframe)
        dataframe = delete_unnecessary_columns(dataframe, ["code", "upper", "single", "identical"])
        overview.loc[key, "Status transformation"] = bring_data_to_desired_structure_and_save(dataframe, key, FOLDERNAME_FINAL_DATA)

    except (RuntimeError, Exception) as e:
        overview.loc[key, "Status transformation"] = "failed: " + type(e).__name__ + "(" + str(e) + ")"

    overview.to_csv("Overview.csv")

