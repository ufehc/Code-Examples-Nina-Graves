import country_converter as coco

from functions_data_prep_overall import *

converter = coco.CountryConverter()


def identify_potential_country_columns_and_translations(dataframe, potential_country_list=None):
    """goes through all columns in the list of columns given or if none given all columns and checks if there are
    country names given in the column"""
    if not potential_country_list:
        potential_country_list = [x for x in dataframe.columns if x != "value" and x != "year"]

    if not isinstance(potential_country_list, list):
        potential_country_list = [potential_country_list]

    translations = {}

    for col in potential_country_list:
        values = list(dataframe[col].unique())
        conversions = converter.convert(names=values, to="ISO3")

        conversions = [str(x) for x in conversions]

        if "['CHN', 'MAC']" in conversions:
            conversions = [x if x != "['CHN', 'MAC']" else "MAC" for x in conversions]

        not_countries = conversions.count("not found") + conversions.count("n")

        if not_countries <= len(values) / 2:
            translation_countries = dict(zip(values, conversions))
            translations[col] = translation_countries

    if len(translations) == 1:
        country_status = "Country column identified"

    if len(translations) == 0:
        country_status = "No potential country column identified"

    if len(translations) >= 2:
        country_status = "Several potential country columns found"

    return translations, country_status


def translate_country_column(dataframe, country_list, translation):
    """transform the country names in the country column to ISO3, delete irrelevant countries, rename column"""
    column_name_list = list(translation.keys())
    column_name = column_name_list[0]
    translation_countries = translation[column_name]

    dataframe[column_name] = dataframe[column_name].map(translation_countries)

    dataframe_new = dataframe[dataframe[column_name].isin(country_list)]

    dataframe_new = dataframe_new.rename({column_name: "country"}, axis=1)

    return dataframe_new


def prepare_country_data(dataframe, country_list):
    """analyse dataframe and transform country data if possible. returns dataframe and a status on country data"""
    dataframe_new = dataframe.copy()
    potential_country_columns_names = ["country", "declaring country", "reporter country", "donor",
                                       "reporting country", "economy", "reporter", "location"]
    country_columns_names = [x for x in potential_country_columns_names if x in dataframe_new.columns]

    if country_columns_names:
        translation, status = identify_potential_country_columns_and_translations(dataframe_new, country_columns_names)
    elif "Territory Level and Typology" in dataframe_new.columns:
        dataframe_new["Territory Level and Typology"] = dataframe_new["Territory Level and Typology"].str.lower()
        if "country" in dataframe_new["Territory Level and Typology"].unique():
            dataframe_new = dataframe_new.loc[dataframe_new["Territory Level and Typology"] == "country", :]
            dataframe_new = delete_entire_columns(dataframe_new, "Territory Level and Typology")
            translation, status = identify_potential_country_columns_and_translations(dataframe_new)
    else:
        translation, status = identify_potential_country_columns_and_translations(dataframe_new)

    if status == "Country column identified":
        dataframe_new = translate_country_column(dataframe_new, country_list, translation)
        status = "Country data identified, renamed and reduced"
    else:
        status = f"{status} and translations: {translation}"

    return dataframe_new, status
