import urllib.parse
from pathlib import Path

import requests as rq


def load_and_save(key, folder, url, file_type):
    """ load given url and save in given directory """
    path = Path(folder) / f"{key}.{file_type}"

    if not path.parent.is_dir():
        path.parent.mkdir()

    if path.is_file():  # if already exists, do nothing
        return

    # use requests library to get data from API using the given url
    data = rq.get(url).content.decode('utf-8-sig')

    with open(path, "w", encoding='utf-8') as file:
        file.write(data)


# just as a help to taking a closer look at the different data tables - some might have aggregated information
# or only include specific data (years/countries)
# finding groups by anticipating similar name tags
# https://stats.oecd.org/Index.aspx?DataSetCode=LAND_COVER_FUA
# https://stats.oecd.org/restsdmx/sdmx.ashx/GetDataStructure/EO43_VINTAGE
def help_analysis(keys):
    """Look at the key names and count letter combinations that appear often"""
    pairs = {}
    triplets = {}
    for key in keys:
        for index, letter in enumerate(key[:-2]):
            next_letter = key[index + 1]
            pair = letter + next_letter
            next_next_letter = key[index + 2]
            triplet = pair + next_next_letter
            pairs[pair] = pairs.setdefault(pair, 0) + 1
            triplets[triplet] = triplets.setdefault(triplet, 0) + 1
    p_analysis = {k: v for k, v in sorted(pairs.items(), reverse=True, key=lambda item: item[1])}
    trip_analysis = {k: v for k, v in sorted(triplets.items(), reverse=True, key=lambda item: item[1])}
    print(p_analysis)
    print(trip_analysis)


# after taking a closer look at different data sets several data sets that aren't relevant have been identified
# create a new list of keys including only the relevant ones we want to download
def remove_keys_starting_with(keys, beginning, exceptions=None):
    """Remove all keys beginning with a certain sting except for exceptions"""
    if not exceptions:
        exceptions = []

    if not isinstance(exceptions, list):
        exceptions = [exceptions]

    return [key for key in keys if not (key.startswith(beginning) and key not in exceptions)]


def remove_keys_ending_with(keys, ending, exceptions=None):
    """Remove all keys beginning with a certain sting except for exceptions"""
    if not exceptions:
        exceptions = []

    if not isinstance(exceptions, list):
        exceptions = [exceptions]

    return [key for key in keys if not (key.endswith(ending) and key not in exceptions)]


def remove_irrelevant_keys(keys):
    """Removes all keys that are considered irrelevant by nina"""
    # ALFS tables are several older versions that are updated by adding columns, other tables have different views of
    # the same data - ALFS_SUMTAB only relevant table
    # ALFS_POP_VITAL is a key that doesn't have anything to do with ALFS and it's old data
    keys = remove_keys_starting_with(keys, "ALFS", "ALFS_SUMTAB")

    # ANBERD Archived Data isn't relevant as all data is aggregated and newer in ANBERD_REV4
    keys = remove_keys_starting_with(keys, "ANBERD", "ANBERD_REV4")

    # all BLI data is mixed over several years - I suggest to deleting all yearly editions and only taking the overall
    keys = remove_keys_starting_with(keys, "BLI", "BLI")

    # Old BTDIXE Data isn't relevant as all data is aggregated and newer in BTDIXE_I4
    keys = remove_keys_starting_with(keys, "BTDIXE", "BTDIXE_I4")

    # BERD, PERS and GERD partly relevant
    keys = remove_keys_starting_with(keys, "BERD", ["BERD_INDU", "BERD_MA_SOF", "BERD_MA_TOE", "BERD_SOF_SIZE"])
    keys = remove_keys_starting_with(keys, "GERD", ["GERD_FORD", "GERD_SEO", "GERD_SOF", "GERD_TOE", "GERD_TORD"])
    keys = remove_keys_starting_with(keys, "PERS", ["PERS_FORD", "PERS_QUALIF", "PERS_FUNC", "PERS_INDU"])

    # IO_GHG_2019 newest version
    keys = remove_keys_starting_with(keys, "IO_GHG", "IO_GHG_2019")

    # Data sets beginning with "REV" or "LAC_REV" include Revenue data of one specific country
    # This data is aggregated in the table "REFSERIES_GL" and RS_GLB
    keys = remove_keys_starting_with(keys, "REV")
    keys = remove_keys_starting_with(keys, "LAC_REV")
    keys = remove_keys_starting_with(keys, "REFSERIES_", "REFSERIES_GL")
    keys = remove_keys_starting_with(keys, "RS_", "RS_GBL")

    # EO data is economic outlook of different years
    # it is aggregated in "EO" but only one of the differnet scenarios is used
    # EO_EDITIONS - includes the data of all previous years and both types of "calculation"
    keys = remove_keys_starting_with(keys, "EO", "EO_EDITIONS")

    # FDI a lot of financial data - but extremely detailed. Only take Summary
    keys = remove_keys_starting_with(keys, "FDI", ["FDI_AGGR_SUMM", "FDI_CTRY_IND_SUMM", "FDIINDEX", ])

    # GOV is the overview of the tables that are yearly added
    keys = remove_keys_starting_with(keys, "GOV_")

    # all old EAG data
    keys = remove_keys_starting_with(keys, "CHAPTER_", )

    # MON tables are created for each year including the information of all previous years
    # The total overview of the data can be found in MON_REFERENCE_TABLE and MON_SINGLE_COMMODITY_INDICATORS
    keys = remove_keys_starting_with(keys, "MON", ["MON_REFERENCE_TABLE", "MON_SINGLE_COMMODITY_INDICATORS"])

    # Except for the Years 2019 and 2020 no Year provides actual historical data - only a forecast 2019 includes the
    # actual data from 1990 - 2017 (and an estimate from 2018 onwards), and 2020 from 2010 until 2019 (and an
    # estimate from 2019 onwards)
    keys = remove_keys_starting_with(keys, "HIGH_AGLINK_", ["HIGH_AGLINK_2019", "HIGH_AGLINK_2020"])

    # NAAG_YEAR data only contains data up to the given year
    # NAAG has an overview of the same data of all years and countries from 2000 until 2019
    keys = remove_keys_starting_with(keys, "NAAG_")

    # Subnational information not relevant
    keys = remove_keys_ending_with(keys, "_SUBNAT")

    # SNA Tables are very detailed and partly old - only summary in Table 1-4
    keys = remove_keys_starting_with(keys, "SNA", ["SNA_TABLE1", "SNA_TABLE2", "SNA_TABLE3", "SNA_TABLE4"])
    keys = remove_keys_ending_with(keys, "_SNA93")
    keys = remove_keys_ending_with(keys, "_ARCHIVE")  # Archive data is newer good - there is newer one!
    keys = remove_keys_starting_with(keys, "QNA")  # Quarterly data, not annual

    # many many archived STAN tables - newest data (including historical data) in STANI4-2020
    keys = remove_keys_starting_with(keys, "STAN", "STANI4_2020")

    # TSEC - old data - newer data in TEC_REV4, TSE newer data MON
    keys = remove_keys_starting_with(keys, "TSE")
    keys = remove_keys_starting_with(keys, "PSE")
    keys = remove_keys_starting_with(keys, "CPSE")
    keys = remove_keys_starting_with(keys, "CSE")
    keys = remove_keys_starting_with(keys, "GSSE")
    keys = remove_keys_starting_with(keys, "OECD_TSE")

    keys = remove_keys_starting_with(keys, "TIVA", ["TIVA_2018_C1",
                                                    "TIVA_2018_C2",
                                                    "TIVA_2018_C3",
                                                    "TIVA_2018_C4",
                                                    "TIVA_20 18_C5"])

    # AFA all different data
    # AMNE like AFA
    # CBCR all relevant
    # DAC relevant
    # DIOC relevant
    # ITF is fine
    # EPL all relevant
    # TEC(NUM)_REV4 are all different values - don't delete
    # TRADEENV_IND(NUM) are all different values - don't delete
    # SNA tables are all different - don't delete
    # EAG are all different and very important - these are the educational tables
    # EDU same as EAG
    # TRADEENV fine

    # Archived Educational data - used elsewhere:
    # RPERS, RFIN1, RFINI2, RFOREIGN, RGRADAGE, RGRADSTY, RNENTEAGE, ROVERAGE, RENLARGE, RENRL, RPOP
    # POP_FIVE_HIST and POP_PROJ are old tables that aren't updated newer versions are in HISTPOP and POPPROJ
    # Newer version and overview of TIM in TIM_MAIN_2019
    # IOTS is an older version - IOTSI4-2018 is newer
    # Archived Tables included in the newest TIVA data:
    # SITC_SECTION, PARTNER, REFSERIES_MSIT, CALCULATED, INDICE_OECD
    # BTD_ED_2010 - also summarised in BTDIXE
    # TES3 is a typo - should be TSE3
    # Archived data Payments
    # Archive Research and Development GBAORD_NABS2007, RD_ACTIVITY_PRE1981, RD_ACTIVITY, ONRD_FUNDS, ONRD_COST
    remove = ["POP_FIVE_HIST", "POP_PROJ", "TIM2015_C1", "IOTS", "SITC_SECTION",
              "PARTNER", "REFSERIES_MSIT", "CALCULATED", "INDICE_OECD", "BTD_ED_2010",
              "RPERS", "RFIN1", "RFINI2", "RFOREIGN", "RGRADAGE", "RGRADSTY", "RNENTEAGE",
              "ROVERAGE", "RENLARGE", "RENRL", "RPOP", "TES3", "MEI_BOP",
              "GBAORD_NABS2007", "RD_ACTIVITY_PRE1981", "RD_ACTIVITY", "ONRD_FUNDS", "ONRD_COST", "TABLE2A", "BIMTS_CPA"]

    return [x for x in keys if x not in remove]


def oecd_url_by_key(key, startyear, endyear):
    """create url for the API given a key, a start year and an end year"""
    parameters = {"startTime": startyear,
                  "endTime": endyear,
                  "contentType": "csv"}

    return f"https://stats.oecd.org/SDMX-JSON/data/{key}/all/all?{urllib.parse.urlencode(parameters)}"


def oecd_data_load_and_save(key, startyear, endyear, foldername):
    """loads the fuction creating the url and the downloading function"""
    url = oecd_url_by_key(key, startyear, endyear)

    load_and_save(key, foldername, url, "csv")
