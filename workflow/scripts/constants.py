"""
Soft coded centalized `constants`
"""

import numpy as np

SNAKEFILE_CHOICES = ["Snakefile", "snakefile"]

PLOT_COST_UNITS = 1e9  # bnEur
PLOT_CAP_UNITS = 1e3  # MW->GW
PLOT_CAP_LABEL = "GW"
PLOT_SUPPLY_UNITS = 1e6  # MWh->TWh
PLOT_SUPPLY_LABEL = "TWh"
PLOT_CO2_UNITS = 1e6  # tCO2->MtCO2
PLOT_CO2_LABEL = "Mt CO2"
COST_UNIT = 1

CURRENCY = "Eur"

# ==== data inputs ====
YEARBOOK_DATA2POP = 1e4
POP_YEAR = "2020"
TIMEZONE = "Asia/Shanghai"
# THIS is used to heating demand and is a bit of a problem since currently all are set to
# the administrative timezone and not the geo timezoones
REGIONAL_GEO_TIMEZONES = {
    "Anhui": TIMEZONE,
    "Beijing": TIMEZONE,
    "Chongqing": TIMEZONE,
    "Fujian": TIMEZONE,
    "Gansu": TIMEZONE,
    "Guangdong": TIMEZONE,
    "Guangxi": TIMEZONE,
    "Guizhou": TIMEZONE,
    "Hainan": TIMEZONE,
    "Hebei": TIMEZONE,
    "Heilongjiang": TIMEZONE,
    "Henan": TIMEZONE,
    "Hubei": TIMEZONE,
    "Hunan": TIMEZONE,
    "InnerMongolia": TIMEZONE,
    "Jiangsu": TIMEZONE,
    "Jiangxi": TIMEZONE,
    "Jilin": TIMEZONE,
    "Liaoning": TIMEZONE,
    "Ningxia": TIMEZONE,
    "Qinghai": TIMEZONE,
    "Shaanxi": TIMEZONE,
    "Shandong": TIMEZONE,
    "Shanghai": TIMEZONE,
    "Shanxi": TIMEZONE,
    "Sichuan": TIMEZONE,
    "Tianjin": TIMEZONE,
    "Tibet": TIMEZONE,
    "Xinjiang": TIMEZONE,
    "Yunnan": TIMEZONE,
    "Zhejiang": TIMEZONE,
}
INFLOW_DATA_YR = 2016

# TIME RANGE
YEAR_HRS = 8760
REF_YEAR = 2020
START_YEAR = 2020
END_YEAR = 2060

CRS = 4326  # WGS84

# ===== CHINA ======
# 791 TWh extra space heating demand + 286 Twh extra hot water demand
# 60% CHP efficiency 0.468 40% coal boiler efficiency 0.97
# (((791+286) * 0.6 /0.468) + ((791+286) * 0.4 /0.97))  * 0.34 * 1e6 = 0.62 * 1e9
CO2_EL_2020 = 5.288987673 * 1e9  # tCO2
CO2_HEATING_2020 = 0.628275682 * 1e9  # tCO2

NUCLEAR_EXTENDABLE = [
    "Liaoning",
    "Shandong",
    "Jiangsu",
    "Zhejiang",
    "Fujian",
    "Guangdong",
    "Hainan",
    "Guangxi",
]
# FACTORS
LOAD_CONVERSION_FACTOR = 1e6  # convert from  ? to ?
DEFAULT_OFFSHORE_WIND_CORR_FACTOR = 1.0

# ====== Line COSTS =======
# cannot take straightest path due to property and terrain
NON_LIN_PATH_SCALING = 1.25
LINE_SECURITY_MARGIN = 1.45
FOM_LINES = 1.02  # of cap costs
ECON_LIFETIME_LINES = 40  # years

# TODO fix mismatch in ref year, move to config
# ===== YEARLY HEAT DEMAND INCREASE MODEL ======
# In 2008 China 228.4 Twh for urban residential DHW
# MWh/capita/year = 228.4 * 1e6 / 62403/1e4 = 0.366008
UNIT_HOT_WATER_START_YEAR = 0.366008  # MWh/capita/yr 2020 [!! INCONSISTENT]
# We can consider that, on average,
# the 52 M in developed countries is around 1000 kWh per person
# http://www.estif.org/fileadmin/estif/content/publications/downloads/UNEP_2015/factsheet_single_family_houses_v05.pdf
UNIT_HOT_WATER_END_YEAR = 1.0

# ========= names ==========
PROV_NAMES = [
    "Anhui",
    "Beijing",
    "Chongqing",
    "Fujian",
    "Gansu",
    "Guangdong",
    "Guangxi",
    "Guizhou",
    "Hainan",
    "Hebei",
    "Heilongjiang",
    "Henan",
    "Hubei",
    "Hunan",
    "InnerMongolia",
    "Jiangsu",
    "Jiangxi",
    "Jilin",
    "Liaoning",
    "Ningxia",
    "Qinghai",
    "Shaanxi",
    "Shandong",
    "Shanghai",
    "Shanxi",
    "Sichuan",
    "Tianjin",
    "Tibet",
    "Xinjiang",
    "Yunnan",
    "Zhejiang",
]

OFFSHORE_WIND_NODES = np.array(
    [
        "Fujian",
        "Guangdong",
        "Guangxi",
        "Hainan",
        "Hebei",
        "Jiangsu",
        "Liaoning",
        "Shandong",
        "Shanghai",
        "Tianjin",
        "Zhejiang",
    ],
    dtype=str,
)

# ==== technologies

CARRIERS = [
    "coal",
    "CHP coal",
    "CHP gas",
    "OCGT",
    "solar",
    "solar thermal",
    "onwind",
    "offwind",
    "coal boiler",
    "ground heat pump",
    "nuclear",
]

NICE_NAMES_DEFAULT = {
    "solar": "solar PV",
    "Sabatier": "methanation",
    "offwind": "offshore wind",
    "offwind-ac": "offshore wind (AC)",
    "offwind-dc": "offshore wind (DC)",
    "offwind-float": "offshore wind (Float)",
    "onwind": "onshore wind",
    "ror": "hydroelectricity",
    "hydro": "hydroelectricity",
    "PHS": "hydroelectricity",
    "NH3": "ammonia",
    "co2 Store": "DAC",
    "co2 stored": "CO2 sequestration",
    "AC": "transmission lines",
    "DC": "transmission lines",
    "B2B": "transmission lines",
}

# tests
TESTS_RUNNAME = "automated_test_run"
TESTS_CUTOUT = "China-tests-cutout"
