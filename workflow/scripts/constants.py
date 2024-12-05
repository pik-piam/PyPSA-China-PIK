import numpy as np

SNAKEFILE_CHOICES = ["Snakefile", "snakefile"]

PLOT_COST_UNITS = 1e9  # bnEur
PLOT_CAP_UNITS = 1e3  # MW->GW
PLOT_SUPPLY_UNITS = 1e6  # MWh->TWh
COST_UNIT = 1

# ==== data inputs ====
YEARBOOK_DATA2POP = 1e4
POP_YEAR = "2020"
TIMEZONE = "Asia/Shanghai"
INFLOW_DATA_YR = 2016

# TIME RANGE
YEAR_HRS = 8760
REF_YEAR = 2020
START_YEAR = 2020
END_YEAR = 2060

CRS = 4326  # WGS84

# ===== CHINA ======
CO2_EL_2020 = 5.288987673 * 1e9  # tCO2 # TODO verify unit
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

# ====== HEATING LINEAR MODEL (atlite cutout) ========
HEATING_START_TEMP = 15.0  # C
HEATING_HOUR_SHIFT = 8.0  # hrs, accounts for timezone
HEATING_LIN_SLOPE = 1  # slope
HEATING_OFFET = 0  # linear model offset

# ===== YEARLY HEAT DEMAND INCREASE MODEL ======
# In 2008 China 228.4 Twh for urban residential DHW
# MWh/capita/year = 228.4 * 1e6 / 62403/1e4 = 0.366008
UNIT_HOT_WATER_START_YEAR = 0.366008  # MWh/capita/yr 202
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
