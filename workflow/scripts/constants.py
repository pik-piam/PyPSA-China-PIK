import numpy as np

# data inputs
CRS = 4326  # WGS84
YEARBOOK_DATA2POP = 1e4
POP_YEAR = 2020

# TIME RANGE
START_YEAR = 2020
END_YEAR = 2060

# RESOLUTION
FREQ = "1H"
# FACTORS
DEFAULT_OFFSHORE_WIND_CORR_FACTOR = 1.0
# ====== HEATING LINEAR MODEL (atlite cutout) ========
HEATING_START_TEMP = 15.0  # C
HEATING_HOUR_SHIFT = 8.0  # hrs, accounts for timezone
HEATING_LIN_SLOPE = 1  # slope
HEATING_OFFET = 0  # linear model offset
# ===== YEARLY DEMAND INCREASE MODEL ======
# In 2008 China 228.4 Twh for urban residential DHW
# MWh/capita/year = 228.4 * 1e6 / 62403/1e4 = 0.366008
UNIT_HOT_WATER_START_YEAR = 0.366008  # MWh/capita/yr 202
# We can consider that, on average,
# the 52 M in developed countries is around 1000 kWh per person
# http://www.estif.org/fileadmin/estif/content/publications/downloads/UNEP_2015/factsheet_single_family_houses_v05.pdf
UNIT_HOT_WATER_END_YEAR = 1.0


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