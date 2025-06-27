"""
Soft coded centalized `constants`
"""

import os
import re
import pandas as pd
from typing import List
import logging

logger = logging.getLogger(__name__)

# ======= CONVERSIONS =======
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
# TODO move to config
YEARBOOK_DATA2POP = 1e4
POP_YEAR = "2020"

# ========= SETUP REGIONS ==========
# problem section due to pytests and snakemake not integrating well (snakmekae is as subprocess)

TIMEZONE = "Asia/Shanghai"
# THIS is used to heating demand and is a bit of a problem since currently all are set to
# the administrative timezone and not the geo timezoones

# Default province names for network construction
REGIONAL_GEO_TIMEZONES_DEFAULT = {
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


def get_province_names() -> List[str]:
    """
    Get the list of province names for network construction.
    
    This function supports multiple ways to specify province names:
    1. Environment variable PROV_NAMES (for testing)
    2. Province codes CSV file
    3. Default province list
    
    Returns:
        List[str]: List of province names to build the network
        
    Raises:
        ValueError: If PROV_NAMES environment variable has invalid format
        
    Example:
        >>> get_province_names()
        ['Beijing', 'Shanghai', 'Guangdong', ...]
        
        # For testing with specific provinces:
        >>> os.environ['PROV_NAMES'] = '["Beijing", "Shanghai"]'
        >>> get_province_names()
        ['Beijing', 'Shanghai']
    """
    default_province_names = list(REGIONAL_GEO_TIMEZONES_DEFAULT.keys())
    
    # Try environment variable first (for testing)
    env_provinces = _get_provinces_from_environment()
    if env_provinces is not None:
        return env_provinces
    
    # Try CSV file
    csv_provinces = _get_provinces_from_csv()
    if csv_provinces is not None:
        return csv_provinces
    
    # Use default province list
    logger.info(f"Using default province list with {len(default_province_names)} provinces")
    return default_province_names


def _get_provinces_from_environment() -> List[str]:
    """
    Get province names from PROV_NAMES environment variable.
    
    Returns:
        List[str]: Province names from environment variable, or None if not set
    """
    env_value = os.getenv("PROV_NAMES")
    if env_value is None:
        return None
    
    if isinstance(env_value, str):
        # Parse string format like '["Beijing", "Shanghai"]'
        provinces = re.findall(r'[\w\']+', env_value)
        if not provinces:
            raise ValueError(
                f"PROV_NAMES environment variable '{env_value}' has invalid format. "
                "Expected format: '[\"region1\", \"region2\", ...]'"
            )
        logger.info(f"Using {len(provinces)} provinces from PROV_NAMES environment variable")
        return provinces
    
    if isinstance(env_value, list):
        logger.info(f"Using {len(env_value)} provinces from PROV_NAMES environment variable")
        return env_value
    
    raise ValueError("PROV_NAMES environment variable must be a string or list")


def _get_provinces_from_csv() -> List[str]:
    """
    Get province names from province_codes.csv file.
    
    Returns:
        List[str]: Province names from CSV file, or None if file not found or invalid
    """
    csv_path = "resources/data/regions/province_codes.csv"
    
    if not os.path.exists(csv_path):
        logger.debug(f"Province codes file not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        if "Full name" not in df.columns:
            logger.warning(f"CSV file {csv_path} does not contain 'Full name' column")
            return None
        
        provinces = df["Full name"].dropna().unique().tolist()
        
        if not provinces:
            logger.warning(f"No valid province names found in {csv_path}")
            return None
        
        logger.info(f"Using {len(provinces)} provinces from {csv_path}")
        return provinces
        
    except Exception as e:
        logger.warning(f"Failed to read province codes from {csv_path}: {e}")
        return None


def filter_buses(names) -> list:
    return [name for name in names if name in PROV_NAMES]


PROV_NAMES = get_province_names()
REGIONAL_GEO_TIMEZONES = {
    k: v for k, v in REGIONAL_GEO_TIMEZONES_DEFAULT.items() if k in PROV_NAMES
}

NUCLEAR_EXTENDABLE_DEFAULT = [
    "Liaoning",
    "Shandong",
    "Jiangsu",
    "Zhejiang",
    "Fujian",
    "Guangdong",
    "Hainan",
    "Guangxi",
]
NUCLEAR_EXTENDABLE = filter_buses(NUCLEAR_EXTENDABLE_DEFAULT)

OFFSHORE_WIND_NODES_DEFAULT = [
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
]
OFFSHORE_WIND_NODES = filter_buses(OFFSHORE_WIND_NODES_DEFAULT)

# TIMES
INFLOW_DATA_YR = 2016

# TIME RANGE
YEAR_HRS = 8760
REF_YEAR = 2020
START_YEAR = 2020
END_YEAR = 2060

# geographical
CRS = 4326  # WGS84
COUNTRY_NAME = "China"
COUNTRY_ISO = "CN"
EEZ_PREFIX = "chinese"

# ===== CHINA ======
# 791 TWh extra space heating demand + 286 Twh extra hot water demand
# 60% CHP efficiency 0.468 40% coal boiler efficiency 0.97
# (((791+286) * 0.6 /0.468) + ((791+286) * 0.4 /0.97))  * 0.34 * 1e6 = 0.62 * 1e9
CO2_EL_2020 = 5.288987673 * 1e9  # tCO2
CO2_HEATING_2020 = 0.628275682 * 1e9  # tCO2
CO2_BASEYEAR_EM = CO2_EL_2020 + CO2_HEATING_2020  # tCO2

# FACTORS
LOAD_CONVERSION_FACTOR = 1
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

# TODO soft-code based on the eez shapefile


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
