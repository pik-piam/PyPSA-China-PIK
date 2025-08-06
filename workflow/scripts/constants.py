"""
Soft coded centalized `constants`
"""

import os
import re
import logging
import pandas as pd
import yaml
from functools import lru_cache
from typing import List

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

CURRENCY = "Eur"  # 2015

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

# Cache for province names to avoid repeated file reads
_province_names_cache = None

def load_province_config(config_path: str = None) -> dict:
    """Load province configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, uses default path.
        
    Returns:
        dict: Province configuration including names and splits
    """
    if config_path is None:
        config_path = "config/provinces.yaml"
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded province config from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load province config from {config_path}: {e}")
    
    # Default configuration
    # TODO: Implement province splits
    return {
        "provinces": list(REGIONAL_GEO_TIMEZONES_DEFAULT.keys()),
        "splits": {
            # example
            "Inner Mongolia": ["Inner Mongolia East", "Inner Mongolia West"]
        }
    }

def generate_province_names(config: dict = None) -> List[str]:
    """Generate province names from configuration, including splits.
    
    Args:
        config: Province configuration dict. If None, loads from default.
        
    Returns:
        List[str]: List of province names including splits
    """
    if config is None:
        config = load_province_config()
    
    provinces = config.get("provinces", list(REGIONAL_GEO_TIMEZONES_DEFAULT.keys()))
    splits = config.get("splits", {})
    
    # TODO: Implement province splits
    final_provinces = []
    for province in provinces:
        if province in splits:
            final_provinces.extend(splits[province])
        else:
            final_provinces.append(province)
    
    logger.info(f"Generated {len(final_provinces)} province names (including splits)")
    return final_provinces

@lru_cache(maxsize=1)
def get_province_names_cached() -> List[str]:
    """Cached version of get_province_names to avoid repeated calls.
    
    Returns:
        List[str]: List of province names
    """
    # Track function calls
    call_count = getattr(get_province_names_cached, '_call_count', 0) + 1
    get_province_names_cached._call_count = call_count
    
    # Log call count to file
    with open('get_province_names_calls.txt', 'a') as f:
        f.write(f"Call #{call_count} at {pd.Timestamp.now()}\n")
    
    # Try environment variable first (for testing)
    env_provs = os.getenv("PROV_NAMES")
    if env_provs:
        if isinstance(env_provs, str):
            provinces = re.findall(r"[\w']+", env_provs)
            if provinces:
                logger.info(f"Using {len(provinces)} provinces from PROV_NAMES env var")
                return provinces
    
    # Try CSV file (for configuration)
    csv_path = "resources/data/regions/province_codes.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if "Full name" in df.columns:
                provinces = df["Full name"].dropna().unique().tolist()
                if provinces:
                    logger.info(f"Using {len(provinces)} provinces from {csv_path}")
                    return provinces
        except Exception as e:
            logger.warning(f"Failed to read {csv_path}: {e}")
    
    # Default fallback
    default_provinces = list(REGIONAL_GEO_TIMEZONES_DEFAULT.keys())
    logger.info(f"Using default province list with {len(default_provinces)} provinces")
    return default_provinces

def get_province_names() -> List[str]:
    """Get province names for network construction.
    
    Priority: 1. Environment variable (HACK for testing)
              2. CSV file (for configuration)
              3. Default list (fallback)
    """
    return get_province_names_cached()


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
