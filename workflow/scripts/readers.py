"""file reading support functions"""

import os

import pandas as pd


import os
import pandas as pd

def load_h2_conversion_efficiency(remind_output_dir: str, region: str = "CHA", year: int = None) -> float:
    """Load H2 conversion efficiency (elh2) from REMIND output files"""
    eta_file = os.path.join(remind_output_dir, "pm_eta_conv.csv")
    if not os.path.exists(eta_file):
        raise FileNotFoundError(f"Efficiency file not found: {eta_file}")
    
    eta_df = pd.read_csv(eta_file)
    h2_eta = eta_df.query("all_te == 'elh2' and all_regi == @region")
    if h2_eta.empty:
        raise ValueError(f"H2 conversion efficiency (elh2) not found for region {region}")
    
    if year is not None:
        if 'ttot' in h2_eta.columns:
            year_eta = h2_eta.query("ttot == @year")
            if not year_eta.empty:
                return float(year_eta["value"].iloc[0])
    
    return float(h2_eta["value"].iloc[0])


def merge_sectors_by_config(yearly_proj: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Merge sectors by configuration and convert H2 to electricity demand when disabled"""
    sectors_cfg = config.get("sectors", {})
    mapping = config.get("sector_mapping", {})
    
    if not sectors_cfg or not mapping:
        raise ValueError("Missing sectors or sector_mapping configuration")

    data_sectors = set(yearly_proj["sector"].unique())
    available_sectors = []
    for k in sectors_cfg.keys():
        if k in mapping:
            mapping_sectors = mapping.get(k, [])
            if any(s in data_sectors for s in mapping_sectors):
                available_sectors.append(k)
    
    available_values = [sectors_cfg[k] for k in available_sectors]
    
    if all(available_values):
        sectors = mapping.get("base", [])
    elif not any(available_values):
        sectors = mapping.get("base", [])
        for k, active in sectors_cfg.items():
            if not active and k in mapping:
                mapping_sectors = mapping.get(k, [])
                sectors += [s for s in mapping_sectors if s in data_sectors]
    else:
        sectors = mapping.get("base", [])
        for k, active in sectors_cfg.items():
            if not active and k in mapping:
                mapping_sectors = mapping.get(k, [])
                sectors += [s for s in mapping_sectors if s in data_sectors]

    merged = yearly_proj[yearly_proj["sector"].isin(sectors)].copy()
    if merged.empty:
        raise ValueError(f"No sector data found for merging: {sectors}")

    h2_sectors = mapping.get("add_H2", [])
    if not sectors_cfg.get("add_H2", False) and any(s in merged["sector"].values for s in h2_sectors):
        remind_dir = config["paths"]["remind_outpt_dir"]
        region = config["run"]["remind"]["region"]
        year_cols = [c for c in merged.columns if c.isdigit()]
        
        for s in h2_sectors:
            if s in merged["sector"].values:
                for year_col in year_cols:
                    year = int(year_col)
                    eta = load_h2_conversion_efficiency(remind_dir, region, year)
                    merged.loc[merged["sector"] == s, year_col] /= eta

    year_cols = [c for c in merged.columns if c.isdigit()]
    result = merged.groupby("province")[year_cols].sum()
    return result


def read_yearly_load_projections(
    yearly_projections_p: os.PathLike = "resources/data/load/Province_Load_2020_2060.csv",
    conversion=1,
    config: dict = None,
) -> pd.DataFrame:
    """Prepare projections for model use

    Args:
        yearly_projections_p (os.PathLike, optional): the data path.
                Defaults to "resources/data/load/Province_Load_2020_2060.csv".
        conversion (int, optional): the conversion factor to MWh. Defaults to 1.
        config (dict, optional): configuration dictionary for sector merging. Defaults to None.

    Returns:
        pd.DataFrame: the formatted data, in MWh
    """
    yearly_proj = pd.read_csv(yearly_projections_p)
    yearly_proj.rename(columns={"Unnamed: 0": "province", "region": "province"}, inplace=True)
    
    if "province" not in yearly_proj.columns:
        raise ValueError(
            "The province (or region or unamed) column is missing in the yearly projections data"
            ". Index cannot be built"
        )
    
    if "sector" in yearly_proj.columns:
        if config is None:
            raise ValueError("Config is required when processing REMIND data with sector column")
        yearly_proj = merge_sectors_by_config(yearly_proj, config)
    else:
        yearly_proj.set_index("province", inplace=True)
    
    yearly_proj.rename(columns={c: int(c) for c in yearly_proj.columns if c.isdigit()}, inplace=True)

    return yearly_proj * conversion
