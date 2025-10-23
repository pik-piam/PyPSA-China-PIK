"""File reading support functions for PyPSA-China-PIK workflow.

This module provides functions for reading and processing yearly load projections
from REMIND data, with support for sector coupling (electric vehicles) and 
flexible data format handling.
"""

import os

import pandas as pd


def merge_sectors_by_config(yearly_proj: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Merge sectors based on configuration settings.
    
    Processes REMIND data by combining sectors according to the configuration.
    Supports electric vehicle sector coupling and flexible sector mapping.
    
    Args:
        yearly_proj (pd.DataFrame): Raw yearly projections data with sector column
        config (dict): Configuration dictionary containing:
            - sectors: dict of sector flags (e.g., {"electric_vehicles": True})
            - sector_mapping: dict mapping sector keys to data columns
            - paths: dict with data file paths
            - run.remind: dict with REMIND region settings
    
    Returns:
        pd.DataFrame: Processed data aggregated by province and year
        
    Raises:
        ValueError: If required configuration is missing or no matching sectors found
    """
    sectors_cfg = config.get("sectors", {})
    mapping = config.get("sector_mapping", {})

    if not sectors_cfg or not mapping:
        raise ValueError("Missing sectors or sector_mapping configuration")

    # Get base sectors that are always included
    sectors_to_include = set(mapping.get("base", []))

    # Add sectors based on configuration flags
    for sector_key, is_enabled in sectors_cfg.items():
        if is_enabled and sector_key in mapping:
            mapped_sectors = mapping.get(sector_key, [])
            sectors_to_include.update(mapped_sectors)

    # Filter data to only include selected sectors
    filtered = yearly_proj[yearly_proj["sector"].isin(sectors_to_include)].copy()
    if filtered.empty:
        raise ValueError(f"No sector data found for merging. Available sectors: {sectors_to_include}")

    # Aggregate by province and year
    year_cols = [c for c in filtered.columns if c.isdigit()]
    result = filtered.groupby("province")[year_cols].sum()
    return result


def read_yearly_load_projections(
    file_path: os.PathLike = "resources/data/load/Province_Load_2020_2060.csv",
    conversion: float = 1.0,
    config: dict = None,
) -> pd.DataFrame:
    """Read and process yearly load projections from CSV files.
    
    Supports both simple load data and REMIND sector-coupled data with 
    electric vehicle integration. Automatically detects data format and
    applies appropriate processing.
    
    Args:
        file_path (os.PathLike): Path to the yearly projections CSV file.
            Defaults to "resources/data/load/Province_Load_2020_2060.csv".
        conversion (float): Conversion factor to apply to the data (e.g., to MWh).
            Defaults to 1.0.
        config (dict, optional): Configuration dictionary for sector processing.
            Required when processing REMIND data with sector columns.
            Should contain 'sectors' and 'sector_mapping' keys.
    
    Returns:
        pd.DataFrame: Processed load projections data with:
            - Province names as index (for simple data) or columns
            - Year columns as integers
            - Data converted by the conversion factor
    
    Raises:
        ValueError: If required columns are missing or configuration is invalid
        FileNotFoundError: If the input file does not exist
    
    Examples:
        >>> # Simple load data
        >>> data = read_yearly_load_projections("simple_load.csv")
        
        >>> # REMIND data with electric vehicles
        >>> config = {
        ...     "sectors": {"electric_vehicles": True},
        ...     "sector_mapping": {
        ...         "base": ["ac"],
        ...         "electric_vehicles": ["ev_freight", "ev_pass"]
        ...     }
        ... }
        >>> data = read_yearly_load_projections("remind_data.csv", config=config)
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Standardize province column name
    province_candidates = ["province", "region", "Unnamed: 0"]
    province_col = next((col for col in province_candidates if col in df.columns), None)

    if province_col is None:
        raise ValueError(
            f"No province column found in {file_path}. "
            f"Expected one of: {province_candidates}"
        )

    if province_col != "province":
        df = df.rename(columns={province_col: "province"})

    # Process data based on whether it contains sector information
    if "sector" in df.columns:
        if config is None:
            raise ValueError(
                "REMIND data contains sector column but no config provided. "
                "Please provide config with 'sectors' and 'sector_mapping' keys."
            )
        df = merge_sectors_by_config(df, config)
    else:
        # Simple data format - set province as index
        df = df.set_index("province")

    # Convert year columns to integers for consistency
    year_cols = {col: int(col) for col in df.columns if col.isdigit()}
    df = df.rename(columns=year_cols)

    # Apply conversion factor
    return df * conversion
