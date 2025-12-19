"""File reading support functions for PyPSA-China-PIK workflow.

This module provides functions for reading and processing yearly load projections
from REMIND data, with support for sector coupling (electric vehicles) and
flexible data format handling.
"""

import os

import pandas as pd


def aggregate_sectoral_loads(yearly_proj: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Aggregate REMIND load sectors according to the model configuration.

    Sectors that are NOT enabled for independent modeling will be aggregated
    into the main electricity load. For example, if EV sector is not enabled
    as an independent sector (enabled: false), its load will be added to the
    AC load in the aggregation.

    Args:
        yearly_proj: REMIND output with columns ['province', 'sector', '2020', '2025', ...].
            Each row represents one sector's load for one province across all years.
        config: Configuration dict with structure:
            - sectors.electric_vehicles.enabled: bool (whether to model EV as independent sector)
            - sectors.sector_mapping.base: list (always-included sectors, e.g., ['ac'])
            - sectors.sector_mapping.electric_vehicles: list (EV sectors, e.g., ['ev_pass', 'ev_freight'])

    Returns:
        DataFrame with provinces as index and years as columns, containing total annual
        load (MWh) summed across sectors that should be aggregated.

    Raises:
        ValueError: If sector_mapping is missing or no matching sectors found.
    """
    sectors_cfg = config.get("sectors", {})
    mapping = sectors_cfg.get("sector_mapping", {})

    if not mapping:
        raise ValueError("Missing sector_mapping configuration")

    # Always include base sectors (e.g., AC)
    sectors_to_include = set(mapping.get("base", []))

    # For each optional sector, if it's NOT enabled for independent modeling,
    # aggregate it into the main load
    # Electric vehicles
    if not sectors_cfg.get("electric_vehicles", {}).get("enabled", False):
        if "electric_vehicles" in mapping:
            sectors_to_include.update(mapping["electric_vehicles"])

    # Heat coupling (if exists in the future)
    if not sectors_cfg.get("heat_coupling", {}).get("enabled", False):
        if "heat_coupling" in mapping:
            sectors_to_include.update(mapping["heat_coupling"])

    # Filter data to only include selected sectors
    filtered = yearly_proj[yearly_proj["sector"].isin(sectors_to_include)].copy()
    if filtered.empty:
        raise ValueError(
            f"No sector data found. "
            f"Requested sectors: {sectors_to_include}, "
            f"Available sectors: {yearly_proj['sector'].unique().tolist()}"
        )

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
            f"No province column found in {file_path}. Expected one of: {province_candidates}"
        )

    if province_col != "province":
        df = df.rename(columns={province_col: "province"})

    # Process data based on whether it contains sector information
    if "sector" in df.columns:
        if config is None:
            raise ValueError(
                "Data file contains sector column but no config provided. "
                "Please provide config with 'sectors' and 'sector_mapping' keys."
            )
        df = aggregate_sectoral_loads(df, config)
    else:
        # Simple data format - set province as index
        df = df.set_index("province")

    # Convert year columns to integers for consistency
    year_cols = {col: int(col) for col in df.columns if col.isdigit()}
    df = df.rename(columns=year_cols)

    # Apply conversion factor
    return df * conversion
