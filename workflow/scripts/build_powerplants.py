# SPDX-FileCopyrightText: : 2025 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT
"""
Build the existing capacities for each node from GEM (global energy monitor) tracker data.
This script is intended for use as part of the Snakemake workflow.

The GEM data has to be downloaded manually and placed in the source directory of the snakemake rule.
download page: https://globalenergymonitor.org/projects/global-integrated-power-tracker/download-data/

Nodes can be assigned to specific GEM IDs based on their GPS location or administrative region location.

"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
from shapely.geometry import Point

from _helpers import mock_snakemake, configure_logging

logger = logging.getLogger(__name__)

ADM_COLS = {
    0: "Country",
    1: "Subnational unit (state, province)",
    2: "Major area (prefecture, district)",
    3: "Local area (taluk, county)",
}
ADM_LVL1, ADM_LVL2 = ADM_COLS[1], ADM_COLS[2]


def load_gem_excel(
    path: os.PathLike, sheetname="Units", country_col="Country/area", country_names=["China"]
) -> pd.DataFrame:
    """
    Load a Global Energy monitor excel file as a dataframe.

    Args:
        path (os.PathLike): Path to the Excel file.
        sheetname (str): Name of the sheet to load. Default is "Units".
        country_col (str): Column name for country names. Default is "Country/area".
        country_names (list): List of country names to filter by. Default is ["China"].
    """

    df = pd.read_excel(path, sheet_name=sheetname, engine="openpyxl")
    # replace problem characters in column names
    df.columns = df.columns.str.replace("/", "_")
    country_col = country_col.replace("/", "_")

    if not country_col in df.columns:
        logger.warning(f"Column {country_col} not found in {path}. Returning unfiltered DataFrame.")
        return df

    return df.query(f"{country_col} in @country_names")


def clean_gem_data(gem_data: pd.DataFrame, gem_cfg: dict) -> pd.DataFrame:
    """
    Clean the GEM data by
     - mapping GEM types onto pypsa types
     - filtering for relevant project statuses
     - cleaning invalid entries (e.g "not found"->nan)

    Args:
        gem_data (pd.DataFrame): GEM dataset.
        gem_cfg (dict): Configuration dictionary, 'global_energy_monitor.yaml'
    Returns:
        pd.DataFrame: Cleaned GEM data."""

    valid_project_states = gem_cfg["status"]
    GEM = gem_data.query("Status in @valid_project_states")
    GEM.rename(columns={"Plant _ Project name": "Plant name"}, inplace=True)
    GEM.loc[:, "Retired year"] = GEM["Retired year"].replace("not found", np.nan)
    GEM.loc[:, "Start year"] = GEM["Start year"].replace("not found", np.nan)
    GEM = GEM[gem_cfg["relevant_columns"]]

    # Remove whitespace from admin columns
    # Remove all whitespace (including tabs, newlines) from admin columns
    admin_cols = [col for col in ADM_COLS.values() if col in GEM.columns]
    GEM[admin_cols] = GEM[admin_cols].apply(lambda x: x.str.replace(r"\s+", "", regex=True))

    # split oil and gas, rename bioenergy
    gas_mask = GEM.query("Type == 'oil/gas' & Fuel.str.contains('gas', case=False, na=False)").index
    GEM.loc[gas_mask, "Type"] = "gas"
    GEM.Type = GEM.Type.str.replace("bioenergy", "biomass")

    # split CHP (potential issue: split before type split. After would be better)
    if gem_cfg["CHP"].get("split", False):
        GEM.loc[:, "CHP"] = GEM.loc[:, "CHP"].map({"yes": True}).fillna(False)
        chp_mask = GEM[GEM["CHP"] == True].index

        aliases = gem_cfg["CHP"].get("aliases", [])
        for alias in aliases:
            chp_mask = chp_mask.append(
                GEM[GEM["Plant name"].str.contains(alias, case=False, na=False)].index
            )
        chp_mask = chp_mask.unique()
        GEM.loc[chp_mask, "Type"] = "CHP " + GEM.loc[chp_mask, "Type"]

    GEM["tech"] = ""
    for tech, mapping in gem_cfg["tech_map"].items():
        if not isinstance(mapping, dict):
            raise ValueError(
                f"Mapping for {tech} is a {type(mapping)} - expected dict. Check your config."
            )

        tech_mask = GEM.query(f"Type == '{tech}'").index
        if tech_mask.empty:
            continue
        GEM.loc[tech_mask, "Type"] = GEM.loc[tech_mask, "Technology"].map(mapping)

        # apply defaults if requested
        if "default" not in mapping:
            continue
        fill_val = mapping["default"]
        if fill_val is not None:
            GEM.loc[tech_mask, "Type"] = GEM.loc[tech_mask, "Type"].fillna(value=fill_val)
        else:
            GEM.loc[tech_mask, "Type"] = GEM.loc[tech_mask, "Type"].dropna()

    return GEM.dropna(subset=["Type"])


def group_by_year(df: pd.DataFrame, year_bins: list, base_year=2020) -> pd.DataFrame:
    """
    Group the DataFrame by year bins.

    Args:
        df (pd.DataFrame): DataFrame with a 'Start year' column.
        year_bins (list): List of year bins to group by.
        base_year (int): cut-off for histirocal period. Default is 2020.

    Returns:
        pd.DataFrame: DataFrame with a new 'grouping_year' column.
    """
    min_start_year = min(year_bins) - 2.5
    base_year = 2020
    df = df[df["Start year"] > min_start_year]
    df = df[df["Retired year"].isna() | (df["Retired year"] > base_year)].reset_index(drop=True)
    df["grouping_year"] = np.take(year_bins, np.digitize(df["Start year"], year_bins, right=True))

    return df


def assign_node_from_gps(gem_data: pd.DataFrame, nodes: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Assign plant node based on GPS coordinates of the plant.
    Will cause issues if the nodes tolerance is too low

    Args:
        gem_data (pd.DataFrame): GEM data
        nodes (gpd.GeoDataFrame): node geometries (nodes as index).
    Returns:
        pd.DataFrame: DataFrame with assigned nodes."""

    gem_data["geometry"] = gem_data.apply(
        lambda row: Point(row["Longitude"], row["Latitude"]), axis=1
    )
    gem_gdf = gpd.GeoDataFrame(gem_data, geometry="geometry", crs="EPSG:4326")

    joined = nodes.reset_index(names="node").sjoin_nearest(gem_gdf, how="right")
    missing = joined[joined.node.isna()]
    if not missing.empty:
        logger.warning(
            f"Some GEM locations are not covered by the nodes at GPS: {missing['Plant name'].head()}"
        )
    return joined


def partition_gem_across_nodes(
    gem_data: pd.DataFrame, nodes: gpd.GeoDataFrame, admin_level=None
) -> pd.DataFrame:
    """
    Partition GEM data across nodes based on geographical coordinates.

    Args:
        gem_data (pd.DataFrame): DataFrame containing GEM data.
        nodes (geopandas.GeoDataFrame): GeoDataFrame containing node geometries (nodes as index).
        admin_level (int, optional): Administrative level for partitioning. Default is None (GPS).

    Returns:
        pd.DataFrame: DataFrame with GEM data partitioned across nodes.
    """
    if admin_level is not None and admin_level not in [0, 1, 2]:
        raise ValueError("admin_level must be None, 0, 1, or 2")

    # snap to admin_level
    if admin_level is not None:
        admin = ADM_COLS[admin_level]
        gem_data[admin] = gem_data[admin].str.replace(" ", "")
        uncovered_gem = set(gem_data[ADM_COLS[admin_level]]) - set(nodes.index)
        if uncovered_gem:
            logger.warning(
                f"Some GEM locations are not covered by the nodes at admin level {admin_level}: {uncovered_gem}"
                ". Consider partitioning with at a different admin_level or with GPS (None)."
            )
        gem_data["node"] = gem_data[admin]
        gem_data.dropna(subset=["node"], inplace=True)
        return gem_data
    else:
        gem_data["geometry"] = gem_data.apply(
            lambda row: Point(row["Longitude"], row["Latitude"]), axis=1
        )
        gem_gdf = gpd.GeoDataFrame(gem_data, geometry="geometry", crs="EPSG:4326")
        joined = nodes.reset_index(names="node").sjoin_nearest(gem_gdf, how="right")
        missing = joined[joined.node.isna()]
        if not missing.empty:
            logger.warning(
                f"Some GEM locations are not covered by the nodes at GPS: {missing['Plant name'].head()}"
            )
        return joined["node"]


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "build_powerplants",
            topology="current+FCG",
            co2_pathway="exp175default",
            planning_horizons="2020",
            # configfiles="resources/tmp/remind_coupled.yaml",
        )

    configure_logging(snakemake, logger=logger)

    config = snakemake.config
    cfg_GEM = config["global_energy_monitor_plants"]
    output_paths = dict(snakemake.output.items())
    params = snakemake.params

    # TODO add offsore for offsore wind
    nodes = gpd.read_file(snakemake.input.nodes)
    gem_data = load_gem_excel(snakemake.input.GEM_plant_tracker, sheetname="Power facilities")
    cleaned = clean_gem_data(gem_data, cfg_GEM)
    cleaned = group_by_year(
        cleaned, config["existing_capacities"]["grouping_years"], base_year=cfg_GEM["base_year"]
    )

    processed, requested = cleaned.Type.unique(), set(output_paths.keys())
    missing = requested - set(processed)
    extra = set(processed) - requested
    if missing:
        raise ValueError(
            f"Some techs requested existing_baseyear missing from GEM\n\t:{missing}\nAvailable Global Energy Monitor techs after processing:\n\t{processed}."
        )
    if extra:
        logger.warning(f"Techs from GEM {extra} not covered by existing_baseyear techs.")

    # TODO assign nodes
    assign_mode = config["existing_capacities"].get("node_assignment_mode", "simple")
    node_cfg = config["nodes"]

    if not node_cfg["split_provinces"]:
        assign_mode = "simple"
        cleaned["node"] = cleaned[ADM_LVL1]
    elif assign_mode == "simple":
        splits_inv = {}  # invert to get admin2 -> node
        for admin1, splits in node_cfg["splits"].items():
            splits_inv.update({vv: admin1 + "_" + k for k, v in splits.items() for vv in v})
        cleaned["node"] = cleaned[ADM_LVL2].map(splits_inv).fillna(cleaned[ADM_LVL1])
    else:
        if config["fetch_regions"]["simplify_tol"]["land"] > 0.05:
            logger.warning(
                "Using GPS assignment for existing capacities with land simplify_tol > 0.05. "
                "This may lead to inaccurate assignments (eg. Shanxi vs InnerMongolia coal power)."
            )
        cleaned["node"] = assign_node_from_gps(cleaned, nodes)

    datasets = {tech: cleaned[cleaned.Type == tech] for tech in requested}
    for name, ds in datasets.items():
        df = (
            ds.pivot_table(
                columns="grouping_year", index="node", values="Capacity (MW)", aggfunc="sum"
            )
            .fillna(0)
            .astype(int)
        )

        # sanity checks
        logger.debug(f"GEM Dataset for pypsa-tech {name}: has techs \n\t{ds.Technology.unique()}")

        df.to_csv(output_paths[name])
        logger.info(f"cap for {name} {df.sum().sum()/1000}")

    logger.info(f"GEM capacities saved to {os.path.dirname(output_paths[name])}")
