# coding: utf-8
"""
Functions to prepare existing assets for the network

SHORT TERM FIX until PowerPlantMatching is implemented
- required as split from add_existing_baseyear for remind compat
"""
# TODO improve docstring
import logging
import numpy as np
import pandas as pd
import os

from types import SimpleNamespace

from constants import YEAR_HRS
from add_electricity import load_costs
from _helpers import mock_snakemake, configure_logging
from _pypsa_helpers import make_periodic_snapshots

logger = logging.getLogger(__name__)
idx = pd.IndexSlice
spatial = SimpleNamespace()


def determine_simulation_timespan(config: dict, year: int) -> int:
    """Determine the simulation timespan in years (so the network object is not needed)
    Args:
        config (dict): the snakemake config
        year (int): the year to simulate
    Returns:
        int: the simulation timespan in years
    """

    # make snapshots (drop leap days) -> possibly do all the unpacking in the function
    snapshot_cfg = config["snapshots"]
    snapshots = make_periodic_snapshots(
        year=year,
        freq=snapshot_cfg["freq"],
        start_day_hour=snapshot_cfg["start"],
        end_day_hour=snapshot_cfg["end"],
        bounds=snapshot_cfg["bounds"],
        # naive local timezone
        tz=None,
        end_year=(None if not snapshot_cfg["end_year_plus1"] else year + 1),
    )

    # load costs
    n_years = config["snapshots"]["frequency"] * len(snapshots) / YEAR_HRS

    return n_years


def read_existing_capacities(paths_dict: dict[str, os.PathLike]) -> pd.DataFrame:
    """Read existing capacities from csv files and format them
    Args:
        paths_dict (dict[str, os.PathLike]): dictionary with paths to the csv files
    Returns:
        pd.DataFrame: DataFrame with existing capacities
    """
    # TODO fix centralise (make a dict from start?)
    carrier = {
        "coal": "coal power plant",
        "CHP coal": "central coal CHP",
        "CHP gas": "central gas CHP",
        "OCGT": "OCGT gas",
        "solar": "solar",
        "solar thermal": "central solar thermal",
        "onwind": "onwind",
        "offwind": "offwind",
        "coal boiler": "central coal boiler",
        "ground heat pump": "central ground-sourced heat pump",
        "nuclear": "nuclear",
    }
    df_agg = pd.DataFrame()
    for tech in carrier:
        df = pd.read_csv(paths_dict[tech], index_col=0).fillna(0.0)
        df.columns = df.columns.astype(int)
        df = df.sort_index()

        for year in df.columns:
            for node in df.index:
                name = f"{node}-{tech}-{year}"
                capacity = df.loc[node, year]
                if capacity > 0.0:
                    df_agg.at[name, "Fueltype"] = carrier[tech]
                    df_agg.at[name, "Tech"] = tech
                    df_agg.at[name, "Capacity"] = capacity
                    df_agg.at[name, "DateIn"] = year
                    df_agg.at[name, "cluster_bus"] = node

    return df_agg


def fix_existing_capacities(
    existing_df: pd.DataFrame, costs: pd.DataFrame, year_bins: list, baseyear: int
) -> pd.DataFrame:
    """add/fill missing dateIn, drop expired assets, drop too new assets

    Args:
        existing_df (pd.DataFrame): the existing capacities
        costs (pd.DataFrame): the technoeconomic data
        year_bins (list): the year groups
        baseyear (int): the base year (run year)

    Returns:
        pd.DataFrame: fixed capacities
    """
    existing_df.DateIn = existing_df.DateIn.astype(int)
    # add/fill missing dateIn
    if "DateOut" not in existing_df.columns:
        existing_df["DateOut"] = np.nan
    # names matching costs split across FuelType and Tech, apply to both. Fillna means no overwrite
    lifetimes = existing_df.Fueltype.map(costs.lifetime).fillna(
        existing_df.Tech.map(costs.lifetime)
    )
    existing_df.loc[:, "DateOut"] = existing_df.DateOut.fillna(lifetimes) + existing_df.DateIn

    # TODO go through the pypsa-EUR fuel drops for the new ppmatching style
    # drop assets which are already phased out / decommissioned
    phased_out = existing_df[existing_df["DateOut"] < baseyear].index
    existing_df.drop(phased_out, inplace=True)

    newer_assets = (existing_df.DateIn > max(year_bins)).sum()
    if newer_assets:
        logger.warning(
            f"There are {newer_assets} assets with build year "
            f"after last power grouping year {max(year_bins)}. "
            "These assets are dropped and not considered."
            "Consider to redefine the grouping years to keep them."
        )
        to_drop = existing_df[existing_df.DateIn > max(year_bins)].index
        existing_df.drop(to_drop, inplace=True)

    existing_df["lifetime"] = existing_df.DateOut - existing_df["grouping_year"]

    existing_df.rename(columns={"cluster_bus": "bus"}, inplace=True)
    return existing_df


def assign_year_bins(df: pd.DataFrame, year_bins: list) -> pd.DataFrame:
    """
    Assign a year bin to the existing capacities according to the config

    Args:
        df (pd.DataFrame): DataFrame with existing capacities and build years (DateIn)
        year_bins (list): years to bin the existing capacities to
    Returns:
        pd.DataFrame: DataFrame regridded to the year bins
    """

    df_ = df.copy()
    # bin by years (np.digitize)
    df_["grouping_year"] = np.take(year_bins, np.digitize(df.DateIn, year_bins, right=True))
    return df_.fillna(0)


def distribute_vre_by_grade(cap_by_year: pd.Series, grade_capacities: pd.Series) -> pd.DataFrame:
    """distribute vre capacities by grade potential, use up better grades first

    Args:
        cap_by_year (pd.Series): the vre tech potential p_nom_max added per year
        grade_capacities (pd.Series): the vre grade potential for the tech and bus
    Returns:
        pd.DataFrame: DataFrame with the distributed vre capacities (shape: years x buses)
    """

    availability = cap_by_year.sort_index(ascending=False)
    to_distribute = grade_capacities.fillna(0).sort_index()
    n_years = len(to_distribute)
    n_sources = len(availability)

    # To store allocation per year per source (shape: sources x years)
    allocation = np.zeros((n_sources, n_years), dtype=int)
    remaining = availability.values

    for j in range(n_years):
        needed = to_distribute.values[j]
        cumsum = np.cumsum(remaining)
        used_up = cumsum < needed
        cutoff = np.argmax(cumsum >= needed)

        allocation[used_up, j] = remaining[used_up]

        if needed > (cumsum[cutoff - 1] if cutoff > 0 else 0):
            allocation[cutoff, j] = needed - (cumsum[cutoff - 1] if cutoff > 0 else 0)

        # Subtract what was used from availability
        remaining -= allocation[:, j]

    return pd.DataFrame(data=allocation, columns=grade_capacities.index, index=availability.index)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "prepare_baseyear_capacities",
            topology="current+FCG",
            co2_pathway="remind_ssp2NPI",
            planning_horizons="2070",
            heating_demand="positive",
        )

    configure_logging(snakemake, logger=logger)

    config = snakemake.config
    # remind extends beyond pypsa: limit the reference pypsa cost year to the last pypsa year
    plan_year = int(snakemake.wildcards["planning_horizons"])
    cost_year = min(snakemake.params.last_pypsa_cost_year, plan_year)
    tech_costs = snakemake.input.tech_costs.replace(str(plan_year), str(cost_year))
    data_paths = {k: v for k, v in snakemake.input.items()}

    n_years = determine_simulation_timespan(
        snakemake.config, snakemake.wildcards["planning_horizons"]
    )
    baseyear = int(snakemake.wildcards["planning_horizons"])
    costs = load_costs(tech_costs, config["costs"], config["electricity"], cost_year, n_years)

    existing_capacities = read_existing_capacities(data_paths)
    year_bins = config["existing_capacities"]["grouping_years"]
    # TODO add renewables
    existing_capacities = assign_year_bins(existing_capacities, year_bins)
    installed = fix_existing_capacities(existing_capacities, costs, year_bins, baseyear)

    if installed.empty or installed.lifetime.isna().any():
        logger.warning(
            f"The following assets have no lifetime assigned and are for ever lived: \n{installed[installed.lifetime.isna()]}"
        )

    installed.to_csv(snakemake.output.installed_capacities)
