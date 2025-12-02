"""
Functions to prepare existing assets for the network

SHORT TERM FIX until PowerPlantMatching is implemented
- required as split from add_existing_baseyear for remind compat
"""

# TODO improve docstring
import logging
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from _helpers import configure_logging, mock_snakemake
from _pypsa_helpers import make_periodic_snapshots
from add_electricity import load_costs
from constants import YEAR_HRS

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
        end_year=None if not snapshot_cfg["end_year_plus1"] else year + 1,
    )

    # load costs
    n_years = config["snapshots"]["frequency"] * len(snapshots) / YEAR_HRS

    return n_years


# TODO switch this to a single file and remove agg
# Associated #TODO: do not split files in build_powerplants
def read_existing_capacities(paths_dict: dict[str, os.PathLike], techs: list) -> pd.DataFrame:
    """Read existing capacities from csv files and format them
    Args:
        paths_dict (dict[str, os.PathLike]): dictionary with paths to the csv files
        techs (list): list of technologies to read
    Returns:
        pd.DataFrame: DataFrame with existing capacities
    """
    # TODO fix centralise (make a dict from start?)
    carrier = {
        "coal": "coal power plant",
        "CHP coal": "central coal CHP",
        "CHP gas": "central gas CHP",
        "OCGT": "gas OCGT",
        "CCGT": "gas CCGT",
        "solar": "solar",
        "solar thermal": "central solar thermal",
        "onwind": "onwind",
        "offwind": "offwind",
        "coal boiler": "central coal boiler",
        "ground heat pump": "central ground-sourced heat pump",
        "nuclear": "nuclear",
    }
    carrier = {k: v for k, v in carrier.items() if k in techs}

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
    """add/fill missing dateIn, discretize lifetime to grouping year, rename columns
    drop plants that were retired before the smallest sim timeframe

    Args:
        existing_df (pd.DataFrame): the existing capacities
        costs (pd.DataFrame): the technoeconomic data
        year_bins (list): the year groups


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
    if lifetimes.isna().any():
        raise ValueError(
            f"Some assets have no lifetime assigned: \n{lifetimes[lifetimes.isna()]}. "
            "Please check the costs file for the missing lifetimes."
        )
    existing_df.loc[:, "DateOut"] = existing_df.DateOut.fillna(lifetimes) + existing_df.DateIn

    existing_df["lifetime"] = existing_df.DateOut - existing_df["grouping_year"]
    existing_df.rename(columns={"cluster_bus": "bus"}, inplace=True)

    phased_out = existing_df[existing_df["DateOut"] < baseyear].index
    existing_df.drop(phased_out, inplace=True)

    # check the grouping years are appropriate
    newer_assets = (existing_df.DateIn > max(year_bins)).sum()
    if newer_assets:
        raise ValueError(
            f"There are {newer_assets} assets with build year "
            f"after last power grouping year {max(year_bins)}. "
            "These assets are dropped and not considered."
            "Redefine the grouping years to keep them or"
            " remove pre-construction/construction/... states."
        )

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


def convert_CHP_to_poweronly(capacities: pd.DataFrame) -> pd.DataFrame:
    """Convert CHP capacities to power-only capacities by removing the heat part

    Args:
        capacities (pd.DataFrame): DataFrame with existing capacities
    Returns:
        pd.DataFrame: DataFrame with converted capacities
    """
    # Convert CHP to power-only by removing the heat part
    chp_mask = capacities.Tech.str.contains("CHP")
    capacities.loc[chp_mask, "Fueltype"] = (
        capacities.loc[chp_mask, "Fueltype"]
        .str.replace("central coal CHP", "coal power plant")
        .str.replace("central gas CHP", "gas CCGT")
    )
    # update the Tech field based on the converted Fueltype
    capacities.loc[chp_mask, "Tech"] = (
        capacities.loc[chp_mask, "Fueltype"]
        .str.replace(" CHP", "")
        .str.replace("CHP ", " ")
        .str.replace("gas ", "")
        .str.replace("coal power plant", "coal")
    )
    return capacities


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "prepare_baseyear_capacities",
            topology="current+FCG",
            co2_pathway="SSP2-PkBudg1000-pseudo-coupled",
            planning_horizons="2020",
            configfiles="resources/tmp/pseudo_coupled.yml",
        )

    configure_logging(snakemake, logger=logger)

    config = snakemake.config
    params = snakemake.params
    # reference pypsa cost (lifetime) year is simulation Baseyar
    baseyear = min([int(y) for y in config["scenario"]["planning_horizons"]])
    tech_costs = snakemake.input.tech_costs
    data_paths = {k: v for k, v in snakemake.input.items()}

    n_years = determine_simulation_timespan(snakemake.config, baseyear)
    costs = load_costs(tech_costs, config["costs"], config["electricity"], baseyear, n_years)

    techs = config["existing_capacities"]["techs"]
    existing_capacities = read_existing_capacities(data_paths, techs)
    existing_capacities = existing_capacities.query("Fueltype in @techs | Tech in @techs")

    year_bins = config["existing_capacities"]["grouping_years"]
    # TODO add renewables
    existing_capacities = assign_year_bins(existing_capacities, year_bins)
    if params.CHP_to_elec:
        existing_capacities = convert_CHP_to_poweronly(existing_capacities)

    installed = fix_existing_capacities(existing_capacities, costs, year_bins, baseyear)

    if installed.empty or installed.lifetime.isna().any():
        logger.warning(
            f"The following assets have no lifetime assigned and are for ever lived: \n{installed[installed.lifetime.isna()]}"
        )

    installed.to_csv(snakemake.output.installed_capacities)

    logger.info(f"Installed capacities saved to {snakemake.output.installed_capacities}")
