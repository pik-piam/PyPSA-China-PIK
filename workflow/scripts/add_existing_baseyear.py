# coding: utf-8
"""
Functions to add brownfield capacities to the network for a reference year
"""
# TODO improve docstring
import logging
import numpy as np
import pandas as pd
import pypsa
import os
import re

from types import SimpleNamespace

from constants import YEAR_HRS
from add_electricity import load_costs
from _helpers import mock_snakemake, configure_logging
from _pypsa_helpers import shift_profile_to_planning_year

logger = logging.getLogger(__name__)
idx = pd.IndexSlice
spatial = SimpleNamespace()


def add_build_year_to_new_assets(n: pypsa.Network, baseyear: int):
    """add a build year to new assets

    Args:
        n (pypsa.Network): the network
        baseyear (int): year in which optimized assets are built
    """

    # Give assets with lifetimes and no build year the build year baseyear
    for c in n.iterate_components(["Link", "Generator", "Store"]):
        attr = "e" if c.name == "Store" else "p"

        assets = c.df.index[(c.df.lifetime != np.inf) & (c.df[attr + "_nom_extendable"] is True)]

        # add -baseyear to name
        rename = pd.Series(c.df.index, c.df.index)
        rename[assets] += "-" + str(baseyear)
        c.df.rename(index=rename, inplace=True)

        assets = c.df.index[
            (c.df.lifetime != np.inf)
            & (c.df[attr + "_nom_extendable"] is True)
            & (c.df.build_year == 0)
        ]
        c.df.loc[assets, "build_year"] = baseyear

        # rename time-dependent
        selection = n.component_attrs[c.name].type.str.contains("series") & n.component_attrs[
            c.name
        ].status.str.contains("Input")
        for attr in n.component_attrs[c.name].index[selection]:
            c.pnl[attr].rename(columns=rename, inplace=True)


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
        "CHP coal": "CHP coal",
        "CHP gas": "CHP gas",
        "OCGT": "OCGT gas",
        "solar": "solar",
        "solar thermal": "solar thermal",
        "onwind": "onwind",
        "offwind": "offwind",
        "coal boiler": "coal boiler",
        "ground heat pump": "heat pump",
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
        pd.DataFrame: _description_
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


def add_existing_vre_capacities(
    n: pypsa.Network,
    costs: pd.DataFrame,
    vre_caps: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Add existing VRE capacities to the network and distribute them by vre grade potential.
    Adapted from pypsa-eur but the VRE capacities are province resolved.

    NOTE that using this function requires adding the land-use constraint in solve_network so
      that the existing capacities are subtracted from the available potential

    Args:
        n (pypsa.Network): the network
        costs (pd.DataFrame): costs of the technologies
        vre_caps (pd.DataFrame): existing VRE capacities in MW
        config (dict): snakemake configuration dictionary
    Returns:
        pd.DataFrame: DataFrame with existing VRE capacities distributed by CF grade

    """

    tech_map = {"solar": "PV", "onwind": "Onshore", "offwind-ac": "Offshore", "offwind": "Offshore"}
    tech_map = {k: tech_map[k] for k in tech_map if k in config["Techs"]["vre_techs"]}

    grouped_vre = vre_caps.groupby(["Tech", "bus", "DateIn"]).Capacity.sum()
    vre_df = grouped_vre.unstack().reset_index()
    df_agg = pd.DataFrame()

    for carrier in tech_map:

        df = vre_df[vre_df.Tech == carrier].drop(columns=["Tech"])
        df.set_index("bus", inplace=True)
        df.columns = df.columns.astype(int)

        # fetch existing vre generators (n grade bins per node)
        gen_i = n.generators.query("carrier == @carrier").index
        carrier_gens = n.generators.loc[gen_i]
        res_capacities = []
        # for each bus, distribute the vre capacities by grade potential - best first
        for bus, group in carrier_gens.groupby("bus"):
            if bus not in df.index:
                continue
            res_capacities.append(distribute_vre_by_grade(group.p_nom_max, df.loc[bus]))
        res_capacities = pd.concat(res_capacities, axis=0)

        for year in df.columns:
            for gen in res_capacities.index:
                bus_bin = re.sub(f" {carrier}.*", "", gen)
                bus, bin_id = bus_bin.rsplit(" ", maxsplit=1)
                name = f"{bus_bin} {carrier}-{year}"
                capacity = res_capacities.loc[gen, year]
                if capacity > 0.0:
                    cost_key = carrier.split("-", maxsplit=1)[0]
                    df_agg.at[name, "Fueltype"] = carrier
                    df_agg.at[name, "Capacity"] = capacity
                    df_agg.at[name, "DateIn"] = year
                    df_agg.at[name, "lifetime"] = costs.at[cost_key, "lifetime"]
                    df_agg.at[name, "DateOut"] = year + costs.at[cost_key, "lifetime"] - 1
                    df_agg.at[name, "bus"] = bus
                    df_agg.at[name, "resource_class"] = bin_id

    df_agg.loc[:, "Tech"] = df_agg.Fueltype
    return df_agg


def add_power_capacities_installed_before_baseyear(
    n: pypsa.Network,
    costs: pd.DataFrame,
    config: dict,
    installed_capacities: pd.DataFrame,
    baseyear: int = 2020,
):
    """
    Add existing power capacities to the network

    Args:
        n (pypsa.Network): the network
        grouping_years (list): Mistery # TODO
        costs (pd.DataFrame): costs of the technologies
        config (dict): configuration dictionary
        installed_capacities (pd.DataFrame): installed capacities in MW
        baseyear (int): planning horizon / cutoff year for brownfield assets (< baseyear)
    """

    logger.info("adding power capacities installed before baseyear")

    df = installed_capacities.copy()
    # TODO fix this based on config
    carrier_map = {
        "coal power plant": "coal",
        "CHP coal": "CHP coal",
        "CHP gas": "CHP gas",
        "OCGT gas": "gas OCGT",
        "solar": "solar",
        "solar thermal": "solar thermal",
        "onwind": "onwind",
        "offwind": "offwind",
        "coal boiler": "coal boiler",
        "heat pump": "ground heat pump",
        "nuclear": "nuclear",
    }
    costs_map = {
        "coal power plant": "coal",
        "CHP coal": "central coal CHP",
        "CHP gas": "central gas CHP",
        "OCGT gas": "OCGT",
        "solar": "solar",
        "solar thermal": "central solar thermal",
        "onwind": "onwind",
        "offwind": "offwind",
        "coal boiler": "central coal boier",
        "heat pump": "central ground-sourced heat pump",
        "nuclear": "nuclear",
    }

    # in case user forgot to do it
    df = fix_existing_capacities(
        df, costs, config["existing_capacities"]["grouping_years"], baseyear
    )
    df.resource_class.fillna("", inplace=True)
    df_ = df.pivot_table(
        index=["grouping_year", "Fueltype", "resource_class"],
        columns="bus",
        values="Capacity",
        aggfunc="sum",
    )
    df_.fillna(0, inplace=True)
    # TODO do we really need to loop over the years?
    for grouping_year, generator, resource_grade in df_.index:
        logger.info(generator + " " + str(grouping_year))
        # capacity is the capacity in MW at each node for this
        # capacity is the capacity in MW at each node for this
        capacity = df_.loc[grouping_year, generator]
        capacity = capacity[~capacity.isna()]
        capacity = capacity[capacity > config["existing_capacities"]["threshold_capacity"]].T
        if capacity.values.max() == 0:
            continue
        capacity = capacity[capacity > 0].dropna()
        # fix index for network.add (merge grade to name)
        capacity = capacity.unstack()
        buses = capacity.index.get_level_values(1)
        capacity.index = (
            capacity.index.get_level_values(1) + " " + capacity.index.get_level_values(0)
        )
        capacity.index = capacity.index.str.rstrip() + " " + costs_map[generator]

        costs_key = costs_map[generator]

        vre_carriers = ["solar", "onwind", "offwind"]
        if generator in vre_carriers:
            mask = n.generators_t.p_max_pu.columns.map(n.generators.carrier) == generator
            p_max_pu = n.generators_t.p_max_pu.loc[:, mask]
            n.add(
                "Generator",
                capacity.index,
                suffix="-" + str(grouping_year),
                bus=buses,
                carrier=carrier_map[generator],
                p_nom=capacity,
                p_nom_min=capacity,
                p_nom_extendable=False,
                marginal_cost=costs.at[costs_key, "marginal_cost"],
                efficiency=costs.at[costs_key, "efficiency"],
                p_max_pu=p_max_pu[capacity.index],
                build_year=grouping_year,
                lifetime=costs.at[costs_key, "lifetime"],
                location=buses,
            )

        elif generator == "coal power plant":
            n.add(
                "Generator",
                capacity.index,
                suffix=f"-{grouping_year}",
                bus=buses,
                carrier=carrier_map[generator],
                p_nom=capacity,
                p_nom_min=capacity,
                p_nom_extendable=False,
                marginal_cost=costs.at[costs_key, "marginal_cost"],
                efficiency=costs.at[costs_key, "efficiency"],
                build_year=grouping_year,
                lifetime=costs.at[costs_key, "lifetime"],
                location=buses,
            )

        elif generator == "nuclear":
            n.add(
                "Generator",
                capacity.index,
                suffix="-" + str(grouping_year),
                bus=buses,
                carrier=carrier_map[generator],
                p_nom=capacity,
                p_nom_min=capacity,
                p_nom_extendable=False,
                p_min_pu=0.7,
                marginal_cost=costs.at[costs_key, "marginal_cost"],
                efficiency=costs.at[costs_key, "efficiency"],
                build_year=grouping_year,
                lifetime=costs.at[costs_key, "lifetime"],
                location=buses,
            )

        elif generator == "OCGT gas":
            bus0 = buses + " gas"
            n.add(
                "Link",
                capacity.index,
                suffix="-" + str(grouping_year),
                bus0=bus0,
                bus1=capacity.index,
                carrier=carrier_map[generator],
                marginal_cost=costs.at[costs_key, "efficiency"]
                * costs.at[costs_key, "VOM"],  # NB: VOM is per MWel
                # NB: fixed cost is per MWel
                p_nom=capacity / costs.at[costs_key, "efficiency"],
                p_nom_min=capacity / costs.at[costs_key, "efficiency"],
                p_nom_extendable=False,
                efficiency=costs.at[costs_key, "efficiency"],
                build_year=grouping_year,
                lifetime=costs.at[costs_key, "lifetime"],
                location=buses,
            )
        elif (
            generator in ["solar thermal", "CHP coal", "CHP gas", "heat pump", "coal boiler"]
            and not config["heat_coupling"]
        ):
            logger.info(f"Skipped {generator} because heat coupling is not activated")

        elif generator == "solar thermal":
            p_max_pu = n.generators_t.p_max_pu[capacity.index + " central " + generator]
            p_max_pu.columns = capacity.index
            n.add(
                "Generator",
                capacity.index,
                suffix=f"-{str(grouping_year)}",
                bus=buses + " central heat",
                carrier=carrier_map[generator],
                p_nom=capacity,
                p_nom_min=capacity,
                p_nom_extendable=False,
                marginal_cost=costs.at["central " + generator, "marginal_cost"],
                p_max_pu=p_max_pu,
                build_year=grouping_year,
                lifetime=costs.at["central " + generator, "lifetime"],
                location=buses,
            )

        elif generator == "CHP coal":
            bus0 = buses + " coal"
            # TODO soft-code efficiency !!
            hist_efficiency = 0.37
            n.add(
                "Link",
                capacity.index,
                suffix=f"-{str(grouping_year)}",
                bus0=bus0,
                bus1=capacity.index,
                carrier=carrier_map[generator],
                marginal_cost=hist_efficiency
                * costs.at["central coal CHP", "VOM"],  # NB: VOM is per MWel
                p_nom=capacity / hist_efficiency,
                p_nom_min=capacity / hist_efficiency,
                p_nom_extendable=False,
                efficiency=hist_efficiency,
                p_nom_ratio=1.0,
                c_b=0.75,
                build_year=grouping_year,
                lifetime=costs.at["central coal CHP", "lifetime"],
                location=buses,
            )

            n.add(
                "Link",
                capacity.index,
                suffix=f" boiler-{str(grouping_year)}",
                bus0=bus0,
                bus1=capacity.index + " central heat",
                carrier=carrier_map[generator],
                marginal_cost=hist_efficiency
                * costs.at["central coal CHP", "VOM"],  # NB: VOM is per MWel
                p_nom=capacity / hist_efficiency * costs.at["central coal CHP", "c_v"],
                p_nom_min=capacity / hist_efficiency * costs.at["central coal CHP", "c_v"],
                p_nom_extendable=False,
                efficiency=hist_efficiency / costs.at["central coal CHP", "c_v"],
                build_year=grouping_year,
                lifetime=costs.at["central coal CHP", "lifetime"],
                location=buses,
            )

        elif generator == "CHP gas":
            hist_efficiency = 0.37
            bus0 = buses + " gas"
            n.add(
                "Link",
                capacity.index,
                suffix=f"-{str(grouping_year)}",
                bus0=bus0,
                bus1=capacity.index,
                carrier=carrier_map[generator],
                marginal_cost=hist_efficiency
                * costs.at["central gas CHP", "VOM"],  # NB: VOM is per MWel
                capital_cost=hist_efficiency
                * costs.at["central gas CHP", "capital_cost"],  # NB: fixed cost is per MWel,
                p_nom=capacity / hist_efficiency,
                p_nom_min=capacity / hist_efficiency,
                p_nom_extendable=False,
                efficiency=hist_efficiency,
                p_nom_ratio=1.0,
                c_b=costs.at["central gas CHP", "c_b"],
                build_year=grouping_year,
                lifetime=costs.at["central gas CHP", "lifetime"],
                location=buses,
            )
            n.add(
                "Link",
                capacity.index,
                suffix=f" boiler-{str(grouping_year)}",
                bus0=bus0,
                bus1=capacity.index + " central heat",
                carrier=carrier_map[generator],
                marginal_cost=hist_efficiency
                * costs.at["central gas CHP", "VOM"],  # NB: VOM is per MWel
                p_nom=capacity / hist_efficiency * costs.at["central gas CHP", "c_v"],
                p_nom_min=capacity / hist_efficiency * costs.at["central gas CHP", "c_v"],
                p_nom_extendable=False,
                efficiency=hist_efficiency / costs.at["central gas CHP", "c_v"],
                build_year=grouping_year,
                lifetime=costs.at["central gas CHP", "lifetime"],
                location=buses,
            )

        elif generator == "coal boiler":
            bus0 = buses + " coal"
            for cat in [" central "]:
                n.add(
                    "Link",
                    capacity.index,
                    suffix="" + cat + generator + "-" + str(grouping_year),
                    bus0=bus0,
                    bus1=capacity.index + cat + "heat",
                    carrier=carrier_map[generator],
                    marginal_cost=costs.at[cat.lstrip() + generator, "efficiency"]
                    * costs.at[cat.lstrip() + generator, "VOM"],
                    capital_cost=costs.at[cat.lstrip() + generator, "efficiency"]
                    * costs.at[cat.lstrip() + generator, "capital_cost"],
                    p_nom=capacity / costs.at[cat.lstrip() + generator, "efficiency"],
                    p_nom_min=capacity / costs.at[cat.lstrip() + generator, "efficiency"],
                    p_nom_extendable=False,
                    efficiency=costs.at[cat.lstrip() + generator, "efficiency"],
                    build_year=grouping_year,
                    lifetime=costs.at[cat.lstrip() + generator, "lifetime"],
                    location=buses,
                )
        # TODO fix centralise
        # TODO fix this to make air pumps?
        elif generator == "heat pump":
            # TODO separate the read operation from the add operation
            with pd.HDFStore(snakemake.input.cop_name, mode="r") as store:
                gshp_cop = store["gshp_cop_profiles"]
                gshp_cop.index = gshp_cop.index.tz_localize(None)
                gshp_cop = shift_profile_to_planning_year(
                    gshp_cop, snakemake.wildcards.planning_horizons
                )
                gshp_cop = gshp_cop.loc[n.snapshots]
            n.add(
                "Link",
                capacity.index,
                suffix="-" + str(grouping_year),
                bus0=capacity.index,
                bus1=capacity.index + " central heat",
                carrier="heat pump",
                efficiency=(
                    gshp_cop[capacity.index]
                    if config["time_dep_hp_cop"]
                    else costs.at["decentral ground-sourced heat pump", "efficiency"]
                ),
                capital_cost=costs.at["decentral ground-sourced heat pump", "efficiency"]
                * costs.at["decentral ground-sourced heat pump", "capital_cost"],
                marginal_cost=costs.at["decentral ground-sourced heat pump", "efficiency"]
                * costs.at["decentral ground-sourced heat pump", "marginal_cost"],
                p_nom=capacity / costs.at["decentral ground-sourced heat pump", "efficiency"],
                p_nom_min=capacity / costs.at["decentral ground-sourced heat pump", "efficiency"],
                p_nom_extendable=False,
                build_year=grouping_year,
                lifetime=costs.at["decentral ground-sourced heat pump", "lifetime"],
                location=buses,
            )

        else:
            logger.warning(
                f"Skipped existing capacitity for {generator}"
                + " - tech not implemented as existing capacity"
            )


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "add_existing_baseyear",
            topology="current+FCG",
            co2_pathway="remind_ssp2NPI",
            planning_horizons="2025",
            heating_demand="positive",
        )

    configure_logging(snakemake, logger=logger)
    # options = snakemake.config["sector"]
    # sector_opts = '168H-T-H-B-I-solar+p3-dist1'
    # opts = sector_opts.split('-')

    n = pypsa.Network(snakemake.input.network)
    n_years = n.snapshot_weightings.generators.sum() / YEAR_HRS

    # define spatial resolution of carriers
    # spatial = define_spatial(n.buses[n.buses.carrier=="AC"].index, options)
    vre_techs = ["solar", "onwind", "offwind"]
    baseyear = snakemake.params["baseyear"]
    # add_build_year_to_new_assets(n, baseyear)
    if snakemake.params["add_baseyear_to_assets"]:
        add_build_year_to_new_assets(n, baseyear)

    config = snakemake.config
    tech_costs = snakemake.input.tech_costs
    cost_year = snakemake.wildcards["planning_horizons"]
    data_paths = {k: v for k, v in snakemake.input.items()}

    costs = load_costs(tech_costs, config["costs"], config["electricity"], cost_year, n_years)

    existing_capacities = read_existing_capacities(data_paths)
    year_bins = config["existing_capacities"]["grouping_years"]
    # TODO add renewables
    existing_capacities = assign_year_bins(existing_capacities, year_bins)
    df = fix_existing_capacities(existing_capacities, costs, year_bins, baseyear)

    vre_caps = df.query("Tech in @vre_techs | Fueltype in @vre_techs")
    # vre_caps.loc[:, "Country"] = coco.CountryConverter().convert(["China"], to="iso2")
    vres = add_existing_vre_capacities(n, costs, vre_caps, config)
    vres = assign_year_bins(vres, year_bins)
    df = pd.concat([df.query("Tech not in @vre_techs & Fueltype not in @vre_techs"), vres], axis=0)

    # add to the network
    add_power_capacities_installed_before_baseyear(n, costs, config, df)
    n.export_to_netcdf(snakemake.output[0])

    logger.info("Existing capacities successfully added to network")
