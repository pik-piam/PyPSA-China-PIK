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

from types import SimpleNamespace
from constants import YEAR_HRS, CARRIERS
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


def read_existing_capacities(paths_dict: dict) -> pd.DataFrame:
    # TODO fix/centralise ()
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
    # TODO fix centralise (make a dict from start?)
    for tech in carrier:

        # TODO make argument
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


def assign_year_bins(df: pd.DataFrame, year_bins: list) -> pd.DataFrame:
    """
    Assign a year bin to the existing capacities according to the config

    Args:
        df (pd.DataFrame): DataFrame with existing capacities and build years (DateIn)
        year_bins (list): years to bin the existing capacities to
    """

    # Assign a year bin to the existing capacities
    df_agg = existing_capacities

    # bin by years (np.digitize)
    df_agg["grouping_year"] = np.take(year_bins, np.digitize(df_agg.DateIn, year_bins, right=True))
    df = df_agg.pivot_table(
        index=["grouping_year", "Tech"], columns="cluster_bus", values="Capacity", aggfunc="sum"
    )
    return df.fillna(0)


def distribute_vres_by_grade(p_max_nom: pd.DataFrame, df_agg: pd.DataFrame, config: dict):
    """Assign the built up capacity to the best vre grades

    Args:

        df_agg (_type_): _description_
        config (_type_): _description_

    Raises:
        NotImplementedError: _description_
    """

    raise NotImplementedError("This function is not implemented yet.")


def add_power_capacities_installed_before_baseyear(
    n: pypsa.Network,
    grouping_years: list,
    costs: pd.DataFrame,
    config: dict,
    installed_capacities: pd.DataFrame,
):
    """
    Add existing power capacities to the network
    Args:
        n (pypsa.Network): the network
        grouping_years (list): Mistery # TODO
        costs (pd.DataFrame): costs of the technologies
        config (dict): configuration dictionary
        installed_capacities (pd.DataFrame): installed capacities in MW
    """

    logger.info("adding power capacities installed before baseyear")

    df = installed_capacities.copy()

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

    # generator seems to be a carrier here
    for grouping_year, generator in df.index:

        # capacity is the capacity in MW at each node for this
        capacity = df.loc[grouping_year, generator]
        capacity = capacity[~capacity.isna()]
        capacity = capacity[capacity > config["existing_capacities"]["threshold_capacity"]]

        vre_carriers = ["solar", "onwind", "offwind"]
        if generator in vre_carriers:

            mask = n.generators_t.p_max_pu.columns.map(n.generators.carrier) == generator
            p_max_pu = n.generators_t.p_max_pu.loc[:, mask]
            n.add(
                "Generator",
                capacity.index,
                suffix=" " + generator + "-" + str(grouping_year),
                bus=capacity.index,
                carrier=carrier[generator],
                p_nom=capacity,
                p_nom_min=capacity,
                p_nom_extendable=False,
                marginal_cost=costs.at[generator, "marginal_cost"],
                capital_cost=costs.at[generator, "capital_cost"],
                efficiency=costs.at[generator, "efficiency"],
                p_max_pu=p_max_pu.rename(columns=n.generators.bus),
                build_year=grouping_year,
                lifetime=costs.at[generator, "lifetime"],
            )

        if generator == "coal":
            n.add(
                "Generator",
                capacity.index,
                suffix=" " + generator + "-" + str(grouping_year),
                bus=capacity.index,
                carrier=carrier[generator],
                p_nom=capacity,
                p_nom_extendable=False,
                marginal_cost=costs.at[generator, "marginal_cost"],
                capital_cost=costs.at[generator, "capital_cost"],
                efficiency=costs.at[generator, "efficiency"],
                build_year=grouping_year,
                lifetime=costs.at[generator, "lifetime"],
            )

        if generator == "nuclear":
            n.add(
                "Generator",
                capacity.index,
                suffix=" " + generator + "-" + str(grouping_year),
                bus=capacity.index,
                carrier=carrier[generator],
                p_nom=capacity,
                p_nom_min=capacity,
                p_nom_extendable=False,
                p_min_pu=0.7,
                marginal_cost=costs.at[generator, "marginal_cost"],
                capital_cost=costs.at[generator, "capital_cost"],
                efficiency=costs.at[generator, "efficiency"],
                build_year=grouping_year,
                lifetime=costs.at[generator, "lifetime"],
            )

        if generator == "solar thermal":
            p_max_pu = n.generators_t.p_max_pu[capacity.index + " central " + generator]
            p_max_pu.columns = capacity.index
            n.add(
                "Generator",
                capacity.index,
                suffix=" central " + generator + "-" + str(grouping_year),
                bus=capacity.index + " central heat",
                carrier=carrier[generator],
                p_nom=capacity,
                p_nom_min=capacity,
                p_nom_extendable=False,
                marginal_cost=costs.at["central " + generator, "marginal_cost"],
                capital_cost=costs.at["central " + generator, "capital_cost"],
                p_max_pu=p_max_pu,
                build_year=grouping_year,
                lifetime=costs.at["central " + generator, "lifetime"],
            )

        if generator == "CHP coal":
            bus0 = capacity.index + " coal"
            n.add(
                "Link",
                capacity.index,
                suffix=" " + generator + " generator" + "-" + str(grouping_year),
                bus0=bus0,
                bus1=capacity.index,
                carrier=carrier[generator],
                marginal_cost=0.37 * costs.at["central coal CHP", "VOM"],  # NB: VOM is per MWel
                capital_cost=0.37
                * costs.at["central coal CHP", "capital_cost"],  # NB: fixed cost is per MWel,
                p_nom=capacity / 0.37,
                p_nom_min=capacity / 0.37,
                p_nom_extendable=False,
                efficiency=0.37,
                p_nom_ratio=1.0,
                c_b=0.75,
                build_year=grouping_year,
                lifetime=costs.at["central coal CHP", "lifetime"],
            )
            n.add(
                "Link",
                capacity.index,
                suffix=" " + generator + " boiler" + "-" + str(grouping_year),
                bus0=bus0,
                bus1=capacity.index + " central heat",
                carrier=carrier[generator],
                marginal_cost=0.37 * costs.at["central coal CHP", "VOM"],  # NB: VOM is per MWel
                p_nom=capacity / 0.37 * 0.15,
                p_nom_min=capacity / 0.37 * 0.15,
                p_nom_extendable=False,
                efficiency=0.37 / 0.15,
                build_year=grouping_year,
                lifetime=costs.at["central coal CHP", "lifetime"],
            )

        if generator == "CHP gas":
            bus0 = capacity.index + " gas"
            n.add(
                "Link",
                capacity.index,
                suffix=" " + generator + " generator" + "-" + str(grouping_year),
                bus0=bus0,
                bus1=capacity.index,
                carrier=carrier[generator],
                marginal_cost=costs.at["central gas CHP", "efficiency"]
                * costs.at["central gas CHP", "VOM"],  # NB: VOM is per MWel
                capital_cost=costs.at["central gas CHP", "efficiency"]
                * costs.at["central gas CHP", "capital_cost"],  # NB: fixed cost is per MWel,
                p_nom=capacity / costs.at["central gas CHP", "efficiency"],
                p_nom_min=capacity / costs.at["central gas CHP", "efficiency"],
                p_nom_extendable=False,
                efficiency=costs.at["central gas CHP", "efficiency"],
                p_nom_ratio=1.0,
                c_b=costs.at["central gas CHP", "c_b"],
                build_year=grouping_year,
                lifetime=costs.at["central gas CHP", "lifetime"],
            )
            n.add(
                "Link",
                capacity.index,
                suffix=" " + generator + " boiler" + "-" + str(grouping_year),
                bus0=bus0,
                bus1=capacity.index + " central heat",
                carrier=carrier[generator],
                marginal_cost=costs.at["central gas CHP", "efficiency"]
                * costs.at["central gas CHP", "VOM"],  # NB: VOM is per MWel
                p_nom=capacity
                / costs.at["central gas CHP", "efficiency"]
                * costs.at["central gas CHP", "c_v"],
                p_nom_min=capacity
                / costs.at["central gas CHP", "efficiency"]
                * costs.at["central gas CHP", "c_v"],
                p_nom_extendable=False,
                efficiency=costs.at["central gas CHP", "efficiency"]
                / costs.at["central gas CHP", "c_v"],
                build_year=grouping_year,
                lifetime=costs.at["central gas CHP", "lifetime"],
            )

        if generator == "OCGT":
            bus0 = capacity.index + " gas"
            n.add(
                "Link",
                capacity.index,
                suffix=" " + generator + "-" + str(grouping_year),
                bus0=bus0,
                bus1=capacity.index,
                carrier=carrier[generator],
                marginal_cost=costs.at[generator, "efficiency"]
                * costs.at[generator, "VOM"],  # NB: VOM is per MWel
                capital_cost=costs.at[generator, "efficiency"]
                * costs.at[generator, "capital_cost"],
                # NB: fixed cost is per MWel
                p_nom=capacity / costs.at[generator, "efficiency"],
                p_nom_min=capacity / costs.at[generator, "efficiency"],
                p_nom_extendable=False,
                efficiency=costs.at[generator, "efficiency"],
                build_year=grouping_year,
                lifetime=costs.at[generator, "lifetime"],
            )

        if generator == "coal boiler":
            bus0 = capacity.index + " coal"
            for cat in [" central "]:
                n.add(
                    "Link",
                    capacity.index,
                    suffix="" + cat + generator + "-" + str(grouping_year),
                    bus0=bus0,
                    bus1=capacity.index + cat + "heat",
                    carrier=carrier[generator],
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
                )
        # TODO fix centralise
        if generator == "ground heat pump":
            date_range = pd.date_range(
                "2025-01-01 00:00",
                "2025-12-31 23:00",
                freq=config["snapshots"]["freq"],
                tz="Asia/shanghai",
            )
            date_range = date_range.map(lambda t: t.replace(year=2020))

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
                suffix=" " + generator + "-" + str(grouping_year),
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
            )


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "add_existing_baseyear",
            topology="current+FCG",
            co2_pathway="exp175default",
            planning_horizons="2020",
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
    df = assign_year_bins(existing_capacities, year_bins)
    # add to the network

    add_power_capacities_installed_before_baseyear(n, year_bins, costs, baseyear, config, df)
    n.export_to_netcdf(snakemake.output[0])

    logger.info("Existing capacities successfully added to network")
