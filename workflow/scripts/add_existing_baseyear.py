# coding: utf-8
"""
Functions to add brownfield capacities to the network for a reference year
- adds VREs per grade and corrects technical potential. Best available grade is chosen
"""
# SPDX-FileCopyrightText: : 2025 The PyPSA-China-PIK Authors
#
# SPDX-License-Identifier: MIT


import logging
import numpy as np
import pandas as pd
import pypsa

import re
from types import SimpleNamespace

from constants import YEAR_HRS
from add_electricity import load_costs
from _helpers import mock_snakemake, configure_logging, ConfigManager
from _pypsa_helpers import shift_profile_to_planning_year

# TODO possibly reimplement to have env separation
from rpycpl.technoecon_etl import to_list

logger = logging.getLogger(__name__)
idx = pd.IndexSlice
spatial = SimpleNamespace()


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


def add_base_year(n: pypsa.Network, plan_year: int):
    """Add base year to new builds

    Args:
        n (pypsa.Network): the network
        plan_year (int): the plan year
    """

    for component in ["links", "generators"]:
        comp = getattr(n, component)
        mask = comp.query("p_nom_extendable==True").index
        comp.loc[mask, "build_year"] = plan_year


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

        if res_capacities:
            res_capacities = pd.concat(res_capacities, axis=0)

            for year in df.columns:
                for gen in res_capacities.index:
                    bus_bin = re.sub(f" {carrier}.*", "", gen)
                    bus, bin_id = bus_bin.rsplit(" ", maxsplit=1)
                    name = f"{bus_bin} {carrier}-{int(year)}"
                    capacity = res_capacities.loc[gen, year]
                    if capacity > 0.0:
                        cost_key = carrier.split("-", maxsplit=1)[0]
                        df_agg.at[name, "Fueltype"] = carrier
                        df_agg.at[name, "Capacity"] = capacity
                        df_agg.at[name, "DateIn"] = int(year)
                        df_agg.at[name, "grouping_year"] = int(year)
                        df_agg.at[name, "lifetime"] = costs.at[cost_key, "lifetime"]
                        df_agg.at[name, "DateOut"] = year + costs.at[cost_key, "lifetime"] - 1
                        df_agg.at[name, "bus"] = bus
                        df_agg.at[name, "resource_class"] = bin_id

    if df_agg.empty:
        return df_agg

    df_agg.loc[:, "Tech"] = df_agg.Fueltype
    return df_agg


def add_power_capacities_installed_before_baseyear(
    n: pypsa.Network,
    costs: pd.DataFrame,
    config: dict,
    installed_capacities: pd.DataFrame,
    eff_penalty_hist=0.0,
):
    """
    Add existing power capacities to the network.
    Note: hydro dams brownfield handled by prepare_network

    Args:
        n (pypsa.Network): the network
        costs (pd.DataFrame): techno-economic data
        config (dict): configuration dictionary
        installed_capacities (pd.DataFrame): installed capacities in MW
        eff_penalty_hist (float): efficiency penalty for historical plants (1-x)*current
    """

    logger.info("adding power capacities installed before baseyear")

    df = installed_capacities.copy()
    # fix fuel type CHP order to match network
    df["tech_clean"] = df["Fueltype"].str.replace(r"^CHP (.+)$", r"\1 CHP", regex=True)
    df["tech_clean"] = df["tech_clean"].str.replace("central ", "")
    df["tech_clean"] = df["tech_clean"].str.replace("decentral ", "")

    # TODO fix this based on config / centralise / other
    carrier_map = {
        "coal": "coal",
        "coal power plant": "coal",
        "CHP coal": "CHP coal",
        "coal CHP": "CHP coal",
        "CHP gas": "CHP gas",
        "gas CHP": "CHP gas",
        "gas OCGT": "gas OCGT",
        "gas CCGT": "gas CCGT",
        "solar": "solar",
        "solar thermal": "solar thermal",
        "onwind": "onwind",
        "offwind": "offwind",
        "coal boiler": "coal boiler central",
        "ground-sourced heat pump": "heat pump",
        "ground heat pump": "heat pump",
        "air heat pump": "heat pump",
        "nuclear": "nuclear",
        "PHS": "PHS",
    }
    costs_map = {
        "coal power plant": "coal",
        "coal CHP": "central coal CHP",
        "gas CHP": "central gas CHP",
        "gas OCGT": "OCGT",
        "gas CCGT": "CCGT",
        "solar": "solar",
        "solar thermal": "central solar thermal",
        "onwind": "onwind",
        "offwind": "offwind",
        "coal boiler": "central coal boiler",
        "heat pump": "central ground-sourced heat pump",
        "ground-sourced heat pump": "central ground-sourced heat pump",
        "nuclear": "nuclear",
    }

    # add techs that may have a direct match to the technoecon data
    missing_techs = {k: k for k in df.Fueltype.unique() if k not in costs_map}
    costs_map.update(missing_techs)

    if "resource_class" not in df.columns:
        df["resource_class"] = ""
    else:
        df.resource_class.fillna("", inplace=True)
    logger.info(df.grouping_year.unique())
    df.grouping_year = df.grouping_year.astype(int, errors="ignore")
    # TODO: exclude collapse of coal & coal CHP IF CCS retrofitting is enabled
    if config["existing_capacities"].get("collapse_years", False):
        df.grouping_year = "brownfield"

    df_ = df.pivot_table(
        index=["grouping_year", "tech_clean", "resource_class"],
        columns="bus",
        values="Capacity",
        aggfunc="sum",
    )

    df_.fillna(0, inplace=True)

    defined_carriers = n.carriers.index.unique().to_list()
    vre_carriers = ["solar", "onwind", "offwind"]

    # TODO do we really need to loop over the years? / so many things?
    # something like df_.unstack(level=0) would be more efficient
    for grouping_year, generator, resource_grade in df_.index:
        build_year = 0 if grouping_year == "brownfield" else grouping_year

        logger.info(f"Adding existing generator {generator} with year grp {grouping_year}")
        if not carrier_map.get(generator, "missing") in defined_carriers:
            logger.warning(
                f"Carrier {carrier_map.get(generator, None)} for {generator} not defined in network"
                "Consider adding to the CARRIER_MAP"
            )
        elif costs_map.get(generator) is None:
            raise ValueError(f"{generator} not defined in technoecon map - check costs_map")

        # capacity is the capacity in MW at each node for this
        capacity = df_.loc[grouping_year, generator, resource_grade]
        if capacity.values.max() == 0:
            continue
        capacity = capacity[capacity > config["existing_capacities"]["threshold_capacity"]].dropna()
        buses = capacity.index
        # fix index for network.add (merge grade to name)
        if resource_grade:
            capacity.index += " " + resource_grade
        capacity.index += " " + costs_map[generator]

        costs_key = costs_map[generator]

        if generator in vre_carriers:
            mask = n.generators_t.p_max_pu.columns.map(n.generators.carrier) == generator
            p_max_pu = n.generators_t.p_max_pu.loc[:, mask]
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
                p_max_pu=p_max_pu[capacity.index],
                build_year=build_year,
                lifetime=costs.at[costs_key, "lifetime"],
                location=buses,
            )

        elif generator in ["nuclear", "coal power plant", "biomass", "oil"]:
            n.add(
                "Generator",
                capacity.index,
                suffix="-" + str(grouping_year),
                bus=buses,
                carrier=carrier_map[generator],
                p_nom=capacity,
                p_nom_max=capacity,
                p_nom_min=capacity,
                p_nom_extendable=False,
                p_max_pu=config["nuclear_reactors"]["p_max_pu"] if generator == "nuclear" else 1,
                p_min_pu=config["nuclear_reactors"]["p_min_pu"] if generator == "nuclear" else 0,
                marginal_cost=costs.at[costs_key, "marginal_cost"],
                efficiency=costs.at[costs_key, "efficiency"] * (1 - eff_penalty_hist),
                build_year=build_year,
                lifetime=costs.at[costs_key, "lifetime"],
                location=buses,
            )

        # TODO this does not add the carrier to the list
        elif generator in ["gas CCGT", "gas OCGT"]:
            bus0 = buses + " gas"
            carrier_ = carrier_map[generator]
            # ugly fix to register the carrier. Emissions for sub carrier are 0: they are accounted for at gas bus
            n.carriers.loc[carrier_] = {
                "co2_emissions": 0,
                "color": config["plotting"]["tech_colors"][carrier_],
                "nice_name": config["plotting"]["nice_names"][carrier_],
                "max_growth": np.inf,
                "max_relative_growth": 0,
            }
            # now add link - carrier should exist
            n.add(
                "Link",
                capacity.index,
                suffix="-" + str(grouping_year),
                bus0=bus0,
                bus1=buses,
                carrier=carrier_map[generator],
                marginal_cost=costs.at[costs_key, "efficiency"]
                * costs.at[costs_key, "VOM"],  # NB: VOM is per MWel
                # NB: fixed cost is per MWel
                p_nom=capacity / costs.at[costs_key, "efficiency"],
                p_nom_min=capacity / costs.at[costs_key, "efficiency"],
                p_nom_max=capacity / costs.at[costs_key, "efficiency"],
                p_nom_extendable=False,
                efficiency=costs.at[costs_key, "efficiency"] * (1 - eff_penalty_hist),
                build_year=build_year,
                lifetime=costs.at[costs_key, "lifetime"],
                location=buses,
            )
        elif generator in [
            "solar thermal",
            "CHP coal",
            "CHP gas",
            "heat pump",
            "coal boiler",
        ] and not config.get("heat_coupling", False):
            logger.info(f"Skipped {generator} because heat coupling is not activated")

        elif generator == "solar thermal":
            p_max_pu = n.generators_t.p_max_pu[capacity.index]
            p_max_pu.columns = capacity.index
            n.add(
                "Generator",
                capacity.index,
                suffix=f"-{str(grouping_year)}",
                bus=buses + " central heat",
                carrier=carrier_map[generator],
                p_nom=capacity,
                p_nom_min=capacity,
                p_nom_max=capacity,
                p_nom_extendable=False,
                marginal_cost=costs.at["central " + generator, "marginal_cost"],
                p_max_pu=p_max_pu,
                build_year=build_year,
                lifetime=costs.at["central " + generator, "lifetime"],
                location=buses,
            )

        elif generator in ["CHP coal", "coal CHP"]:
            bus0 = buses + " coal fuel"
            n.add(
                "Link",
                capacity.index,
                suffix=f" generator-{str(grouping_year)}",
                bus0=bus0,
                bus1=buses,
                carrier=carrier_map[generator],
                marginal_cost=costs.at["central coal CHP", "efficiency"]
                * costs.at["central coal CHP", "VOM"],  # NB: VOM is per MWel
                p_nom=capacity / costs.at["central coal CHP", "efficiency"],
                p_nom_min=capacity / costs.at["central coal CHP", "efficiency"],
                p_nom_max=capacity / costs.at["central coal CHP", "efficiency"],
                p_nom_extendable=False,
                efficiency=costs.at["central coal CHP", "efficiency"] * (1 - eff_penalty_hist),
                heat_to_power=config["chp_parameters"]["coal"]["heat_to_power"],
                build_year=build_year,
                lifetime=costs.at["central coal CHP", "lifetime"],
                location=buses,
            )
            # simplified treatment based on a decrease with c_v and a max htpwr ratio
            htpr = config["chp_parameters"]["coal"]["heat_to_power"]

            n.add(
                "Link",
                capacity.index,
                suffix=f" boiler-{str(grouping_year)}",
                bus0=bus0,
                bus1=buses + " central heat",
                carrier=carrier_map[generator],
                marginal_cost=costs.at["central coal CHP", "efficiency"]
                * costs.at["central coal CHP", "VOM"],  # NB: VOM is per MWel
                # p_max will be constrained by chp constraints
                p_nom=capacity * htpr,
                p_nom_min=capacity * htpr,
                p_nom_max=capacity * htpr,
                p_nom_extendable=False,
                # total eff will be fixed by CHP constraints
                efficiency=config["chp_parameters"]["coal"]["total_eff"],
                build_year=build_year,
                lifetime=costs.at["central coal CHP", "lifetime"],
                location=buses,
            )

        elif generator in ["CHP gas", "gas CHP"]:
            bus0 = buses + " gas"
            n.add(
                "Link",
                capacity.index,
                suffix=f" generator-{str(grouping_year)}",
                bus0=bus0,
                bus1=buses,
                carrier=carrier_map[generator],
                marginal_cost=costs.at["central gas CHP CC", "efficiency"]
                * costs.at["central gas CHP CC", "VOM"],  # NB: VOM is per MWel
                capital_cost=costs.at["central gas CHP CC", "efficiency"]
                * costs.at["central gas CHP CC", "capital_cost"],  # NB: fixed cost is per MWel,
                p_nom=capacity / costs.at["central gas CHP CC", "efficiency"],
                p_nom_min=capacity / costs.at["central gas CHP CC", "efficiency"],
                p_nom_extendable=False,
                efficiency=costs.at["central gas CHP CC", "efficiency"] * (1 - eff_penalty_hist),
                heat_to_power=config["chp_parameters"]["gas"]["heat_to_power"],
                c_b=costs.at["central gas CHP CC", "c_b"],
                build_year=build_year,
                lifetime=costs.at["central gas CHP CC", "lifetime"],
                location=buses,
            )
            # simplified treatment based on a decrease with c_v and a max htpwr ratio
            htpr = config["chp_parameters"]["gas"]["heat_to_power"]

            n.add(
                "Link",
                capacity.index,
                suffix=f" boiler-{str(grouping_year)}",
                bus0=bus0,
                bus1=buses + " central heat",
                carrier=carrier_map[generator],
                marginal_cost=costs.at["central gas CHP CC", "efficiency"]
                * costs.at["central gas CHP CC", "VOM"],  # NB: VOM is per MWel
                # pmax will be constrained by chp constraints
                p_nom=capacity * htpr,
                p_nom_min=capacity * htpr,
                p_nom_max=capacity * htpr,
                p_nom_extendable=False,
                # will be constrained by chp constraints
                efficiency=config["chp_parameters"]["gas"]["total_eff"],
                build_year=build_year,
                lifetime=costs.at["central gas CHP CC", "lifetime"],
                location=buses,
            )

        elif generator.find("coal boiler") != -1:
            bus0 = buses + " coal"
            cat = "central" if generator.find("decentral") == -1 else "decentral"
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
                p_nom_max=capacity / costs.at[cat.lstrip() + generator, "efficiency"],
                p_nom_extendable=False,
                efficiency=costs.at[cat.lstrip() + generator, "efficiency"],
                build_year=build_year,
                lifetime=costs.at[cat.lstrip() + generator, "lifetime"],
                location=buses,
            )

        # TODO fix read operation in func, fix snakemake in function, make air pumps?
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
                p_nom_max=capacity / costs.at["decentral ground-sourced heat pump", "efficiency"],
                p_nom_extendable=False,
                build_year=build_year,
                lifetime=costs.at["decentral ground-sourced heat pump", "lifetime"],
                location=buses,
            )

        elif generator == "PHS":
            # pure pumped hydro storage, fixed, 6h energy by default, no inflow
            n.add(
                "StorageUnit",
                capacity.index,
                suffix="-" + str(grouping_year),
                bus=buses,
                carrier="PHS",
                p_nom=capacity,
                p_nom_min=capacity,
                p_nom_max=capacity,
                p_nom_extendable=False,
                max_hours=config["hydro"]["PHS_max_hours"],
                efficiency_store=np.sqrt(costs.at["PHS", "efficiency"]),
                efficiency_dispatch=np.sqrt(costs.at["PHS", "efficiency"]),
                cyclic_state_of_charge=True,
                marginal_cost=0.0,
                location=buses,
            )

        else:
            logger.warning(
                f"Skipped existing capacitity for '{generator}'"
                + " - tech not implemented as existing capacity"
            )

    return


def _add_paidoff_biomass(
    n: pypsa.Network,
    costs: pd.DataFrame,
    paid_off_cap: float,
    tech_group: str = "biomass",
):
    """
    Add existing biomass in case no biomass was defined previously.
    Paid-off capacities can only be copied from existing generators

    Args:
        n (pypsa.Network): the network
        costs (pd.DataFrame): techno-economic data
        config (dict): configuration dictionary
        paid_off_cap (float): the paid-off biomass capacity in MW
        tech_group (str, optional): the remind technology group for biomass.
            Defaults to "biomass".
    """
    # 0 emissions
    n.add(
        "Generator",
        n.buses.query("carrier == 'AC'").index,
        suffix=" biomass paidoff",
        bus=n.buses.query("carrier == 'AC'").index,
        carrier="biomass",
        capital_cost=0,
        p_nom=0.0,
        p_nom_max=np.inf,
        p_nom_extendable=True,
        marginal_cost=costs.at["biomass", "marginal_cost"],
        efficiency=costs.at["biomass", "efficiency"],
        lifetime=costs.at["biomass", "lifetime"],
        location=n.buses.query("carrier == 'AC'").index,
        p_nom_max_rcl=paid_off_cap,
        tech_group=tech_group,
    )


def add_paid_off_capacity(
    network: pypsa.Network, paid_off_caps: pd.DataFrame, costs: pd.DataFrame, cutoff=100
):
    """
    Add capacities that have been paid off to the network. This is intended
    for REMIND coupling, where (some of) the REMIND investments can be freely allocated
    to the optimal node. NB: an additional constraing is needed to ensure that
    the capacity is not exceeded.

    Args:
        network (pypsa.Network): the network to which the capacities are added.
        paid_off_caps (pd.DataFrame): DataFrame with paid off capacities & columns
            [tech_group, Capacity, techs]
        costs (pd.DataFrame): techno-economic data for the technologies
        cutoff (int, optional): minimum capacity to be considered. Defaults to 100 MW."""

    paid_off = paid_off_caps.reset_index()

    # explode tech list per tech group (constraint will apply to group)
    paid_off.techs = paid_off.techs.apply(to_list)
    paid_off = paid_off.explode("techs")
    paid_off["carrier"] = paid_off.techs.str.replace("'", "")
    paid_off.set_index("carrier", inplace=True)
    # clip small capacities
    paid_off["p_nom_max"] = paid_off.Capacity.apply(lambda x: 0 if x < cutoff else x)
    paid_off.drop(columns=["Capacity", "techs"], inplace=True)
    paid_off = paid_off.query("p_nom_max > 0")

    component_settings = {
        "Generator": {
            "join_col": "carrier",
            "attrs_to_fix": ["p_min_pu", "p_max_pu"],
        },
        "Link": {
            "join_col": "carrier",
            "attrs_to_fix": ["p_min_pu", "p_max_pu", "efficiency", "efficiency2"],
        },
        "Store": {
            "join_col": "carrier",
            "attrs_to_fix": [],
        },
    }

    # TODO make a centralised setting or update cpl config
    rename_carriers = {"OCGT": "gas OCGT", "CCGT": "gas CCGT"}
    paid_off.rename(rename_carriers, inplace=True)

    for component, settings in component_settings.items():
        prefix = "e" if component == "Store" else "p"
        paid_off_comp = paid_off.rename(columns={"p_nom_max": f"{prefix}_nom_max"})

        # exclude brownfield capacities
        df = getattr(network, component.lower() + "s").query(f"{prefix}_nom_extendable == True")
        # join will add the tech_group and p_nom_max_rcl columns, used later for constraints
        # rcl is legacy name from Adrian for region country limit
        paid = df.join(paid_off_comp, on=[settings["join_col"]], how="right", rsuffix="_rcl")
        paid.dropna(subset=[f"{prefix}_nom_max", f"{prefix}_nom_max_rcl"], inplace=True)
        paid = paid.loc[paid.index.dropna()]
        if paid.empty:
            continue

        # REMIND cap is in output, PyPSA link in input
        if component == "Link":
            paid.loc[:, "p_nom_max_rcl"] /= paid.loc[:, "efficiency"]

        paid.index += "_paid_off"
        # set permissive options for the paid-off capacities (constraint per group added to model later)
        paid["capital_cost"] = 0
        paid[f"{prefix}_nom_min"] = 0.0
        paid[f"{prefix}_nom"] = 0.0
        paid[f"{prefix}_nom_max"] = np.inf

        # add to the network
        network.add(component, paid.index, **paid)
        # now add the dynamic attributes not carried over by n.add (per unit avail etc)
        for missing_attr in settings["attrs_to_fix"]:
            df_t = getattr(network, component.lower() + "s_t")[missing_attr]
            if not df_t.empty:
                base_cols = [
                    x for x in paid.index.str.replace("_paid_off", "") if x in df_t.columns
                ]
                df_t.loc[:, pd.Index(base_cols) + "_paid_off"] = df_t[base_cols].rename(
                    columns=lambda x: x + "_paid_off"
                )

    if "biomass" in paid_off.index and "biomass" not in network.generators.carrier.unique():
        _add_paidoff_biomass(
            network,
            costs,
            paid_off.loc["biomass", "p_nom_max"],
            tech_group=paid_off.loc["biomass", "tech_group"],
        )

    # TODO go through the pypsa-EUR fuel drops for the new ppmatching style


def filter_brownfield_capacities(existing_df: pd.DataFrame, plan_year: int) -> pd.DataFrame:
    """
    Filter brownfield capacities to remove retired/not yet built plants .
    Parameters:
        existing_df (pd.DataFrame): DataFrame containing asset information with at least the columns 'DateOut', 'DateIn', 'grouping_year', and 'cluster_bus'.
        plan_year (int): The year modelled year/horizon
    Returns:
        pd.DataFrame: The filtered and updated DataFrame.
    """

    # drop assets which are already phased out / decommissioned
    phased_out = existing_df[existing_df["DateOut"] < plan_year].index
    existing_df.drop(phased_out, inplace=True)

    to_drop = existing_df[existing_df.DateIn > plan_year].index
    existing_df.drop(to_drop, inplace=True)

    existing_df.rename(columns={"cluster_bus": "bus"}, inplace=True)

    return existing_df


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "add_existing_baseyear",
            topology="current+FCG",
            co2_pathway="exp175default",
            planning_horizons="2030",
            # configfiles="resources/tmp/pseudo_coupled.yml",
            heating_demand="positive",
        )

    configure_logging(snakemake, logger=logger)

    config = snakemake.config
    # TODO then collapse everything but coal
    if config["existing_capacities"].get("collapse_years", False) and config["Techs"].get(
        "coal_ccs_retrofit", False
    ):
        raise ValueError(
            "Incompatible configuration: collapse_years and coal_ccs_retrofit cannot be both enabled."
            " Retrofit requires the date information."
        )

    tech_costs = snakemake.input.tech_costs
    plan_year = int(snakemake.wildcards["planning_horizons"]) # plan_year]
    data_paths = {k: v for k, v in snakemake.input.items()}
    vre_techs = snakemake.params["vre_carriers"]

    n = pypsa.Network(snakemake.input.network)
    add_base_year(n, cost_year)
    n_years = n.snapshot_weightings.generators.sum() / YEAR_HRS

    costs = load_costs(tech_costs, config["costs"], config["electricity"], plan_year, n_years)

    existing_capacities = pd.read_csv(snakemake.input.installed_capacities, index_col=0)
    # Existing capacities is multi-year frame in remind coupled mode
    if config["run"].get("is_remind_coupled", False) or "year" in existing_capacities.columns:
        existing_capacities = existing_capacities.query("remind_year == @cost_year")
    existing_capacities = filter_capacities(existing_capacities, cost_year)

    vre_caps = existing_capacities.query("Tech in @vre_techs | Fueltype in @vre_techs")
    # vre_caps.loc[:, "Country"] = coco.CountryConverter().convert(["China"], to="iso2")
    vres = add_existing_vre_capacities(n, costs, vre_caps, config)
    # TODO: fix bug, installed has less vre/wind cap than vres.
    installed = pd.concat(
        [existing_capacities.query("Tech not in @vre_techs & Fueltype not in @vre_techs"), vres],
        axis=0,
    )

    # add to the network
    add_power_capacities_installed_before_baseyear(n, costs, config, installed)
    # add paid-off REMIND capacities if requested
    if config["run"].get("is_remind_coupled", False) & (
        data_paths.get("paid_off_capacities_remind", None) is not None
    ):
        logger.info("Adding paid-off REMIND capacities to the network")
        paid_off_caps = pd.read_csv(snakemake.input.paid_off_capacities_remind, index_col=0)
        yr = int(plan_year)
        paid_off_caps = paid_off_caps.query("year == @yr")
        # add to network
        add_paid_off_capacity(n, paid_off_caps, costs)

    compression = snakemake.config.get("io", None)
    if compression:
        compression = compression.get("nc_compression", None)
    n.export_to_netcdf(snakemake.output[0], compression=compression)

    logger.info("Existing capacities successfully added to network")
