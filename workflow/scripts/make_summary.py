# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

"""
Create summary CSV files for all scenario runs including costs, capacities,
capacity factors, curtailment, energy balances, prices and other metrics.
"""
import os
import sys
import logging

import pandas as pd
import numpy as np
import pypsa

from _helpers import mock_snakemake, configure_logging
from _pypsa_helpers import assign_locations
from _pypsa_helpers import calc_lcoe


logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
idx = pd.IndexSlice

opt_name = {"Store": "e", "Line": "s", "Transformer": "s"}


def assign_carriers(n: pypsa.Network):
    """Assign AC where missing
    Args:
        n (pypsa.Network): the network object to fix"""
    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"


# TODO swith to stats backend
def calculate_nodal_cfs(n: pypsa.Network, label: str, nodal_cfs: pd.DataFrame):
    """Calculate the capacity factors by for each node and genertor
    Args:
        n (pypsa.Network): the network object
        label (str): the label used by make summaries
        nodal_cfs (pd.DataFrame): the cap fac dataframe to fill/update
    Returns:
        pd.DataFrame: updated nodal_cfs
    """
    # Beware this also has extraneous locations for country (e.g. biomass)
    # or continent-wide (e.g. fossil gas/oil) stuff
    for c in n.iterate_components(
        (n.branch_components ^ {"Line", "Transformer"})
        | n.controllable_one_port_components ^ {"Load", "StorageUnit"}
    ):
        capacities_c = c.df.groupby(["location", "carrier"])[
            opt_name.get(c.name, "p") + "_nom_opt"
        ].sum()

        if c.name == "Link":
            p = c.pnl.p0.abs().mean()
        elif c.name == "Generator":
            p = c.pnl.p.abs().mean()
        elif c.name == "Store":
            p = c.pnl.e.abs().mean()
        else:
            sys.exit()

        c.df["p"] = p
        p_c = c.df.groupby(["location", "carrier"])["p"].sum()
        cf_c = p_c / capacities_c

        index = pd.MultiIndex.from_tuples([(c.list_name,) + t for t in cf_c.index.to_list()])
        nodal_cfs = nodal_cfs.reindex(index.union(nodal_cfs.index))
        nodal_cfs.loc[index, label] = cf_c.values

    return nodal_cfs


def calculate_cfs(n: pypsa.Network, label: str, cfs: pd.DataFrame) -> pd.DataFrame:
    """Calculate the capacity factors by carrier

    Args:
        n (pypsa.Network): the network object
        label (str): the label used by make summaries
        cfs (pd.DataFrame): the dataframe to fill/update
    Returns:
        pd.DataFrame: updated cfs
    """
    for c in n.iterate_components(
        n.branch_components | n.controllable_one_port_components ^ {"Load", "StorageUnit"}
    ):
        capacities_c = c.df[opt_name.get(c.name, "p") + "_nom_opt"].groupby(c.df.carrier).sum()

        if c.name in ["Link", "Line", "Transformer"]:
            p = c.pnl.p0.abs().mean()
        elif c.name == "Store":
            p = c.pnl.e.abs().mean()
        else:
            p = c.pnl.p.abs().mean()

        p_c = p.groupby(c.df.carrier).sum()
        cf_c = p_c / capacities_c
        cf_c = pd.concat([cf_c], keys=[c.list_name])
        cfs = cfs.reindex(cf_c.index.union(cfs.index))
        cfs.loc[cf_c.index, label] = cf_c

    return cfs


def calculate_nodal_costs(n: pypsa.Network, label: str, nodal_costs: pd.DataFrame):
    """Calculate the costs by carrier and location
    Args:
        n (pypsa.Network): the network object
        label (str): the label used by make summaries
        nodal_costs (pd.DataFrame): the dataframe to fill/update
    Returns:
        pd.DataFrame: updated nodal_costs
    """
    # Beware this also has extraneous locations for country (e.g. biomass)
    #  or continent-wide (e.g. fossil gas/oil) stuff
    for c in n.iterate_components(
        n.branch_components | n.controllable_one_port_components ^ {"Load"}
    ):
        c.df["capital_costs"] = c.df.capital_cost * c.df[opt_name.get(c.name, "p") + "_nom_opt"]
        capital_costs = c.df.groupby(["location", "carrier"])["capital_costs"].sum()
        index = pd.MultiIndex.from_tuples(
            [(c.list_name, "capital") + t for t in capital_costs.index.to_list()]
        )
        nodal_costs = nodal_costs.reindex(index.union(nodal_costs.index))
        nodal_costs.loc[index, label] = capital_costs.values

        if c.name == "Link":
            p = c.pnl.p0.multiply(n.snapshot_weightings.generators, axis=0).sum()
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0)
            p_all[p_all < 0.0] = 0.0
            p = p_all.sum()
        else:
            p = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0).sum()

        # correct sequestration cost
        if c.name == "Store":
            items = c.df.index[(c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.0)]
            c.df.loc[items, "marginal_cost"] = -20.0

        c.df["marginal_costs"] = p * c.df.marginal_cost
        marginal_costs = c.df.groupby(["location", "carrier"])["marginal_costs"].sum()
        index = pd.MultiIndex.from_tuples(
            [(c.list_name, "marginal") + t for t in marginal_costs.index.to_list()]
        )
        nodal_costs = nodal_costs.reindex(index.union(nodal_costs.index))
        nodal_costs.loc[index, label] = marginal_costs.values

    return nodal_costs


def calculate_costs(n: pypsa.Network, label: str, costs: pd.DataFrame) -> pd.DataFrame:
    """Calculate the costs by carrier
    Args:
        n (pypsa.Network): the network object
        label (str): the label used by make summaries
        costs (pd.DataFrame): the dataframe to fill/update
    Returns:
        pd.DataFrame: updated costs
    """

    for c in n.iterate_components(
        n.branch_components | n.controllable_one_port_components ^ {"Load"}
    ):
        capital_costs = c.df.capital_cost * c.df[opt_name.get(c.name, "p") + "_nom_opt"]
        capital_costs_grouped = capital_costs.groupby(c.df.carrier).sum()

        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=["capital"])
        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=[c.list_name])

        costs = costs.reindex(capital_costs_grouped.index.union(costs.index))

        costs.loc[capital_costs_grouped.index, label] = capital_costs_grouped

        if c.name == "Link":
            p = c.pnl.p0.multiply(n.snapshot_weightings.generators, axis=0).sum()
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0)
            p_all[p_all < 0.0] = 0.0
            p = p_all.sum()
        else:
            p = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0).sum()

        # correct sequestration cost
        if c.name == "Store":
            items = c.df.index[(c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.0)]
            c.df.loc[items, "marginal_cost"] = -20.0

        marginal_costs = p * c.df.marginal_cost

        marginal_costs_grouped = marginal_costs.groupby(c.df.carrier).sum()

        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=["marginal"])
        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=[c.list_name])

        costs = costs.reindex(marginal_costs_grouped.index.union(costs.index))

        costs.loc[marginal_costs_grouped.index, label] = marginal_costs_grouped

    # TODO remove/see if needed, and if yes soft-code
    # add back in all hydro
    # costs.loc[("storage_units", "capital", "hydro"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="hydro", "p_nom"].sum()
    # costs.loc[("storage_units", "capital", "PHS"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="PHS", "p_nom"].sum()
    # costs.loc[("generators", "capital", "ror"),label] = (0.02)*3e6*n.generators.loc[n.generators.group=="ror", "p_nom"].sum()

    return costs


def calculate_nodal_lcoe(n: pypsa.Network, label: str, nodal_lcoe: pd.DataFrame):
    """Calculate LCOE by province and technology
    
    Args:
        n (pypsa.Network): the network object
        label (str): the label used by make summaries
        nodal_lcoe (pd.DataFrame): the dataframe to fill/update
    Returns:
        pd.DataFrame: updated nodal_lcoe
    """

    lcoe_data = calc_lcoe(n, groupby=["location", "carrier"])
    
    lcoe_series = lcoe_data["LCOE"]
    
    nodal_lcoe[label] = lcoe_series
    
    return nodal_lcoe


def calculate_nodal_capacities(
    n: pypsa.Network, label: str, nodal_capacities: pd.DataFrame
) -> pd.DataFrame:
    """Calculate the capacities by carrier and node

    Args:
        n (pypsa.Network): the network object
        label (str): the label used by make summaries
        nodal_capacities (pd.DataFrame): the dataframe to fill/update
    Returns:
        pd.DataFrame: updated nodal_capacities"""
    # Beware this also has extraneous locations for country (e.g. biomass) or continent-wide
    #  (e.g. fossil gas/oil) stuff

    # Filter out reversed links to avoid double-counting transmission capacity
    # Only include positive links since positive and reversed links have the same capacity
    positive_links_mask = n.links.index.str.contains("positive")

    # Create a temporary network with only positive links for capacity calculation
    n_temp = n.copy()
    reversed_links = n.links.index[~positive_links_mask]
    n_temp.links = n_temp.links.drop(reversed_links)

    nodal_cap = n_temp.statistics.optimal_capacity(groupby=pypsa.statistics.get_bus_and_carrier)
    nodal_capacities[label] = nodal_cap.sort_index(level=0)
    return nodal_capacities


def calculate_capacities(
    n: pypsa.Network, label: str, capacities: pd.DataFrame, adjust_link_capacities=None
) -> pd.DataFrame:
    """Calculate the optimal capacities by carrier and bus carrier

    For links that connect to AC buses (bus1=AC), the capacity can be multiplied by efficiency
    to report the actual capacity available at the AC side rather than the input side.
    This ensures consistent capacity reporting across the network.

    Args:
        n (pypsa.Network): the network object
        label (str): the label used by make summaries
        capacities (pd.DataFrame): the dataframe to fill/update
        adjust_link_capacities (bool, optional): Whether to adjust link capacities by efficiency.
            If None, reads from config. Defaults to None.

    Returns:
        pd.DataFrame: updated capacities
    """

    # Temporarily save original link capacities
    original_p_nom_opt = n.links.p_nom_opt.copy()

    # Drop reversed links & report AC capacities for links from X to AC
    if adjust_link_capacities:

        # For links where bus1 is AC, multiply capacity by efficiency coefficient to get AC side capacity
        ac_links = n.links[n.links.bus1.map(n.buses.carrier) == "AC"].index
        n.links.loc[ac_links, "p_nom_opt"] *= n.links.loc[ac_links, "efficiency"]

        # ignore lossy link dummies
        pseudo_links = n.links.query("Link.str.contains('reversed') & capital_cost ==0 ").index
        n.links.loc[pseudo_links, "p_nom_opt"] = 0
    # Calculate optimal capacity using default grouper
    caps = n.statistics.optimal_capacity(
        groupby=pypsa.statistics.get_carrier_and_bus_carrier, nice_names=False
    )

    # Restore original link capacities to avoid modifying the network object
    n.links.p_nom_opt = original_p_nom_opt

    if "load shedding" in caps.index.get_level_values(1):
        caps.drop("load shedding", level=1, inplace=True)
    caps.rename(index={"AC": "Transmission Lines"}, inplace=True, level=1)

    # track links that feed into AC
    mask = (n.links.bus1.map(n.buses.carrier) == "AC") & (n.links.carrier != "stations")
    to_ac = n.links.loc[mask, "carrier"].unique()

    caps_df = caps.reset_index()
    ac_mask = caps_df["carrier"].isin(to_ac)
    caps_df.loc[ac_mask, "end_carrier"] = "AC"
    caps = caps_df.fillna("-").set_index(["component", "carrier", "bus_carrier", "end_carrier"])[0]

    capacities[label] = caps.sort_index(level=0)
    return capacities


def calculate_expanded_capacities(
    n: pypsa.Network, label: str, capacities: pd.DataFrame
) -> pd.DataFrame:
    """calculate the capacities by carrier

    Args:
        n (pypsa.Network): the network object
        label (str): the label used by make summaries
        capacities (pd.DataFrame): the dataframe to fill

    Returns:
        pd.Dataframe: updated capacities (bad style)
    """
    caps = n.statistics.expanded_capacity(
        groupby=pypsa.statistics.get_carrier_and_bus_carrier, nice_names=False
    )

    if "load shedding" in caps.index.get_level_values(1):
        caps.drop("load shedding", level=1, inplace=True)

    caps.rename(index={"AC": "Transmission Lines"}, inplace=True, level=1)

    # track links that feed into AC
    mask = (n.links.bus1.map(n.buses.carrier) == "AC") & (n.links.carrier != "stations")
    to_ac = n.links.loc[mask, "carrier"].unique()

    caps_df = caps.reset_index()
    ac_mask = caps_df["carrier"].isin(to_ac)
    caps_df.loc[ac_mask, "end_carrier"] = "AC"
    caps = caps_df.fillna("-").set_index(["component", "carrier", "bus_carrier", "end_carrier"])[0]

    capacities[label] = caps.sort_index(level=0)
    return capacities


def calculate_co2_balance(
    n: pypsa.Network,
    label: str,
    co2_balance: pd.DataFrame,
    withdrawal_stores=["CO2 capture"],
) -> pd.DataFrame:
    """calc the co2 balance [DOES NOT INCLUDE EMISSION GENERATING LINKSs]
    Args:
        n (pypsa.Network): the network object
        withdrawal_stores (list, optional): names of stores. Defaults to ["CO2 capture"].
        label (str): the label for the column
        co2_balance (pd.DataFrame): the df to update

    Returns:
       pd.DataFrame: updated co2_balance (bad style)
    """

    # year *(assumes one planning year intended),
    year = int(np.round(n.snapshots.year.values.mean(), 0))

    # emissions from generators (from fneumann course)
    emissions = (
        n.generators_t.p
        / n.generators.efficiency
        * n.generators.carrier.map(n.carriers.co2_emissions)
    )  # t/h
    emissions_carrier = (
        (n.snapshot_weightings.generators @ emissions).groupby(n.generators.carrier).sum()
    )

    # format and drop 0 values
    emissions_carrier = emissions_carrier.where(emissions_carrier > 0).dropna()
    emissions_carrier.rename(year, inplace=True)
    emissions_carrier = emissions_carrier.to_frame()
    # CO2 withdrawal
    stores = n.stores_t.e.T.groupby(n.stores.carrier).sum()
    co2_stores = stores.index.intersection(withdrawal_stores)
    co2_withdrawal = stores.iloc[:, -1].loc[co2_stores] * -1
    co2_withdrawal.rename(year, inplace=True)
    co2_withdrawal = co2_withdrawal.to_frame()
    year_balance = pd.concat([emissions_carrier, co2_withdrawal])

    #  combine with previous
    co2_balance = co2_balance.reindex(year_balance.index.union(co2_balance.index))
    co2_balance.loc[year_balance.index, label] = year_balance[year]

    return co2_balance


def calculate_curtailment(n: pypsa.Network, label: str, curtailment: pd.DataFrame) -> pd.DataFrame:
    """Calculate curtailed energy by carrier

    Args:
        n (pypsa.Network): the network object
        label (str): the label used by make summaries
        curtailment (pd.DataFrame): the dataframe to fill/update
    Returns:
        pd.DataFrame: updated curtailment
    """
    p_avail_by_carr = (
        n.generators_t.p_max_pu.multiply(n.generators.p_nom_opt)
        .sum()
        .groupby(n.generators.carrier)
        .sum()
    )
    used = n.generators_t.p.sum().groupby(n.generators.carrier).sum()

    curtailment[label] = (
        ((p_avail_by_carr - used).clip(0) / p_avail_by_carr).fillna(0) * 100
    ).round(3)

    return curtailment


# TODO what does this actually do? is it needed?
def calculate_energy(n: pypsa.Network, label: str, energy: pd.DataFrame) -> pd.DataFrame:
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        if c.name in n.one_port_components:
            c_energies = (
                c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0)
                .sum()
                .multiply(c.df.sign)
                .groupby(c.df.carrier)
                .sum()
            )
        else:
            c_energies = pd.Series(0.0, c.df.carrier.unique())
            for port in [col[3:] for col in c.df.columns if col[:3] == "bus"]:
                totals = c.pnl["p" + port].multiply(n.snapshot_weightings.generators, axis=0).sum()
                # remove values where bus is missing (bug in nomopyomo)
                no_bus = c.df.index[c.df["bus" + port] == ""]
                totals.loc[no_bus] = float(n.component_attrs[c.name].loc["p" + port, "default"])
                c_energies -= totals.groupby(c.df.carrier).sum()

        c_energies = pd.concat([c_energies], keys=[c.list_name])

        energy = energy.reindex(c_energies.index.union(energy.index))

        energy.loc[c_energies.index, label] = c_energies

    return energy


def calculate_peak_dispatch(n: pypsa.Network, label: str, supply: pd.DataFrame) -> pd.DataFrame:
    """Calculate the MAX dispatch of each component at the buses aggregated by
    carrier.

    Args:
        n (pypsa.Network): the network object
        label (str): the labe representing the pathway
        supply (pd.DataFrame): supply energy balance (empty df)

    Returns:
        pd.DataFrame: updated supply DF
    """

    sup_ = n.statistics.supply(
        groupby=pypsa.statistics.get_carrier_and_bus_carrier, aggregate_time="max"
    )
    supply_reordered = sup_.reorder_levels([2, 0, 1])
    supply_reordered.sort_index(inplace=True)
    supply[label] = supply_reordered

    return supply


def calculate_supply_energy(
    n: pypsa.Network, label: str, supply_energy: pd.DataFrame
) -> pd.DataFrame:
    """Calculate the total energy supply/consuption of each component at the buses
    aggregated by carrier.

    Args:
        n (pypsa.Network): the network object
        label (str): the labe representing the pathway
        supply_energy (pd.DataFrame): supply energy balance (empty df)

    Returns:
        pd.DataFrame: updated supply energy balance
    """

    eb = n.statistics.energy_balance(groupby=pypsa.statistics.get_carrier_and_bus_carrier)
    # fragile
    eb_reordered = eb.reorder_levels([2, 0, 1])
    eb_reordered.sort_index(inplace=True)
    eb_reordered.rename(index={"AC": "transmission losses"}, level=2, inplace=True)

    supply_energy[label] = eb_reordered

    return supply_energy


def calculate_metrics(n: pypsa.Network, label: str, metrics: pd.DataFrame):
    """LEGACY calculate a set of metrics for lines and co2
    Args:
        n (pypsa.Network): the network object
        label (str): the label to update the table row with
        metrics (pd.DataFrame): the dataframe to write to (not needed, refactor)
    Returns:
        pd.DataFrame: updated metrics"""

    metrics_list = [
        "line_volume",
        "line_volume_limit",
        "line_volume_AC",
        "line_volume_DC",
        "line_volume_shadow",
        "co2_shadow",
        "co2_budget",
    ]

    metrics = metrics.reindex(pd.Index(metrics_list).union(metrics.index))

    metrics.at["line_volume_DC", label] = (n.links.length * n.links.p_nom_opt)[
        n.links.carrier == "DC"
    ].sum()
    metrics.at["line_volume_AC", label] = (n.lines.length * n.lines.s_nom_opt).sum()
    metrics.at["line_volume", label] = metrics.loc[
        ["line_volume_AC", "line_volume_DC"], label
    ].sum()

    if "lv_limit" in n.global_constraints.index:
        metrics.at["line_volume_limit", label] = n.global_constraints.at["lv_limit", "constant"]
        metrics.at["line_volume_shadow", label] = n.global_constraints.at["lv_limit", "mu"]

    if "co2_limit" in n.global_constraints.index:
        metrics.at["co2_shadow", label] = n.global_constraints.at["co2_limit", "mu"]
        metrics.at["co2_budget", label] = n.global_constraints.at["co2_limit", "constant"]
    return metrics


def calculate_t_avgd_prices(n: pypsa.Network, label: str, prices: pd.DataFrame):
    """Time averaged prices for nodes averaged over carrier (bit silly?)

    Args:
        n (pypsa.Network): the network object
        label (str): the label representing the pathway (not needed, refactor)
        prices (pd.DataFrame): the dataframe to write to (not needed, refactor)
    Returns:
        pd.DataFrame: updated prices
    """
    prices = prices.reindex(prices.index.union(n.buses.carrier.unique()))

    # WARNING: this is time-averaged, see weighted_prices for load-weighted average
    prices[label] = n.buses_t.marginal_price.mean().groupby(n.buses.carrier).mean()

    return prices


def calculate_weighted_prices(
    n: pypsa.Network, label: str, weighted_prices: pd.DataFrame
) -> pd.DataFrame:
    """Demand-weighed prices for stores and loads.
        For stores if withdrawal is zero, use supply instead.
    Args:
        n (pypsa.Network): the network object
        label (str): the label representing the pathway (not needed, refactor)
        weighted_prices (pd.DataFrame): the dataframe to write to (not needed, refactor)

    Returns:
        pd.DataFrame: updated weighted_prices
    """
    entries = pd.Index(["electricity", "heat", "H2", "CO2 capture", "gas", "biomass"])
    weighted_prices = weighted_prices.reindex(entries)

    # loads
    load_rev = -1 * n.statistics.revenue(comps="Load", groupby=pypsa.statistics.get_bus_carrier)
    prices = load_rev / n.statistics.withdrawal(
        comps="Load", groupby=pypsa.statistics.get_bus_carrier
    )
    prices.rename(index={"AC": "electricity"}, inplace=True)

    # stores
    w = n.statistics.withdrawal(comps="Store")
    # biomass stores have no withdrawal for some reason
    w[w == 0] = n.statistics.supply(comps="Store")[w == 0]
    store_rev = n.statistics.revenue(comps="Store")
    mask = store_rev > load_rev.sum() / 400  # remove small
    wp_stores = store_rev[mask] / w[mask]
    weighted_prices[label] = pd.concat([prices, wp_stores.rename({"stations": "reservoir inflow"})])
    return weighted_prices


def calculate_market_values(
    n: pypsa.Network, label: str, market_values: pd.DataFrame
) -> pd.DataFrame:
    """Calculate the market value of the generators and links
    Args:
        n (pypsa.Network): the network object
        label (str): the label representing the pathway
        market_values (pd.DataFrame): the dataframe to write to (not needed, refactor)
    Returns:
        pd.DataFrame: updated market_values
    """
    # Warning: doesn't include storage units

    carrier = "AC"

    buses = n.buses.index[n.buses.carrier == carrier]

    # === First do market value of generators  ===
    # === First do market value of generators  ===

    generators = n.generators.index[n.buses.loc[n.generators.bus, "carrier"] == carrier]

    techs = n.generators.loc[generators, "carrier"].value_counts().index

    market_values = market_values.reindex(market_values.index.union(techs))

    for tech in techs:
        gens = generators[n.generators.loc[generators, "carrier"] == tech]

        dispatch = (
            n.generators_t.p[gens]
            .groupby(n.generators.loc[gens, "bus"], axis=1)
            .sum()
            .reindex(columns=buses, fill_value=0.0)
        )

        revenue = dispatch * n.buses_t.marginal_price[buses]

        market_values.at[tech, label] = revenue.sum().sum() / dispatch.sum().sum()

    # === Now do market value of links  ===
    # === Now do market value of links  ===

    for i in ["0", "1"]:
        carrier_links = n.links[n.links["bus" + i].isin(buses)].index

        techs = n.links.loc[carrier_links, "carrier"].value_counts().index

        market_values = market_values.reindex(market_values.index.union(techs))

        for tech in techs:
            links = carrier_links[n.links.loc[carrier_links, "carrier"] == tech]

            dispatch = (
                n.links_t["p" + i][links]
                .groupby(n.links.loc[links, "bus" + i], axis=1)
                .sum()
                .reindex(columns=buses, fill_value=0.0)
            )

            revenue = dispatch * n.buses_t.marginal_price[buses]

            market_values.at[tech, label] = revenue.sum().sum() / dispatch.sum().sum()

    return market_values


# TODO improve netwroks_dict arg
def make_summaries(
    networks_dict: dict[tuple, os.PathLike], opts: dict = None
) -> dict[str, pd.DataFrame]:
    """Make summary tables for the given network
    Args:
        networks_dict (dict): a dictionary of (pathway, time):network_path used in the run
        opts (dict): options for each summary function
    Returns:
        dict: a dictionary of dataframes with the summary tables

    """
    if opts is None:
        opts = {}

    output_funcs = {
        "nodal_costs": calculate_nodal_costs,
        "nodal_capacities": calculate_nodal_capacities,
        "nodal_cfs": calculate_nodal_cfs,
        "nodal_lcoe": calculate_nodal_lcoe,
        "cfs": calculate_cfs,
        "costs": calculate_costs,
        "co2_balance": calculate_co2_balance,
        "capacities": calculate_capacities,
        "capacities_expanded": calculate_expanded_capacities,
        "curtailment_pc": calculate_curtailment,
        "peak_dispatch": calculate_peak_dispatch,
        # "energy": calculate_energy,
        "supply_energy": calculate_supply_energy,
        "time_averaged_prices": calculate_t_avgd_prices,
        "weighted_prices": calculate_weighted_prices,
        # "price_statistics": calculate_price_statistics,
        "market_values": calculate_market_values,
        "metrics": calculate_metrics,
    }

    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(), names=["co2_pathway", "planning_horizons"]
    )
    dataframes_dict = {}

    # TO DO: not needed, could be made by the functions
    for output in output_funcs.keys():
        dataframes_dict[output] = pd.DataFrame(columns=columns, dtype=float)

    for label, filename in networks_dict.items():
        logger.info(f"Make summary for scenario {label}, using {filename}")

        n = pypsa.Network(filename)
        assign_carriers(n)
        assign_locations(n)

        for output, output_fn in output_funcs.items():
            if output in opts:
                dataframes_dict[output] = output_fn(
                    n, label, dataframes_dict[output], **opts[output]
                )
            else:
                dataframes_dict[output] = output_fn(n, label, dataframes_dict[output])

    return dataframes_dict


# TODO move to helper?
def expand_from_wildcard(key, config) -> list:
    """return a list of values for the given key in the config file
    Args:
        key (str): the key to look for in the config file
        config (dict): the config file
    Returns:
        list: a list of values for the given key
    """
    w = getattr(wildcards, key)
    return config["scenario"][key] if w == "all" else [w]


if __name__ == "__main__":
    if "snakemake" not in globals():

        snakemake = mock_snakemake(
            "make_summary",
            topology="current+FCG",
            # co2_pathway="exp175default",
            planning_horizons="2050",
            co2_pathway="SSP2-PkBudg1000-CHA-pypsaelh2",
            # heating_demand="positive",
            configfiles=["resources/tmp/remind_coupled_cg.yaml"],
        )

    configure_logging(snakemake)

    config = snakemake.config
    wildcards = snakemake.wildcards

    # The original was intended to handle a list of files
    # this doesnt r eflect the current snakefile
    # previous code was hardcoded and could cause issue w snakefile input

    # To go back to the all in one, would need to parse the file list
    # or hope that snamek wildcards are ordered in a sensible way

    # == here would be the way to get the list of possible wildcards
    pathways = expand_from_wildcard("co2_pathway", config)
    years = expand_from_wildcard("planning_horizons", config)

    if len(pathways) != 1 or len(years) != 1:
        raise ValueError("Multi file mode not implemented for summary")
    else:
        pathway, planning_horizons = pathways[0], years[0]

    networks_dict = {(pathway, planning_horizons): snakemake.input.network}

    # Access snakemake config only in main
    reporting_cfg = snakemake.config.get("reporting", {})
    summary_cfg = {
        "capacities": {
            "adjust_link_capacities": reporting_cfg.get(
                "adjust_link_capacities_by_efficiency", False
            )
        }
    }

    df = make_summaries(networks_dict, opts=summary_cfg)
    df["metrics"].loc["total costs"] = df["costs"].sum()

    def to_csv(dfs, dir):
        os.makedirs(dir, exist_ok=True)
        for key, df in dfs.items():
            df.to_csv(os.path.join(dir, f"{key}.csv"))

    to_csv(df, snakemake.output[0])

    logger.info(f"Made summary for {planning_horizons} in {pathway}")
