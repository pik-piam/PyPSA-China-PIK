#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2024 The PyPSA-Eur Authors
# 2014 Adapted from pypsa-eur by PyPSA-China authors
#
# SPDX-License-Identifier: MIT

import matplotlib.pyplot as plt
import matplotlib.axes as axes
import pypsa
import seaborn as sns
import os
import logging
from pandas import DataFrame
import pandas as pd
import numpy as np

from _helpers import configure_logging, mock_snakemake, set_plot_test_backend
from _plot_utilities import rename_index, fix_network_names_colors, filter_carriers
from _pypsa_helpers import calc_lcoe
from constants import (
    PLOT_CAP_LABEL,
    PLOT_CAP_UNITS,
    PLOT_SUPPLY_UNITS,
    PLOT_SUPPLY_LABEL,
)

sns.set_theme("paper", style="whitegrid")
logger = logging.getLogger(__name__)


def plot_static_per_carrier(ds: DataFrame, ax: axes.Axes, colors: DataFrame, drop_zero_vals=True):
    """Generic function to plot different statics

    Args:
        ds (DataFrame): the data to plot
        ax (matplotlib.axes.Axes): plotting axes
        colors (DataFrame): colors for the carriers
        drop_zero_vals (bool, optional): Drop zeroes from data. Defaults to True.
    """
    if drop_zero_vals:
        ds = ds[ds != 0]
    ds = ds.dropna()
    logger.info("debuggin plot stat")
    logger.info(colors)
    c = colors[ds.index.get_level_values("carrier")]
    logger.info(c)
    logger.info(ds.index.get_level_values("carrier"))
    logger.info(colors.loc[ds.index.get_level_values("carrier")])
    ds = ds.pipe(rename_index)
    label = f"{ds.attrs['name']} [{ds.attrs['unit']}]"
    ds.plot.barh(color=c.values, xlabel=label, ax=ax)
    ax.grid(axis="y")


if __name__ == "__main__":
    if "snakemake" not in globals():

        snakemake = mock_snakemake(
            "plot_statistics",
            carrier="AC",
            planning_horizons="2025",
            # co2_pathway="exp175default",
            # planning_horizons="2130",
            co2_pathway="SSP2-PkBudg1000-CHA-higher_minwind_cf",
            topology="current+FCG",
            # heating_demand="positive",
            configfiles="resources/tmp/remind_coupled_cg.yaml",
        )
    configure_logging(snakemake)
    set_plot_test_backend(snakemake.config)

    carrier = snakemake.params.carrier

    n = pypsa.Network(snakemake.input.network)
    # # incase an old version need to add missing info to network
    fix_network_names_colors(n, snakemake.config)
    n.loads.carrier = "load"
    n.carriers.loc["load", ["nice_name", "color"]] = (
        "Load",
        snakemake.config["plotting"]["tech_colors"]["electric load"],
    )

    colors = n.carriers.set_index("nice_name").color.where(lambda s: s != "", "lightgrey")

    outp_dir = snakemake.output.stats_dir
    if not os.path.exists(outp_dir):
        os.makedirs(outp_dir)

    stats_list = snakemake.params.stat_types

    attached_carriers = filter_carriers(n, carrier)
    if "capacity_factor" in stats_list:
        fig, ax = plt.subplots()
        ds = n.statistics.capacity_factor(groupby=["carrier"], nice_names = False).dropna()
        # avoid grouping battery uif same name
        if ("Link", "battery") in ds.index:
            ds.loc[("Link", "battery charger")] = ds.loc[("Link", "battery")]
            ds.drop(index=("Link", "battery"), inplace=True)
        ds = ds.groupby(level=1).first()
        ds = ds.loc[ds.index.isin(attached_carriers)]
        ds.index = ds.index.map(lambda idx: n.carriers.loc[idx, "nice_name"])
        plot_static_per_carrier(ds, ax, colors=colors)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "capacity_factor.png"))

    if "installed_capacity" in stats_list:
        fig, ax = plt.subplots()
        ds = n.statistics.installed_capacity(groupby=["carrier"], nice_names=False).dropna()
        ds.drop("stations", level=1, inplace=True)
        ds = ds.groupby(level=1).sum()
        ds = ds.loc[ds.index.isin(attached_carriers)]
        ds.index = ds.index.map(lambda idx: n.carriers.loc[idx, "nice_name"])
        if "Line" in ds.index:
            ds = ds.drop("Line")
        ds = ds.drop(("Generator", "Load"), errors="ignore")
        ds = ds.abs() / PLOT_CAP_UNITS
        ds.attrs["unit"] = PLOT_CAP_LABEL
        plot_static_per_carrier(ds.abs(), ax, colors=colors)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "installed_capacity.png"))

    if "optimal_capacity" in stats_list:
        fig, ax = plt.subplots()

        # Temporarily save original link capacities
        original_p_nom_opt = n.links.p_nom_opt.copy()

        # Get configuration from snakemake
        adjust_link_capacities = snakemake.config.get("reporting", {}).get(
            "adjust_link_capacities_by_efficiency", False
        )

        # Drop reversed links & report AC capacities for links from X to AC
        if adjust_link_capacities:
            # For links where bus1 is AC, multiply capacity by efficiency coefficient to get AC side capacity
            ac_links = n.links[n.links.bus1.map(n.buses.carrier) == "AC"].index
            n.links.loc[ac_links, "p_nom_opt"] *= n.links.loc[ac_links, "efficiency"]

            # ignore lossy link dummies
            pseudo_links = n.links.query("Link.str.contains('reversed') & capital_cost ==0 ").index
            n.links.loc[pseudo_links, "p_nom_opt"] = 0

        # Calculate optimal capacity for all components
        ds = n.statistics.optimal_capacity(groupby=["carrier"], nice_names = False).dropna()

        # Restore original link capacities to avoid modifying the network object
        n.links.p_nom_opt = original_p_nom_opt

        # Handle battery components correctly
        if ("Link", "battery") in ds.index:
            ds.loc[("Link", "battery charger")] = ds.loc[("Link", "battery")]
            ds.drop(index=("Link", "battery"), inplace=True)
        ds.drop("stations", level=1, inplace=True)
        if "Load Shedding" in ds.index.get_level_values(1):
            ds.drop("Load Shedding", level=1, inplace=True)
        ds = ds.groupby(level=1).sum()
        ds = ds.loc[ds.index.isin(attached_carriers)]
        ds.index = ds.index.map(lambda idx: n.carriers.loc[idx, "nice_name"])
        if "Line" in ds.index:
            ds = ds.drop("Line")
        ds = ds.drop(("Generator", "Load"), errors="ignore")
        ds = ds.abs() / PLOT_CAP_UNITS
        ds.attrs["unit"] = PLOT_CAP_LABEL
        plot_static_per_carrier(ds, ax, colors=colors)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "optimal_capacity.png"))

    if "capital_expenditure" in stats_list:
        fig, ax = plt.subplots()
        ds = n.statistics.capex(groupby=["carrier"]).dropna()
        ds = ds.groupby(level=1).sum()
        ds = ds.loc[ds.index.isin(attached_carriers)]
        ds.index = ds.index.map(lambda idx: n.carriers.loc[idx, "nice_name"])
        plot_static_per_carrier(ds, ax, colors=colors)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "capex.png"))

    if "operational_expenditure" in stats_list:
        fig, ax = plt.subplots()
        ds = n.statistics.opex(groupby=["carrier"]).dropna()
        ds = ds.groupby(level=1).sum()
        ds = ds.loc[ds.index.isin(attached_carriers)]
        ds.index = ds.index.map(lambda idx: n.carriers.loc[idx, "nice_name"])
        plot_static_per_carrier(ds, ax, colors=colors)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "opex.png"))

    if "curtailment" in stats_list:
        fig, ax = plt.subplots()
        ds = n.statistics.curtailment(bus_carrier=carrier)
        # curtailment definition only makes sense for VREs
        vres = ['Offshore Wind', 'Onshore Wind', 'Solar', 'Solar Residential']
        vres = [v for v in vres if v in ds.index.get_level_values("carrier")]
        attrs = ds.attrs.copy()
        ds = ds.unstack()[vres].stack()
        ds.attrs = attrs
        plot_static_per_carrier(ds, ax, colors=colors)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "curtailment.png"))

    if "supply" in stats_list:
        fig, ax = plt.subplots()
        ds = n.statistics.supply(bus_carrier=carrier)
        if "Line" in ds.index:
            ds = ds.drop("Line")
        ds = ds / PLOT_SUPPLY_UNITS
        ds.attrs["unit"] = PLOT_SUPPLY_LABEL
        plot_static_per_carrier(ds, ax, colors=colors)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "supply.png"))

    if "withdrawal" in stats_list:
        fig, ax = plt.subplots()
        ds = n.statistics.withdrawal(bus_carrier=carrier)
        if "Line" in ds.index:
            ds = ds.drop("Line")
        ds = ds / PLOT_SUPPLY_UNITS
        ds.attrs["unit"] = PLOT_SUPPLY_LABEL
        plot_static_per_carrier(ds, ax, colors=colors)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "withdrawal.png"))

    if "market_value" in stats_list:
        fig, ax = plt.subplots()
        ds = n.statistics.market_value(bus_carrier=carrier)
        plot_static_per_carrier(ds, ax, colors=colors)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "market_value.png"))

    if "lcoe" in stats_list:
        rev_costs = calc_lcoe(n, groupby=None)
        ds = rev_costs["LCOE"]
        if "load shedding" in ds.index.get_level_values(1):
            ds.drop("load shedding", level=1, inplace=True)
        if "H2" in ds.index.get_level_values(1):
            ds.drop("H2", level=1, inplace=True)
        ds.attrs = {"name": "LCOE", "unit": "€/MWh"}
        fig, ax = plt.subplots()
        plot_static_per_carrier(ds, ax, colors=colors)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "LCOE.png"))

        rev_costs = calc_lcoe(n, groupby=None)
        ds = rev_costs["profit_pu"]
        ds.attrs = {"name": "MV - LCOE", "unit": "€/MWh"}
        fig, ax = plt.subplots()
        plot_static_per_carrier(ds, ax, colors=colors)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "MV_minus_LCOE.png"))
