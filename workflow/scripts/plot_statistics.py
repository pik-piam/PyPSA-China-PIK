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

from _helpers import configure_logging, mock_snakemake, set_plot_test_backend
from _plot_utilities import rename_index, fix_network_names_colors, filter_carriers
from _pypsa_helpers import calc_lcoe
from constants import PLOT_CAP_LABEL, PLOT_CAP_UNITS, PLOT_SUPPLY_UNITS, PLOT_SUPPLY_LABEL

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
            # planning_horizons="2055",
            # co2_pathway="exp175default",
            planning_horizons="2130",
            co2_pathway="remind_ssp2NPI",
            topology="current+FCG",
            heating_demand="positive",
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
        ds = n.statistics.capacity_factor(groupby=["carrier"]).dropna()
        ds = ds.groupby(level=1).sum()
        ds = ds.loc[ds.index.isin(attached_carriers)]
        ds.index = ds.index.map(lambda idx: n.carriers.loc[idx, "nice_name"])
        plot_static_per_carrier(ds, ax, colors=colors)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "capacity_factor.png"))

    if "installed_capacity" in stats_list:
        fig, ax = plt.subplots()
        ds = n.statistics.installed_capacity(groupby=["carrier"]).dropna()
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
        ds = n.statistics.optimal_capacity(groupby=["carrier"]).dropna()
        ds.drop("stations", level=1, inplace=True)
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
        attached_carriers = filter_carriers(n, carrier)
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
