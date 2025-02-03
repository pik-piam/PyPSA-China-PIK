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

from pandas import DataFrame

from _helpers import configure_logging, mock_snakemake
from _plot_utilities import rename_index, fix_network_names_colors
from constants import PLOT_CAP_LABEL, PLOT_CAP_UNITS, PLOT_SUPPLY_UNITS, PLOT_SUPPLY_LABEL

sns.set_theme("paper", style="whitegrid")


def plot_static_per_carrier(ds: DataFrame, ax: axes.Axes, drop_zero_vals=True):
    """Generic function to plot different statics

    Args:
        ds (DataFrame): the data to plot
        ax (matplotlib.axes.Axes): plotting axes
        drop_zero_vals (bool, optional): Drop zeroes from data. Defaults to True.
    """
    if drop_zero_vals:
        ds = ds[ds != 0]
    ds = ds.dropna()
    c = colors[ds.index.get_level_values("carrier")]
    ds = ds.pipe(rename_index)
    label = f"{ds.attrs['name']} [{ds.attrs['unit']}]"
    ds.plot.barh(color=c.values, xlabel=label, ax=ax)
    ax.grid(axis="y")


if __name__ == "__main__":
    if "snakemake" not in globals():

        snakemake = mock_snakemake(
            "plot_statistics",
            carrier="AC",
            planning_horizons="2030",
            pathway="exp175",
            topology="current+FCG",
            heating_demand="positive",
        )
    configure_logging(snakemake)
    carrier = snakemake.params.carrier

    n = pypsa.Network(snakemake.input.network)
    # # incase an old version need to add missing info to network
    fix_network_names_colors(n, snakemake.config)
    n.loads.carrier = "load"
    n.carriers.loc["load", ["nice_name", "color"]] = "Load", "darkred"
    colors = n.carriers.set_index("nice_name").color.where(lambda s: s != "", "lightgrey")

    outp_dir = snakemake.output.stats_dir
    if not os.path.exists(outp_dir):
        os.makedirs(outp_dir)

    stats_list = snakemake.params.stat_types

    if "capacity_factor" in stats_list:
        fig, ax = plt.subplots()
        ds = n.statistics.capacity_factor(bus_carrier=carrier).dropna()
        plot_static_per_carrier(ds, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "capacity_factor.png"))

    if "installed_capacity" in stats_list:
        fig, ax = plt.subplots()
        ds = n.statistics.installed_capacity(bus_carrier=carrier).dropna()
        if "Line" in ds.index:
            ds = ds.drop("Line")
        ds = ds.drop(("Generator", "Load"), errors="ignore")
        ds = ds / PLOT_CAP_UNITS
        ds.attrs["unit"] = PLOT_CAP_LABEL
        plot_static_per_carrier(ds.abs(), ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "installed_capacity.png"))

    if "optimal_capacity" in stats_list:
        fig, ax = plt.subplots()
        ds = n.statistics.optimal_capacity(bus_carrier=carrier)
        if "Line" in ds.index:
            ds = ds.drop("Line")
        ds = ds.drop(("Generator", "Load"), errors="ignore")
        ds = ds / PLOT_CAP_UNITS
        ds.attrs["unit"] = PLOT_CAP_LABEL
        plot_static_per_carrier(ds, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "optimal_capacity.png"))

    if "capex" in stats_list:
        fig, ax = plt.subplots()
        ds = n.statistics.capex(bus_carrier=carrier)
        plot_static_per_carrier(ds, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "capex.png"))

    if "opex" in stats_list:
        fig, ax = plt.subplots()
        ds = n.statistics.opex(bus_carrier=carrier)
        plot_static_per_carrier(ds, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "opex.png"))

    if "curtailment" in stats_list:
        fig, ax = plt.subplots()
        ds = n.statistics.curtailment(bus_carrier=carrier)
        plot_static_per_carrier(ds, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "curtailment.png"))

    if "supply" in stats_list:
        fig, ax = plt.subplots()
        ds = n.statistics.supply(bus_carrier=carrier)
        if "Line" in ds.index:
            ds = ds.drop("Line")
        ds = ds / PLOT_SUPPLY_UNITS
        ds.attrs["unit"] = PLOT_SUPPLY_LABEL
        plot_static_per_carrier(ds, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "supply.png"))

    if "withdrawal" in stats_list:
        fig, ax = plt.subplots()
        ds = n.statistics.withdrawal(bus_carrier=carrier)
        if "Line" in ds.index:
            ds = ds.drop("Line")
        ds = ds / PLOT_SUPPLY_UNITS
        ds.attrs["unit"] = PLOT_SUPPLY_LABEL
        plot_static_per_carrier(ds, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "withdrawal.png"))

    if "market_value" in stats_list:
        fig, ax = plt.subplots()
        ds = n.statistics.market_value(bus_carrier=carrier)
        plot_static_per_carrier(ds, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outp_dir, "market_value.png"))
