#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2024 The PyPSA-Eur Authors
# 2014 Adapted from pypsa-eur by PyPSA-China authors
#
# SPDX-License-Identifier: MIT

import matplotlib.pyplot as plt
import pypsa
import seaborn as sns
from _helpers import configure_logging
from _plot_utilities import rename_index, fix_network_names_colors

sns.set_theme("paper", style="whitegrid")


def plot_static_per_carrier(ds, ax, drop_zero=True):
    if drop_zero:
        ds = ds[ds != 0]
    ds = ds.dropna()
    c = colors[ds.index.get_level_values("carrier")]
    ds = ds.pipe(rename_index)
    label = f"{ds.attrs['name']} [{ds.attrs['unit']}]"
    ds.plot.barh(color=c.values, xlabel=label, ax=ax)
    ax.grid(axis="y")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_elec_statistics",
            simpl="",
            opts="Ept-12h",
            clusters="37",
            ll="v1.0",
            carrier="AC",
        )
    configure_logging(snakemake)
    carrier = snakemake.params.carrier

    n = pypsa.Network(snakemake.input.network)

    # incase an old version need to add missing info to network
    fix_network_names_colors(n)

    n.loads.carrier = "load"
    n.carriers.loc["load", ["nice_name", "color"]] = "Load", "darkred"
    colors = n.carriers.set_index("nice_name").color.where(lambda s: s != "", "lightgrey")

    fig, ax = plt.subplots()
    ds = n.statistics.capacity_factor(bus_carrier=carrier).dropna()
    plot_static_per_carrier(ds, ax)
    fig.savefig(snakemake.output.capacity_factor_bar)

    fig, ax = plt.subplots()
    ds = n.statistics.installed_capacity(bus_carrier=carrier).dropna()
    if "Line" in ds.index:
        ds = ds.drop("Line")
    ds = ds.drop(("Generator", "Load"), errors="ignore")
    ds = ds / 1e3
    ds.attrs["unit"] = "GW"
    plot_static_per_carrier(ds.abs(), ax)
    fig.savefig(snakemake.output.installed_capacity_bar)

    fig, ax = plt.subplots()
    ds = n.statistics.optimal_capacity(bus_carrier=carrier)
    if "Line" in ds.index:
        ds = ds.drop("Line")
    ds = ds.drop(("Generator", "Load"), errors="ignore")
    ds = ds / 1e3
    ds.attrs["unit"] = "GW"
    plot_static_per_carrier(ds, ax)
    fig.savefig(snakemake.output.optimal_capacity_bar)

    fig, ax = plt.subplots()
    ds = n.statistics.capex(bus_carrier=carrier)
    plot_static_per_carrier(ds, ax)
    fig.savefig(snakemake.output.capital_expenditure_bar)

    fig, ax = plt.subplots()
    ds = n.statistics.opex(bus_carrier=carrier)
    plot_static_per_carrier(ds, ax)
    fig.savefig(snakemake.output.operational_expenditure_bar)

    fig, ax = plt.subplots()
    ds = n.statistics.curtailment(bus_carrier=carrier)
    plot_static_per_carrier(ds, ax)
    fig.savefig(snakemake.output.curtailment_bar)

    fig, ax = plt.subplots()
    ds = n.statistics.supply(bus_carrier=carrier)
    if "Line" in ds.index:
        ds = ds.drop("Line")
    ds = ds / 1e6
    ds.attrs["unit"] = "TWh"
    plot_static_per_carrier(ds, ax)
    fig.savefig(snakemake.output.supply_bar)

    fig, ax = plt.subplots()
    ds = n.statistics.withdrawal(bus_carrier=carrier)
    if "Line" in ds.index:
        ds = ds.drop("Line")
    ds = ds / -1e6
    ds.attrs["unit"] = "TWh"
    plot_static_per_carrier(ds, ax)
    fig.savefig(snakemake.output.withdrawal_bar)

    fig, ax = plt.subplots()
    ds = n.statistics.market_value(bus_carrier=carrier)
    plot_static_per_carrier(ds, ax)
    fig.savefig(snakemake.output.market_value_bar)

    # touch file
    with open(snakemake.output.barplots_touch, "a"):
        pass
