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


def plot_enhanced_market_value(n: pypsa.Network, ax: axes.Axes, colors: DataFrame, carrier="AC", show_mv_text=False, min_gen_share=0.01):
    """
    Plot market value (MV) per technology, optionally showing MV as text,
    and filter out technologies with too low generation share.

    Args:
        n (pypsa.Network): The network object.
        ax (matplotlib.axes.Axes): The axis to plot on.
        colors (DataFrame): Color mapping for technologies.
        carrier (str): Bus carrier to filter.
        show_mv_text (bool): Whether to show MV value as text on bars.
        min_gen_share (float): Minimum generation share (%) to show a technology.
    """
    # Get market value, generation share, and LCOE data
    mv_data = n.statistics.market_value(bus_carrier=carrier, comps="Generator").dropna()
    supply_data = n.statistics.supply(bus_carrier=carrier, comps="Generator")
    total_supply = supply_data.sum()
    gen_shares = (supply_data / total_supply * 100).dropna()
    lcoe_data = calc_lcoe(n, groupby=["carrier"], comps="Generator")["LCOE"].dropna()
    lcoe_data.index = lcoe_data.index.map(
        lambda idx: next(
            (row["nice_name"] for c, row in n.carriers.iterrows() if c.lower() == idx.lower()),
            idx
        )
    )

    # Merge into a DataFrame and drop incomplete rows
    df = pd.DataFrame({
        "MV": mv_data,
        "LCOE": lcoe_data,
        "GenShare": gen_shares
    }).dropna()

    # Keep only the row with the largest generation share for each technology
    df = df.sort_values("GenShare", ascending=False)
    df = df.loc[~df.index.duplicated(keep='first')]

    # Filter out technologies with too low generation share
    if min_gen_share > 0:
        df = df[df["GenShare"] >= min_gen_share]

    # Sort by MV for plotting
    df = df.sort_values("MV")
    y_pos = range(len(df))

    # Draw horizontal bar for MV
    bars = ax.barh(
        y_pos,
        df["MV"],
        color=[colors.get(tech, "lightgrey") for tech in df.index],
        alpha=0.7,
        label="Market Value"
    )

    # Optionally show MV value as text
    if show_mv_text:
        for i, val in enumerate(df["MV"]):
            ax.text(val + 0.5, i, f'{val:.1f}', color='black', va='center', ha='left', fontsize=9)

    # Draw generation share as red dots on a twin x-axis
    ax2 = ax.twiny()
    ax2.plot(
        df["GenShare"],
        y_pos,
        color='red',
        marker='o',
        linestyle='',
        label='Generation Share (%)',
        markersize=10,
        lw=0
    )
    for i, val in enumerate(df["GenShare"]):
        ax2.text(val + 0.5, i, f'{val:.1f}%', color='red', va='center', ha='left', fontsize=9)

    # Set axis labels and ticks
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index)
    ax.set_xlabel("Market Value [€/MWh]")
    ax2.set_xlabel("Generation Share [%]")
    ax.grid(False)
    ax2.grid(False)
    ax2.set_xlim(left=0)
    ax.set_xlim(left=0)

    # Legend
    lines, labels = ax2.get_legend_handles_labels()
    bars_legend = ax.barh([], [], color="lightgrey", alpha=0.7, label="Market Value")
    ax.legend([bars_legend, lines[0]], ["Market Value", "Generation Share (%)"], loc='best')

    return ax, ax2


def plot_enhanced_capacity_factor(n: pypsa.Network, ax: axes.Axes, colors: DataFrame, carrier="AC"):
    """
    Plot actual and theoretical capacity factors for each technology.
    Args:
        n (pypsa.Network): The network object.
        ax (matplotlib.axes.Axes): The axis to plot on.
        colors (DataFrame): Color mapping for technologies.
        carrier (str): Bus carrier to filter.
    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """
    # Special mapping for some carrier names
    special_map = {
        "battery charger": "Battery Storage",
        "battery discharger": "Battery Discharger",
        "battery": "Battery Storage"
    }
    # Actual capacity factor
    cf_data = n.statistics.capacity_factor(groupby=["carrier"]).dropna()
    if ("Link", "battery") in cf_data.index:
        cf_data.loc[("Link", "battery discharger")] = cf_data.loc[("Link", "battery")]
        cf_data.drop(index=("Link", "battery"), inplace=True)
    cf_data = cf_data.groupby(level=1).sum()
    cf_data.index = cf_data.index.map(lambda idx: n.carriers["nice_name"].get(idx, special_map.get(idx, idx)))

    # Theoretical capacity factor (prefer p_nom_opt)
    gen = n.generators.copy()
    gen["theo_cf"] = n.generators_t.p_max_pu.mean(axis=0)
    gen["nice_name"] = gen["carrier"].map(lambda idx: n.carriers["nice_name"].get(idx, special_map.get(idx, idx)))
    gen["p_nom_used"] = gen["p_nom_opt"].where(~gen["p_nom_opt"].isna(), gen["p_nom"])
    gen = gen[(gen["p_nom_used"] > 0) & (~gen["theo_cf"].isna())]
    gen["theoretical_energy"] = gen["theo_cf"] * gen["p_nom_used"]
    theoretical_energy = gen.groupby("nice_name")["theoretical_energy"].sum()
    total_p_nom = gen.groupby("nice_name")["p_nom_used"].sum()
    theoretical_cf_auto = theoretical_energy / total_p_nom

    # Manual calculation for hydro actual CF (for comparison)
    hydro = gen[gen["nice_name"] == "Hydroelectricity"]
    manual_actual_cf = None
    if not hydro.empty:
        actual_energy = n.generators_t.p[hydro.index].sum().sum()
        total_p_nom = hydro["p_nom_used"].sum()
        hours = len(n.snapshots)
        manual_actual_cf = actual_energy / (total_p_nom * hours)
        print(f"Manual hydro actual CF: {manual_actual_cf:.4f}")
        try:
            pypsa_cf = cf_data.loc["Hydroelectricity"]
            print(f"PyPSA hydro actual CF: {pypsa_cf:.4f}")
        except Exception as e:
            print("PyPSA hydro actual CF not found:", e)

    # Only plot technologies present in both actual and theoretical CF
    common_techs = cf_data.index.intersection(theoretical_cf_auto.index)
    cf_filtered = cf_data.loc[common_techs]
    theo_cf_filtered = theoretical_cf_auto.loc[cf_filtered.index]
    cf_filtered = cf_filtered.sort_values(ascending=True)
    theo_cf_filtered = theo_cf_filtered.loc[cf_filtered.index]

    # Replace hydro actual CF with manual value if available
    if manual_actual_cf is not None and "Hydroelectricity" in cf_filtered.index:
        cf_filtered.loc["Hydroelectricity"] = manual_actual_cf

    # Plotting
    x_pos = range(len(cf_filtered))
    width = 0.35
    bars1 = ax.barh([i - width/2 for i in x_pos], cf_filtered.values,
                    width, color=[colors.get(tech, "lightgrey") for tech in cf_filtered.index],
                    alpha=0.8, label='Actual CF')
    bars2 = ax.barh([i + width/2 for i in x_pos], theo_cf_filtered.values,
                    width, color=[colors.get(tech, "lightgrey") for tech in theo_cf_filtered.index],
                    alpha=0.4, label='Theoretical CF')
    for i, (tech, cf_val) in enumerate(cf_filtered.items()):
        ax.text(cf_val + 0.01, i - width/2, f'{cf_val:.2f}', va='center', ha='left', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        theo_val = theo_cf_filtered[tech]
        ax.text(theo_val + 0.01, i + width/2, f'{theo_val:.2f}', va='center', ha='left', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.5))
    ax.set_yticks(x_pos)
    ax.set_yticklabels(cf_filtered.index)
    ax.set_xlabel("Capacity Factor")
    ax.set_xlim(0, max(cf_filtered.max(), theo_cf_filtered.max()) * 1.1)
    ax.grid(False)
    ax.legend()
    return ax


if __name__ == "__main__":
    if "snakemake" not in globals():

        snakemake = mock_snakemake(
            "plot_statistics",
            carrier="AC",
            planning_horizons="2025",
            # co2_pathway="exp175default",
            # planning_horizons="2130",
            co2_pathway="SSP2-PkBudg1000-CHA-pypsaelh2",
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
        ds = n.statistics.capacity_factor(groupby=["carrier"]).dropna()
        # avoid grouping battery uif same name
        if ("Link", "battery") in ds.index:
            ds.loc[("Link", "battery charger")] = ds.loc[("Link", "battery")]
            ds.drop(index=("Link", "battery"), inplace=True)
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
        ds = n.statistics.optimal_capacity(groupby=["carrier"]).dropna()

        # Restore original link capacities to avoid modifying the network object
        n.links.p_nom_opt = original_p_nom_opt

        # Handle battery components correctly
        if ("Link", "battery") in ds.index:
            ds.loc[("Link", "battery charger")] = ds.loc[("Link", "battery")]
            ds.drop(index=("Link", "battery"), inplace=True)
        ds.drop("stations", level=1, inplace=True)
        if "load shedding" in ds.index.get_level_values(1):
            ds.drop("load shedding", level=1, inplace=True)
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
        fig, ax = plt.subplots(figsize=(12, 8))
        # Read min_gen_share from config if available
        min_gen_share = snakemake.config.get("MV_map", {}).get("min_gen_share", 0.01)
        plot_enhanced_market_value(n, ax, colors, carrier, show_mv_text=True, min_gen_share=min_gen_share)
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
