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

from _plot_utilities import heatmap, annotate_heatmap
from _helpers import configure_logging, mock_snakemake, set_plot_test_backend
from _plot_utilities import rename_index, fix_network_names_colors, filter_carriers
from _pypsa_helpers import calc_lcoe, calc_generation_share
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
    c = colors[ds.index.get_level_values("carrier")]
    ds = ds.pipe(rename_index)
    label = f"{ds.attrs['name']} [{ds.attrs['unit']}]"
    ds.plot.barh(color=c.values, xlabel=label, ax=ax)
    for i, (index, value) in enumerate(ds.items()):
        ax.text(value, i, f"{value:.1f}", va='center', ha='left', fontsize=8)
    ax.grid(axis="y")


def add_generation_share(ax, shares, color='red', text_offset=0.5, markersize=8, fontsize=9):
    """
    Add generation share markers and percentage text on a twin x-axis.

    Args:
        ax (matplotlib.axes.Axes): The main axis where bar chart is drawn.
        shares (pd.Series): Generation share values (aligned with y-axis labels).
        color (str): Color of the dots and text.
        text_offset (float): Horizontal offset for the text labels.
        markersize (int): Size of the dots.
        fontsize (int): Font size of the text labels.
    
    Returns:
        ax2 (matplotlib.axes.Axes): The secondary x-axis created for generation share.
    """
    ax2 = ax.twiny()
    y_pos = range(len(shares))

    ax2.plot(
        shares.values,
        y_pos,
        marker='o',
        linestyle='',
        color=color,
        markersize=markersize,
        label="Generation Share (%)"
    )

    for i, val in enumerate(shares.values):
        ax2.text(val + text_offset, i, f"{val:.1f}%", color=color,
                 va='center', ha='left', fontsize=fontsize)

    ax2.set_xlim(left=0)
    ax2.set_xlabel("Generation Share [%]")
    ax2.grid(False)
    ax2.tick_params(axis='x', labelsize=fontsize)  # Remove color setting for ticks

    return ax2


def prepare_capacity_factor_data(n, carrier):
    """
    Prepare Series for actual and theoretical capacity factors per technology.
    Returns:
        cf_filtered: Series of actual capacity factors (index: nice_name)
        theo_cf_filtered: Series of theoretical capacity factors (index: nice_name)
    """
    special_map = {
        "battery charger": "Battery Storage",
        "battery discharger": "Battery Discharger",
        "battery": "Battery Storage"
    }

    # Actual capacity factor
    cf_data = n.statistics.capacity_factor(groupby=["carrier"]).dropna()
    if ("Link", "battery") in cf_data.index:
        cf_data.loc[("Link", "battery charger")] = cf_data.loc[("Link", "battery")]
        cf_data.drop(index=("Link", "battery"), inplace=True)
    cf_data = cf_data.groupby(level=1).mean()  # Use mean instead of sum
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

    # Only keep technologies present in both actual and theoretical CF
    common_techs = cf_data.index.intersection(theoretical_cf_auto.index)
    cf_filtered = cf_data.loc[common_techs]
    theo_cf_filtered = theoretical_cf_auto.loc[cf_filtered.index]
    cf_filtered = cf_filtered.sort_values(ascending=True)
    theo_cf_filtered = theo_cf_filtered.loc[cf_filtered.index]

    return cf_filtered, theo_cf_filtered

def plot_capacity_factor(n: pypsa.Network, ax: axes.Axes, colors: DataFrame, carrier="AC"):
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
    cf_filtered, theo_cf_filtered = prepare_capacity_factor_data(n, carrier)
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


def prepare_province_peakload_capacity_data(n, attached_carriers=None):
    """
    Prepare DataFrame for province peak load and installed capacity by technology.
    Returns:
        df_plot: DataFrame with provinces as index, columns as technologies and 'Peak Load'.
        bar_cols: List of technology columns to plot as bars.
        color_list: List of colors for each technology.
    """
    # Calculate peak load per province
    load = n.loads.copy()
    load["province"] = load["bus"].map(n.buses["location"])
    peak_load = n.loads_t.p_set.groupby(load["province"], axis=1).sum().max()
    peak_load = peak_load / PLOT_CAP_UNITS  # ensure peak load is in GW

    # Calculate installed capacity per province and technology using optimal_capacity
    ds = n.statistics.optimal_capacity(groupby=["location", "carrier"]).dropna()
    valid_components = ["Generator", "StorageUnit", "Link"]
    ds = ds.loc[ds.index.get_level_values(0).isin(valid_components)]
    if ("Link", "battery") in ds.index:
        ds.loc[("Link", "battery charger")] = ds.loc[("Link", "battery")]
        ds = ds.drop(index=("Link", "battery"))
    if "stations" in ds.index.get_level_values(2):
        ds = ds.drop("stations", level=2)
    if "load shedding" in ds.index.get_level_values(2):
        ds = ds.drop("load shedding", level=2)
    ds = ds.groupby(level=[1, 2]).sum()
    ds.index = pd.MultiIndex.from_tuples(
        [
            (prov, n.carriers.loc[carrier, "nice_name"] if carrier in n.carriers.index else carrier)
            for prov, carrier in ds.index
        ],
        names=["province", "nice_name"]
    )
    cap_by_prov_tech = ds.unstack(level=-1).fillna(0)
    cap_by_prov_tech = cap_by_prov_tech.abs() / PLOT_CAP_UNITS

    if "Battery Discharger" in cap_by_prov_tech.columns:
        cap_by_prov_tech = cap_by_prov_tech.drop(columns="Battery Discharger")
    if "AC" in cap_by_prov_tech.columns:
        cap_by_prov_tech = cap_by_prov_tech.drop(columns="AC")
    # Only keep columns in attached_carriers if provided
    if attached_carriers is not None:
        # Ensure nice_name mapping for attached_carriers
        attached_nice_names = [n.carriers.loc[c, "nice_name"] if c in n.carriers.index else c for c in attached_carriers]
        cap_by_prov_tech = cap_by_prov_tech[[c for c in cap_by_prov_tech.columns if c in attached_nice_names]]

    # Merge peak load and capacity
    df_plot = cap_by_prov_tech.copy()
    df_plot["Peak Load"] = peak_load

    # Bar columns: exclude Peak Load, only keep nonzero
    bar_cols = [c for c in df_plot.columns if c != "Peak Load"]
    bar_cols = [c for c in bar_cols if df_plot[c].sum() > 0]
    color_list = [n.carriers.set_index("nice_name").color.get(tech, "lightgrey") for tech in bar_cols]
    return df_plot, bar_cols, color_list


def plot_province_peakload_capacity(df_plot, bar_cols, color_list, outp_dir):
    """
    Plot province peak load vs installed capacity by technology.
    Args:
        df_plot: DataFrame with provinces as index, columns as technologies and 'Peak Load'.
        bar_cols: List of technology columns to plot as bars.
        color_list: List of colors for each technology.
        outp_dir: Output directory for saving the figure.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    df_plot[bar_cols].plot(kind="barh", stacked=True, ax=ax, color=color_list, alpha=0.8)
    # Plot peak load as red vertical line
    for i, prov in enumerate(df_plot.index):
        ax.plot(df_plot.loc[prov, "Peak Load"], i, "r|", markersize=18, label="Peak Load" if i==0 else "")
    ax.set_xlabel("Capacity [GW]")
    ax.set_ylabel("Province")
    ax.set_title("Peak Load vs Installed Capacity by Province")
    ax.grid(False)
    # Only keep one Peak Load legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    new_handles, new_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)
    ax.legend(new_handles, new_labels, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outp_dir, "province_peakload_capacity.png"))


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
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_capacity_factor(n, ax, colors, carrier)
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
        plot_static_per_carrier(ds, ax, colors=colors)
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

        # 1. Calculate curtailment and actual generation by province and technology
        curtailment = n.statistics.curtailment(comps="Generator", groupby=["location", "carrier"], bus_carrier=carrier)
        supply = n.statistics.supply(comps="Generator", groupby=["location", "carrier"], bus_carrier=carrier)

        # 2. Calculate curtailment rate (curtailment / (curtailment + actual generation))
        curtailment_rate = curtailment / (curtailment + supply.replace(0, np.nan)) * 100
        curtailment_rate = curtailment_rate.fillna(0)

        # 3. Convert to DataFrame for plotting
        df_rate = curtailment_rate.unstack(level=-1).fillna(0)
        # Map columns to nice_name
        df_rate.columns = [n.carriers.loc[c, "nice_name"] if c in n.carriers.index else c for c in df_rate.columns]
        colors_nice = n.carriers.set_index("nice_name").color
        color_list = [colors_nice.get(tech, "lightgrey") for tech in df_rate.columns]

        vre_techs = snakemake.config["Techs"]["vre_techs"]
        vre_cols = [c for c in df_rate.columns if any(v.lower() in c.lower() for v in vre_techs)]
        df_vre = df_rate[vre_cols]

        fig, ax = plt.subplots(figsize=(14, 8))
        im, cbar = heatmap(
            df_vre.values, df_vre.index, df_vre.columns, ax=ax,
            cmap="magma_r", cbarlabel="Curtailment Rate [%]", vmin=0, vmax=100
        )
        annotate_heatmap(im, valfmt="{x:.1f}", size=8, threshold=50, textcolors=("black", "white"))
        ax.set_xlabel("Technology")
        ax.set_ylabel("Province")
        ax.set_title("Curtailment Rate Heatmap by Province and Technology")
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(outp_dir, "curtailment_heatmap.png"))
        plt.close()

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
        min_gen_share = snakemake.config.get("market_value", {}).get("min_gen_share", 0.01)
        ds = n.statistics.market_value(bus_carrier=carrier, comps="Generator")
        ds.attrs = {"name": "Market Value", "unit": "€/MWh"}
        df = pd.DataFrame({"MV": ds})
        df = calc_generation_share(df, n, carrier)
        df = df.dropna()
        df = df[df["GenShare"] >= min_gen_share]
        df = df.sort_values("MV")
        # Restore attrs to the series after DataFrame operations
        mv_series = df["MV"]
        mv_series.attrs = {"name": "Market Value", "unit": "€/MWh"}
        plot_static_per_carrier(mv_series, ax=ax, colors=colors)
        add_generation_share(ax, df["GenShare"])
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

    if "province_peakload_capacity" in stats_list:
        df_plot, bar_cols, color_list = prepare_province_peakload_capacity_data(n, attached_carriers=attached_carriers)
        plot_province_peakload_capacity(df_plot, bar_cols, color_list, outp_dir)
