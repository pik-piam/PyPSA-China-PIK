#!/usr/bin/env python3
"""Plot statistical analysis and summary charts for energy system results.

This module creates statistical plots including capacity factors, cost breakdowns,
energy balances, and other key performance indicators for the PyPSA-China model.
Adapted from PyPSA-Eur by PyPSA-China authors.
"""
# SPDX-FileCopyrightText: : 2017-2024 The PyPSA-Eur Authors
# 2014 Adapted from pypsa-eur by PyPSA-China authors
#
# SPDX-License-Identifier: MIT

import logging
import os

import matplotlib.axes as axes
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import seaborn as sns
from _helpers import configure_logging, mock_snakemake, set_plot_test_backend
from _plot_utilities import (
    fix_network_names_colors,
    make_nice_tech_colors,
    rename_index,
)
from _pypsa_helpers import calc_lcoe, filter_carriers
from constants import (
    PLOT_CAP_LABEL,
    PLOT_CAP_UNITS,
    PLOT_SUPPLY_LABEL,
    PLOT_SUPPLY_UNITS,
)

sns.set_theme("paper", style="whitegrid")
logger = logging.getLogger(__name__)


def format_axis_label(name: str, unit: str) -> str:
    """Format axis label by cleaning name and conditionally adding unit.

    Args:
        name (str): The name/description for the axis.
        unit (str): The unit of measurement (may be empty/None).

    Returns:
        str: Formatted label with underscores replaced by spaces,
            capitalized, and unit only included if non-empty.
    """
    clean_name = name.replace("_", " ").capitalize()
    if unit and unit.strip():
        return f"{clean_name} [{unit}]"
    return clean_name


def plot_static_per_carrier(
    ds: pd.Series,
    ax: axes.Axes,
    colors: pd.Series,
    drop_zero_vals=True,
    add_labels=True,
    autofigsize=True,
):
    """Generic function to plot different statics

    Args:
        ds (pd.Series): the data to plot
        ax (matplotlib.axes.Axes): plotting axes
        colors (pd.Series): colors for the carriers
        drop_zero_vals (bool, optional): Drop zeroes from data. Defaults to True.
        add_labels (bool, optional): Add value labels on bars. Defaults to True.
        autofigsize (bool, optional): Automatically size figure based on number
            of bars. Defaults to True.
    """
    if drop_zero_vals:
        ds = ds[ds != 0]
    ds = ds.dropna()

    n_bars = len(ds)

    # Determine figure size
    if autofigsize:
        bar_height = 0.4  # Height per bar in inches
        fig_height = max(4, min(n_bars * bar_height, 16))  # Between 4 and 16 inches
        figsize = (8, fig_height)
    else:
        figsize = None  # Use matplotlib default

    # Create or get figure
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
        if autofigsize:
            fig.set_size_inches(8, fig_height)

    c = colors[ds.index.get_level_values("carrier")]
    ds = ds.pipe(rename_index)
    label = format_axis_label(ds.attrs["name"], ds.attrs["unit"])
    ds.plot.barh(color=c.values, xlabel=label, ax=ax, alpha=0.9)

    # Remove y-axis label
    ax.set_ylabel("")

    # Adjust spacing between bars
    ax.margins(y=0.01)

    if add_labels:
        ymax = ax.get_xlim()[1] * 1.05
        for i, (index, value) in enumerate(ds.items()):
            align = "left"
            txt = f"{value:.2f}" if value <= 100 else f"{value:.1e}"
            ax.text(ymax, i, txt, va="center", ha=align, fontsize=8)
        # # Add outer y-ticks at the right y-axis frame
        # ax.tick_params(axis="y", direction="out", right=True, left=False)
    ax.grid(axis="y")
    fig.tight_layout()

    return fig


def filter_small_caps(n: pypsa.Network, threshold=100):
    """Drop small capacities for plotting (eliminate numerical zeroes)
    -> this would be more robust based on the objective cost tolerance

    Args:
        n (pypsa.Network): the pypsa network to remove small comps from
        threshold (int, optional): the removal threshold. Defaults to 100.
    """
    for c in ["links", "generators", "stores", "storage_units"]:
        attr = "e_nom_opt" if c == "stores" else "p_nom_opt"
        comp = getattr(n, c)
        mask = comp[attr] > threshold
        comp = comp.loc[mask]
        setattr(n, c, comp)


def set_link_output_capacities(n: pypsa.Network, carriers: list) -> pd.DataFrame:
    """Set link capacity to output and not input.
    PyPSA uses input link capacities but typically want to report output
    capacities (e.g MWel)

    Args:
        n (pypsa.Network): The PyPSA network instance.
        carriers (list): List of carrier names to adjust.

    Returns:
        pd.DataFrame: the original link capacities.
    """
    # Temporarily save original link capacities
    original_p_nom_opt = n.links.p_nom_opt.copy()

    # For links where bus1 is AC, multiply capacity by efficiency
    # coefficient to get AC side capacity
    ac_links = n.links[n.links.bus1.map(n.buses.carrier).isin(carriers)].index
    n.links.loc[ac_links, "p_nom_opt"] *= n.links.loc[ac_links, "efficiency"]
    n.links.loc[ac_links, "p_nom"] *= n.links.loc[ac_links, "efficiency"]

    # ignore lossy link dummies
    pseudo_links = n.links.query("Link.str.contains('reversed') & capital_cost ==0 ").index
    n.links.loc[pseudo_links, "p_nom_opt"] = 0

    return original_p_nom_opt


def fix_load_carriers(n: pypsa.Network, config: dict):
    """Set unspecified load carriers to load

    Args:
        n (pypsa.Network): The PyPSA network instance.
        config (dict): the plotting config
    """
    mask = n.loads.query("carrier==''").index
    n.loads.loc[mask, "carrier"] = "Load"
    n.carriers.loc["Load", ["nice_name", "color"]] = (
        "Load",
        config["tech_colors"]["electric load"],
    )


def add_second_xaxis(data: pd.Series, ax, label, **kwargs):
    """
    Add a secondary X-axis to the plot.

    Args:
        data (pd.Series): The data to plot. Its values will be plotted on the secondary X-axis.
        ax (matplotlib.axes.Axes): The main matplotlib Axes object.
        label (str): The label for the secondary X-axis.
        **kwargs: Optional keyword arguments for plot styling.
    """
    defaults = {"color": "red", "text_offset": 0.5, "markersize": 8, "fontsize": 9}
    kwargs.update(defaults)

    ax2 = ax.twiny()
    # y_pos creates a sequence of integers (e.g., [0, 1, 2, 3]) to serve
    # as distinct vertical positions for each data point on the shared Y-axis.
    # This is necessary because data.values are plotted horizontally on the
    # secondary X-axis (ax2), requiring vertical separation for clarity.
    y_pos = range(len(data))

    ax2.plot(
        data.values,
        y_pos,
        marker="o",
        linestyle="",
        color=kwargs["color"],
        markersize=kwargs["markersize"],
        label="Generation Share (%)",
    )

    for i, val in enumerate(data.values):
        ax2.text(
            val + kwargs["text_offset"],
            i,
            f"{val:.1f}%",
            color=kwargs["color"],
            va="center",
            ha="left",
            fontsize=kwargs["fontsize"],
        )

    ax2.set_xlim(left=0)
    ax2.set_xlabel(label)
    ax2.grid(False)
    ax2.tick_params(axis="x", labelsize=kwargs["fontsize"])  # Remove color setting for ticks

    return ax2


def prepare_capacity_factor_data(n: pypsa.Network, carrier: str):
    """
    Prepare Series for actual and theoretical capacity factors per technology.

    Args:
        n (pypsa.Network): The PyPSA network instance.
        carrier (str): The carrier for which to prepare the data.

    Returns:
        cf_filtered: Series of actual capacity factors (index: nice_name)
        theo_cf_filtered: Series of theoretical capacity factors (index: nice_name)
    """
    cf_data = n.statistics.capacity_factor(groupby=["carrier"]).dropna()
    if ("Link", "battery") in cf_data.index:
        cf_data.loc[("Link", "battery charger")] = cf_data.loc[("Link", "battery")]
        cf_data.drop(index=("Link", "battery"), inplace=True)
    cf_data = cf_data.groupby(level=1).mean()

    # Theoretical capacity factor
    gen = n.generators.copy()
    p_max_pu = n.generators_t.p_max_pu
    gen["p_nom_used"] = gen["p_nom_opt"].fillna(gen["p_nom"])
    weighted_energy_per_gen = (p_max_pu * gen["p_nom_used"]).sum()
    gen["weighted_energy"] = weighted_energy_per_gen

    gen["nice_name"] = gen["carrier"].map(
        lambda x: n.carriers.loc[x, "nice_name"] if x in n.carriers.index else x
    )
    grouped_energy = gen.groupby("nice_name")["weighted_energy"].sum()
    grouped_capacity = gen.groupby("nice_name")["p_nom_used"].sum()
    theoretical_cf_weighted = grouped_energy / grouped_capacity / len(n.snapshots)

    # Only keep technologies present in both actual and theoretical CF
    common_techs = cf_data.index.intersection(theoretical_cf_weighted.index)
    cf_filtered = cf_data.loc[common_techs]
    theo_cf_filtered = theoretical_cf_weighted.loc[cf_filtered.index]
    # Todo: use config nondispatchable_techs
    non_zero_mask = (cf_filtered != 0) & (theo_cf_filtered != 0)
    cf_filtered = cf_filtered[non_zero_mask]
    theo_cf_filtered = theo_cf_filtered[non_zero_mask]
    cf_filtered = cf_filtered.sort_values(ascending=True)
    theo_cf_filtered = theo_cf_filtered.loc[cf_filtered.index]

    return cf_filtered, theo_cf_filtered


def plot_capacity_factor(
    cf_filtered: pd.Series, theo_cf_filtered: pd.Series, ax: axes.Axes, colors: dict, **kwargs
):
    """
    Plot actual and theoretical capacity factors for each technology.

    Args:
        cf_filtered (pd.Series): Actual capacity factors indexed by technology.
        theo_cf_filtered (pd.Series): Theoretical capacity factors indexed by technology.
        ax (matplotlib.axes.Axes): The axis to plot on.
        colors (dict): Color mapping for technologies.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """
    x_pos = range(len(cf_filtered))
    width = 0.35

    ax.barh(
        [i - width / 2 for i in x_pos],
        cf_filtered.values,
        width,
        color=[colors.get(tech, "lightgrey") for tech in cf_filtered.index],
        alpha=0.8,
        label="Actual CF",
    )
    ax.barh(
        [i + width / 2 for i in x_pos],
        theo_cf_filtered.values,
        width,
        color=[colors.get(tech, "lightgrey") for tech in theo_cf_filtered.index],
        alpha=0.4,
        label="Theoretical CF",
    )

    for i, (tech, cf_val) in enumerate(cf_filtered.items()):
        ax.text(
            cf_val + 0.01,
            i - width / 2,
            f"{cf_val:.2f}",
            va="center",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )
        theo_val = theo_cf_filtered.get(tech, 0)
        ax.text(
            theo_val + 0.01,
            i + width / 2,
            f"{theo_val:.2f}",
            va="center",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.5),
        )

    ax.set_yticks(list(x_pos))
    ax.set_yticklabels([label.capitalize() for label in cf_filtered.index])
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
        names=["province", "nice_name"],
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
        attached_nice_names = [
            n.carriers.loc[c, "nice_name"] if c in n.carriers.index else c
            for c in attached_carriers
        ]
        cap_by_prov_tech = cap_by_prov_tech[
            [c for c in cap_by_prov_tech.columns if c in attached_nice_names]
        ]

    # Merge peak load and capacity
    df_plot = cap_by_prov_tech.copy()
    df_plot["Peak Load"] = peak_load

    # Bar columns: exclude Peak Load, only keep nonzero
    bar_cols = [c for c in df_plot.columns if c != "Peak Load"]
    bar_cols = [c for c in bar_cols if df_plot[c].sum() > 0]
    color_list = [
        n.carriers.set_index("nice_name").color.get(tech, "lightgrey") for tech in bar_cols
    ]
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
        ax.plot(
            df_plot.loc[prov, "Peak Load"],
            i,
            "r|",
            markersize=18,
            label="Peak Load" if i == 0 else "",
        )
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
            planning_horizons="2060",
            co2_pathway="exp175default",
            topology="current+FCG",
            heating_demand="positive",
            # configfiles="resources/tmp/pseudo-coupled.yaml",
        )
    configure_logging(snakemake)
    set_plot_test_backend(snakemake.config)

    n = pypsa.Network(snakemake.input.network)
    # rename "AC" link to Transmission
    hv_mask = n.links.carrier == "AC"
    n.links.loc[hv_mask, "carrier"] = "Transmission Lines"

    carriers = snakemake.params.carriers
    config = snakemake.config["plotting"]
    outp_dir = snakemake.output.stats_dir
    if not os.path.exists(outp_dir):
        os.makedirs(outp_dir)

    # remove small capacities
    filter_small_caps(n, config["capacity_threshold"])
    # backward compat: add missing info to network
    fix_network_names_colors(n, snakemake.config)
    # fix_load_carriers
    fix_load_carriers(n, config)
    extra_c = {
        "Load": config["tech_colors"]["electric load"],
        "transmission losses": config["tech_colors"]["transmission losses"],
    }
    nice_tech_colors = make_nice_tech_colors(config["tech_colors"], config["nice_names"])
    nice_tech_colors.update(extra_c)
    colors = pd.Series(nice_tech_colors)

    # Get configuration from snakemake
    report_output_power = snakemake.config.get("reporting", {}).get(
        "adjust_link_capacities_by_efficiency", True
    )
    if report_output_power:
        original_p_nom_opt = set_link_output_capacities(n, carriers=["AC", "heat"])

    attached_carriers = filter_carriers(n, carriers)

    stats_list = snakemake.params.stat_types

    # write values for selected plots
    labels_list = config["statistics"]["add_labels"]
    add_label = {plot_type: False for plot_type in stats_list}
    add_label.update({plot_type: True for plot_type in labels_list})

    # DEFINE A SETTING FOR EACH PLOT WITH A FUNCTION AND UNITS
    # loop over all settings, apply function, post process as needed (group)
    # CARE: ds is converted to a seir
    grouper = n.statistics.groupers.get_carrier_and_bus_carrier
    STATS_SETTINGS = {
        "capacity_factor": {
            "calc": n.statistics.capacity_factor,
            "pre": lambda ds: _handle_battery(ds),
            "post": lambda ds: ds.groupby(["carrier", "bus_carrier"]).first(),
            "unit": None,
            "extra": None,
            "filename": "capacity_factor.png",
        },
        "installed_capacity": {
            "calc": n.statistics.installed_capacity,
            "pre": None,
            "post": lambda ds: ds.groupby(["carrier", "bus_carrier"]).sum(),
            "unit": PLOT_CAP_LABEL,
            "conversion": PLOT_CAP_UNITS,
            "extra": lambda ds: ds.drop("Line") if "Line" in ds.index else ds,
            "filename": "installed_capacity.png",
        },
        "optimal_capacity": {
            "calc": n.statistics.optimal_capacity,
            "pre": lambda ds: _handle_battery(ds).abs(),
            "post": lambda ds: ds.groupby(["carrier", "bus_carrier"]).sum(),
            "conversion": PLOT_CAP_UNITS,
            "extra": None,
            "unit": PLOT_CAP_LABEL,
            "filename": "optimal_capacity.png",
        },
        "capital_expenditure": {
            "calc": n.statistics.capex,
            "pre": None,
            "post": lambda ds: ds.groupby(["carrier", "bus_carrier"]).sum(),
            "unit": "bn eur",
            "conversion": 1e9,
            "extra": None,
            "filename": "capex.png",
        },
        "operational_expenditure": {
            "calc": n.statistics.opex,
            "pre": None,
            "post": lambda ds: ds.groupby(["carrier", "bus_carrier"]).sum(),
            "unit": "bn€",
            "conversion": 1e9,
            "extra": None,
            "filename": "opex.png",
        },
        "supply": {
            "calc": n.statistics.supply,
            "pre": None,
            "unit": PLOT_SUPPLY_LABEL,
            "conversion": PLOT_SUPPLY_UNITS,
            "extra": lambda ds: ds.drop("Line") if "Line" in ds.index else ds,
            "filename": "supply.png",
        },
        "withdrawal": {
            "calc": n.statistics.withdrawal,
            "pre": None,
            "post": lambda ds: ds.groupby(["carrier", "bus_carrier"]).sum(),
            "unit": PLOT_SUPPLY_LABEL,
            "conversion": PLOT_SUPPLY_UNITS,
            "extra": lambda ds: ds.drop("Line") if "Line" in ds.index else ds,
            "filename": "withdrawal.png",
        },
        "market_value": {
            "calc": n.statistics.market_value,
            "pre": None,
            "post": lambda ds: ds.groupby(["carrier", "bus_carrier"]).sum(),
            "unit": "€/MWh",
            "figsize": (12, 8),
            "extra": "market_value",
            "filename": "market_value.png",
        },
        "lcoe": {
            # ugly kwargs trick
            "calc": lambda **x: calc_lcoe(n)["LCOE"],
            "pre": None,
            "post": lambda ds: ds.groupby(["carrier"]).first(),
            "unit": "€/MWh",
            "extra": None,
            "filename": "LCOE.png",
        },
        "mv_minus_lcoe": {
            "calc": lambda **x: calc_lcoe(n)["profit_pu"],
            "pre": None,
            "post": lambda ds: ds.groupby(["carrier"]).first(),
            "unit": "€/MWh",
            "extra": None,
            "filename": "MV_minus_LCOE.png",
        },
    }

    # Helper functions for special handling
    def _handle_battery(ds):
        if ("Link", "battery") in ds.index:
            ds.loc[("Link", "battery charger", "AC")] = ds.loc[("Link", "battery", "AC")]
            ds.drop(index=("Link", "battery"), inplace=True)
        return ds

    def _handle_water_tanks(ds):
        if ("Link", "battery") in ds.index:
            ds.loc[("Link", "battery charger")] = ds.loc[("Link", "battery")]
            ds.drop(index=("Link", "battery"), inplace=True)
        return ds

    end_carriers = ["H2", "heat", "AC"]
    # Main loop for statistics
    for stat in stats_list:
        if stat not in STATS_SETTINGS:
            logger.warning(f"Statistic '{stat}' not recognized. Skipping.")
            continue
        settings = STATS_SETTINGS[stat]

        if stat == "optimal_capacity":
            pass
        # Calculate
        _carriers = filter_carriers(n, carriers)
        ds = settings["calc"](groupby=grouper, carrier=_carriers, nice_names=False).dropna()

        conversion = settings.get("conversion", 1)
        ds /= conversion
        # Preprocess
        if settings.get("pre"):
            ds = settings["pre"](ds)
        ds = ds.reset_index()
        # Postprocess
        if settings.get("post"):
            ds = settings["post"](ds)

        ds.attrs.update({"name": stat, "unit": settings.get("unit", "")})

        # perform plotting
        ds = ds.reset_index()
        if "bus_carrier" in ds.columns:
            plot_carriers = ds.bus_carrier.unique()
            plot_carriers = [car for car in plot_carriers if car in end_carriers]
        else:
            plot_carriers = ["AC"]
            ds.loc[:, "bus_carrier"] = "AC"

        for plot_carrier in plot_carriers:
            ds_plot = ds.query("bus_carrier==@plot_carrier").set_index("carrier")
            ds_plot.rename(
                columns={0: "value", "objective": "value", "LCOE": "value", "profit_pu": "value"},
                inplace=True,
            )
            ds_plot = ds_plot["value"]
            fig, ax = plt.subplots(**settings.get("fig_opts", {}))
            plot_static_per_carrier(
                ds_plot, ax, colors=colors, add_labels=add_label.get(stat, False)
            )
            fig.tight_layout()
            if len(plot_carriers) == 1 or plot_carrier == "AC":
                fname = settings["filename"]
            else:
                fname = settings["filename"].replace(".png", f"_{plot_carrier}.png")
            fig.savefig(os.path.join(outp_dir, fname))
