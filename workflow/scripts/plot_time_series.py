import pypsa
import logging
import matplotlib.pyplot as plt
import os.path

import seaborn as sns
import numpy as np
import pandas as pd

from os import makedirs

from _plot_utilities import (
    get_stat_colors,
    set_plot_style,
    make_nice_tech_colors,
    fix_network_names_colors,
)
from _pypsa_helpers import get_location_and_carrier
from _helpers import (
    configure_logging,
    mock_snakemake,
    set_plot_test_backend,
)
from constants import PLOT_CAP_UNITS, PLOT_CAP_LABEL, PROV_NAMES

logger = logging.getLogger(__name__)


def plot_energy_balance(
    n: pypsa.Network,
    plot_config: dict,
    bus_carrier="AC",
    start_date="2060-03-31 21:00",
    end_date="2060-04-06 12:00:00",
    add_load_line=True,
    ax: plt.Axes = None,
):
    """plot the electricity balance of the network for the given time range

    Args:
        n (pypsa.Network): the network
        plot_config (dict): the plotting config (snakemake.config["plotting"])
        bus_carrier (str, optional): the carrier for the energy_balance op. Defaults to "AC".
        start_date (str, optional): the range to plot. Defaults to "2060-03-31 21:00".
        end_date (str, optional): the range to plot. Defaults to "2060-04-06 12:00:00".
        add_load_line (bool, optional): add a dashed line for the load. Defaults to True.
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(16, 8))
    else:
        fig = ax.get_figure()

    p = (
        n.statistics.energy_balance(aggregate_time=False, bus_carrier=bus_carrier)
        .dropna(how="all")
        .groupby("carrier")
        .sum()
        .div(PLOT_CAP_UNITS)
        # .drop("-")
        .T
    )

    p.rename(columns={"-": "Load", "AC": "transmission losses"}, inplace=True)
    p = p.loc[start_date:end_date]

    # aggreg fossil
    coal = p.filter(regex="[C|c]oal")
    p.drop(columns=coal.columns, inplace=True)
    p["Coal"] = coal.sum(axis=1)
    gas = p.filter(regex="[G|g]as")
    p.drop(columns=gas.columns, inplace=True)
    p["Gas"] = gas.sum(axis=1)

    extra_c = {
        "Load": plot_config["tech_colors"]["electric load"],
        "transmission losses": plot_config["tech_colors"]["transmission losses"],
    }
    nice_tech_colors = make_nice_tech_colors(plot_config["tech_colors"], plot_config["nice_names"])
    color_series = get_stat_colors(n, nice_tech_colors, extra_colors=extra_c)
    # colors & names part 1
    p.rename(plot_config["nice_names"], inplace=True)
    p.rename(columns={k: k.title() for k in p.columns}, inplace=True)
    color_series.index = color_series.index.str.strip()
    # split into supply and wothdrawal
    supply = p.where(p > 0).dropna(axis=1, how="all")
    charge = p.where(p < 0).dropna(how="all", axis=1)

    # fix names and order

    charge.rename(columns={"Battery Storage": "Battery"}, inplace=True)
    supply.rename(columns={"Battery Discharger": "Battery"}, inplace=True)
    color_series = color_series[charge.columns.union(supply.columns)]
    color_series.rename(
        {"Battery Discharger": "Battery", "Battery Storage": "Battery"},
        inplace=True,
    )
    # Deduplicate color_series
    color_series = color_series[~color_series.index.duplicated(keep="first")]

    preferred_order = plot_config["preferred_order"]
    plot_order = (
        supply.columns.intersection(preferred_order).to_list()
        + supply.columns.difference(preferred_order).to_list()
    )

    plot_order_charge = [name for name in preferred_order if name in charge.columns] + [
        name for name in charge.columns if name not in preferred_order
    ]

    supply = supply.reindex(columns=plot_order)
    charge = charge.reindex(columns=plot_order_charge)
    if not charge.empty:
        charge.plot.area(ax=ax, linewidth=0, color=color_series.loc[charge.columns])

    supply.plot.area(
        ax=ax,
        linewidth=0,
        color=color_series.loc[supply.columns].values,
    )
    if add_load_line:
        charge["load_pos"] = charge["Load"] * -1
        charge["load_pos"].plot(linewidth=2, color="black", label="Load", ax=ax, linestyle="--")
        charge.drop(columns="load_pos", inplace=True)

    ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=16)
    ax.set_ylabel(PLOT_CAP_LABEL)
    ax.set_ylim(charge.sum(axis=1).min() * 1.07, supply.sum(axis=1).max() * 1.07)
    ax.grid(axis="y")
    ax.set_xlim(supply.index.min(), supply.index.max())

    fig.tight_layout()

    return ax


def plot_load_duration_curve(
    network: pypsa.Network, carrier: str = "AC", ax: plt.Axes = None
) -> plt.Axes:
    """plot the load duration curve for the given carrier

    Args:
        network (pypsa.Network): the pypasa network object
        carrier (str, optional): the load carrier, defaults to AC
        ax (plt.Axes, optional): figure axes, if none fig will be created. Defaults to None.

    Returns:
        plt.Axes: the plotting axes
    """

    if not ax:
        fig, ax = plt.subplots(figsize=(16, 8))
    else:
        fig = ax.get_figure()

    load = network.statistics.withdrawal(
        groupby=get_location_and_carrier,
        aggregate_time=False,
        bus_carrier=carrier,
        comps="Load",
    ).sum()
    load_curve = load.sort_values(ascending=False) / PLOT_CAP_UNITS
    load_curve.reset_index(drop=True).plot(ax=ax, lw=3)
    ax.set_ylabel(f"Load [{PLOT_CAP_LABEL}]")
    ax.set_xlabel("Hours")

    fig.tight_layout()
    return ax


def plot_regional_load_durations(
    network: pypsa.Network, carrier="AC", ax=None, cmap="plasma"
) -> plt.Axes:
    """plot the load duration curve for the given carrier stacked by region

    Args:
        network (pypsa.Network): the pypasa network object
        carrier (str, optional): the load carrier, defaults to AC
        ax (plt.Axes, optional): axes to plot on, if none fig will be created. Defaults to None.

    Returns:
        plt.Axes: the plotting axes
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    loads_all = network.statistics.withdrawal(
        groupby=get_location_and_carrier,
        aggregate_time=False,
        bus_carrier=carrier,
        comps="Load",
    ).sum()
    load_curve_all = loads_all.sort_values(ascending=False) / PLOT_CAP_UNITS
    regio = network.statistics.withdrawal(
        groupby=get_location_and_carrier,
        aggregate_time=False,
        bus_carrier=carrier,
        comps="Load",
    )
    regio = regio.droplevel(1).T
    load_curve_regio = regio.loc[load_curve_all.index] / PLOT_CAP_UNITS
    load_curve_regio.reset_index(drop=True).plot.area(
        ax=ax, stacked=True, cmap=cmap, legend=True, lw=3
    )
    ax.set_ylabel(f"Load [{PLOT_CAP_LABEL}]")
    ax.set_xlabel("Hours")
    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fontsize="small",
        title_fontsize="small",
        fancybox=True,
        shadow=True,
    )

    fig.tight_layout()

    return ax


def plot_residual_load_duration_curve(
    network, ax: plt.Axes = None, vre_techs=["Onshore Wind", "Offshore Wind", "Solar"]
) -> plt.Axes:
    """plot the residual load duration curve for the given carrier

    Args:
        network (pypsa.Network): the pypasa network object
        ax (plt.Axes, optional): Axes to plot on, if none fig will be created. Defaults to None.

    Returns:
        plt.Axes: the plotting axes
    """
    CARRIER = "AC"
    if not ax:
        fig, ax = plt.subplots(figsize=(16, 8))
    load = network.statistics.withdrawal(
        groupby=get_location_and_carrier,
        aggregate_time=False,
        bus_carrier=CARRIER,
        comps="Load",
    ).sum()

    vre_supply = (
        network.statistics.supply(
            groupby=get_location_and_carrier,
            aggregate_time=False,
            bus_carrier=CARRIER,
            comps="Generator",
        )
        .groupby(level=1)
        .sum()
        .loc[vre_techs]
        .sum()
    )

    residual = (load - vre_supply).sort_values(ascending=False) / PLOT_CAP_UNITS
    residual.reset_index(drop=True).plot(ax=ax, lw=3)
    ax.set_ylabel(f"Residual Load [{PLOT_CAP_LABEL}]")
    ax.set_xlabel("Hours")

    return ax


def plot_price_duration_curve(
    network: pypsa.Network, carrier="AC", ax: plt.Axes = None, figsize=(8, 8)
) -> plt.Axes:
    """plot the price duration curve for the given carrier

    Args:
        network (pypsa.Network): the pypasa network object
        carrier (str, optional): the load carrier, defaults to AC
        ax (plt.Axes, optional): Axes to plot on, if none fig will be created. Defaults to None.
        figsize (tuple, optional): size of the figure (if no ax given), defaults to (8, 8)
    Returns:
        plt.Axes: the plotting axes
    """
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ntwk_el_price = (
        -1
        * network.statistics.revenue(bus_carrier=carrier, aggregate_time=False, comps="Load")
        / network.statistics.withdrawal(bus_carrier=carrier, aggregate_time=False, comps="Load")
    ).T
    ntwk_el_price.rename(columns={"-": "Load"}, inplace=True)
    ntwk_el_price.Load.sort_values(ascending=False).reset_index(drop=True).plot(
        title="Price Duration Curve", ax=ax, lw=2
    )
    fig.tight_layout()

    return ax


def plot_price_duration_by_node(
    network: pypsa.Network,
    carrier: str = "AC",
    logy=True,
    y_lower=1e-3,
    fig_shape=(8, 4),
) -> plt.Axes:
    """Plot the price duration curve for the given carrier by node
    Args:
        network (pypsa.Network): the pypsa network object
        carrier (str, optional): the load carrier, defaults to AC (bus suffix)
        logy (bool, optional): use log scale for y axis, defaults to True
        y_lower (float, optional): lower limit for y axis, defaults to 1e-3
        fig_shape (tuple, optional): shape of the figure, defaults to (8, 4)
    Returns:
        plt.Axes: the plotting axes
    Raises:
        ValueError: if the figure shape is too small for the number of regions"""

    if carrier == "AC":
        suffix = ""
    else:
        suffix = f" {carrier}"

    nodal_prices = network.buses_t.marginal_price[pd.Index(PROV_NAMES) + suffix]

    if fig_shape[0] * fig_shape[1] < len(nodal_prices.columns):
        raise ValueError(
            f"Figure shape {fig_shape} is too small for {len(nodal_prices.columns)} regions. "
            + "Please increase the number of subplots."
        )
    fig, axes = plt.subplots(fig_shape[0], fig_shape[1], sharex=True, sharey=True, figsize=(12, 12))

    # region by region sorting of prices
    for i, region in enumerate(nodal_prices.columns):
        reg_pr = nodal_prices[region]
        reg_pr.sort_values(ascending=False).reset_index(drop=True).plot(
            ax=axes[i // 4, i % fig_shape[1]], label=region
        )
        axes[i // 4, i % fig_shape[1]].set_title(region, fontsize=10)
        if logy:
            axes[i // 4, i % fig_shape[1]].semilogy()
        if y_lower:
            axes[i // 4, i % fig_shape[1]].set_ylim(y_lower, reg_pr.max() * 1.2)
        elif reg_pr.min() > 1e-5 and not logy:
            axes[i // 4, i % fig_shape[1]].set_ylim(0, reg_pr.max() * 1.2)
    fig.tight_layout(h_pad=0.2, w_pad=0.2)
    for ax in axes.flat:
        # Remove all x-tick labels except the largest value
        xticks = ax.get_xticks()
        if len(xticks) > 0:
            ax.set_xticks([xticks[0], xticks[-1]])
            ax.set_xticklabels([f"{xticks[0]:.0f}", f"{xticks[-1]:.0f}"])

    return ax


def plot_price_heatmap(
    network: pypsa.Network,
    carrier="AC",
    log_values=False,
    color_map="viridis",
    ax: plt.Axes = None,
) -> plt.Axes:
    """plot the price heat map (region vs time) for the given carrier

    Args:
        network (pypsa.Network): the pypsa network object
        carrier (str, optional): the carrier for which to get the price. Defaults to "AC".
        log_values (bool, optional): whether to use log scale for the prices. Defaults to False.
        color_map (str, optional): the color map to use. Defaults to "viridis".
        ax (plt.Axes, optional): the plotting axis. Defaults to None (new fig).

    Returns:
        plt.Axes: the axes for plotting
    """

    if not ax:
        fig, ax = plt.subplots(figsize=(20, 8))
    else:
        fig = ax.get_figure()

    carrier_buses = network.buses.carrier[network.buses.carrier == carrier].index.values
    nodal_prices = network.buses_t.marginal_price[carrier_buses]
    # Normalize nodal_prices with log transformation
    if log_values:
        # Avoid log(0) by clipping values to a minimum of 0.1
        normalized_prices = np.log(nodal_prices.clip(lower=0.1))
        label = "Log-Transformed Price [€/MWh]"
    else:
        normalized_prices = nodal_prices
        label = "Price [€/MWh]"
    # Create a heatmap of normalized nodal_prices
    sns.heatmap(
        normalized_prices.reset_index(drop=True).T,
        cmap=color_map,
        cbar_kws={"label": label},
        ax=ax,
    )

    # Customize the plot
    ax.set_title("Heatmap of Log-Transformed Nodal Prices")
    ax.set_xlabel("Time")
    ax.set_ylabel("Nodes")
    fig.tight_layout()

    return ax


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_snapshots",
            topology="current+FCG",
            # co2_pathway="exp175default",
            co2_pathway="SSP2-PkBudg1000-CHA-pypsaelh2",
            heating_demand="positive",
            configfiles=["resources/tmp/remind_coupled_cg.yaml"],
            planning_horizons="2055",
            winter_day1="12-10 21:00",  # mm-dd HH:MM
            winter_day2="12-17 12:00",  # mm-dd HH:MM
            spring_day1="03-31 21:00",  # mm-dd HH:MM
            spring_day2="04-06 12:00",  # mm-dd HH:MM
            summer_day1="07-15 21:00",  # mm-dd HH:MM
            summer_day2="07-22 12:00",  # mm-dd HH:MM
        )

    YEAR = snakemake.wildcards.planning_horizons

    configure_logging(snakemake)
    set_plot_test_backend(snakemake.config)

    set_plot_style(
        style_config_file=snakemake.config["plotting"]["network_style_config_file"],
        base_styles=["classic", "seaborn-v0_8-white"],
    )

    config = snakemake.config
    carriers = ["AC"]
    if config.get("heat_coupling", False):
        carriers.append("heat")

    if not os.path.isdir(snakemake.output.outp_dir):
        makedirs(snakemake.output.outp_dir)

    n = pypsa.Network(snakemake.input.network)
    fix_network_names_colors(n, snakemake.config)

    for carrier in carriers:
        fig, ax = plt.subplots(figsize=(16, 8))
        plot_energy_balance(
            n,
            config["plotting"],
            bus_carrier=carrier,
            start_date=f"{YEAR}-{snakemake.params.spring_day1}",
            end_date=f"{YEAR}-{snakemake.params.spring_day2}",
            ax=ax,
        )
        outp = os.path.join(snakemake.output.outp_dir, f"balance_spring_{carrier}.png")
        fig.savefig(outp)

        fig, ax = plt.subplots(figsize=(16, 8))
        plot_energy_balance(
            n,
            config["plotting"],
            bus_carrier=carrier,
            start_date=f"{YEAR}-{snakemake.params.winter_day1}",
            end_date=f"{YEAR}-{snakemake.params.winter_day2}",
            ax=ax,
        )
        outp = os.path.join(snakemake.output.outp_dir, f"balance_winter_{carrier}.png")
        fig.savefig(outp)

        plot_energy_balance(
            n,
            config["plotting"],
            bus_carrier=carrier,
            start_date=f"{YEAR}-{snakemake.params.summer_day1}",
            end_date=f"{YEAR}-{snakemake.params.summer_day2}",
            ax=ax,
        )
        outp = os.path.join(snakemake.output.outp_dir, f"balance_summer_{carrier}.png")
        fig.savefig(outp)

    logger.info(f"Successfully plotted time series for carriers: {", ".join(carriers)}")
