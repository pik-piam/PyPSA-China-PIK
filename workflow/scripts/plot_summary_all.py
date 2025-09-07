# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

# BEWARE YEARS ARE HARDCODED BASED ON THE PLANNING_HORIZONS LINE IN THE MAKESUMMARY OUTPUT....

"""
Plots energy and cost summaries for solved networks.
This script collects functions that plot across planning horizons.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _helpers import configure_logging, mock_snakemake, set_plot_test_backend
from _plot_utilities import label_stacked_bars, set_plot_style
from constants import (
    COST_UNIT,
    PLOT_CAP_LABEL,
    PLOT_CAP_UNITS,
    PLOT_CO2_LABEL,
    PLOT_CO2_UNITS,
    PLOT_COST_UNITS,
    PLOT_SUPPLY_LABEL,
    PLOT_SUPPLY_UNITS,
)

logger = logging.getLogger(__name__)


# consolidate and rename
def rename_techs(label: pd.Index) -> pd.Index:
    """Rename techs into grouped categories

    Args:
        label (pd.Index | iterable): the index techs to rename
    Returns:
        pd.Index | iterable: the renamed index / iterable
    """
    prefix_to_remove = [
        "central ",
        "decentral ",
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "H2": "H2",
        "coal cc": "CC",
    }
    rename_if_contains = ["gas", "coal"]
    rename = {
        "solar": "solar PV",
        "Sabatier": "methanation",
        "offwind": "offshore wind",
        "onwind": "onshore wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "pumped hydro storage",
        "hydro_inflow": "hydroelectricity",
        "stations": "hydroelectricity",
        "AC": "transmission lines",
        "CO2 capture": "biomass carbon capture",
        "CC": "coal carbon capture",
        "battery": "battery",
    }

    for ptr in prefix_to_remove:
        if label[: len(ptr)] == ptr:
            label = label[len(ptr) :]

    for old, new in rename_if_contains_dict.items():
        if old in label:
            label = new

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old, new in rename.items():
        if old == label:
            label = new
    return label


def plot_pathway_costs(
    file_list: list,
    config: dict,
    social_discount_rate=0.0,
    fig_name: os.PathLike = None,
):
    """Plot the costs

    Args:
        file_list (list): the input csvs from make_summary
        config (dict): the configuration for plotting (snakemake.config["plotting"])
        social_discount_rate (float, optional): the social discount rate (0.02). Defaults to 0.0.
        fig_name (os.PathLike, optional): the figure name. Defaults to None.
    """
    # all years in one df
    df = pd.DataFrame()
    for results_file in file_list:
        cost_df = pd.read_csv(results_file, index_col=list(range(3)), header=[1])
        df_ = cost_df.groupby(cost_df.index.get_level_values(2)).sum()
        # do this here so aggregate costs of small items only for that year
        df_ = df_ * COST_UNIT / PLOT_COST_UNITS
        df_ = df_.groupby(df_.index.map(rename_techs)).sum()
        to_drop = df_.index[df_.max(axis=1) < config["costs_threshold"] / PLOT_COST_UNITS]
        df_.loc["Other"] = df_.loc[to_drop].sum(axis=0)
        df_ = df_.drop(to_drop)
        df = pd.concat([df_, df], axis=1)

    df.fillna(0, inplace=True)
    df.rename(columns={int(y): y for y in df.columns}, inplace=True)
    df.sort_index(axis=1, inplace=True, ascending=True)

    # apply social discount rate
    if social_discount_rate > 0:
        base_year = min([int(y) for y in df.columns])
        df = df.apply(
            lambda x: x / (1 + social_discount_rate) ** (int(x.name) - base_year),
            axis=0,
        )
    elif social_discount_rate < 0:
        raise ValueError("Social discount rate must be positive")

    preferred_order = pd.Index(config["preferred_order"])
    new_index = preferred_order.intersection(df.index).append(df.index.difference(preferred_order))
    logger.info(f"Missing technologies in preferred order: {df.index.difference(preferred_order)}")
    fig, ax = plt.subplots()
    fig.set_size_inches((12, 8))

    df.loc[new_index].T.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=[config["tech_colors"][i] for i in new_index],
    )

    handles, labels = ax.get_legend_handles_labels()

    ax.set_ylim([0, df.sum(axis=0).max() * 1.1])
    ax.set_ylabel("System Cost [EUR billion per year]")
    ax.set_xlabel("")
    ax.grid(axis="y")
    # TODO fix this - doesnt work with non-constant interval
    ax.annotate(
        f"Total cost in bn Eur: {df.sum().sum() * 5:.2f}",
        xy=(0.75, 0.9),
        color="darkgray",
        xycoords="axes fraction",
        ha="right",
        va="top",
    )

    ax.legend(
        handles,
        [l.title() for l in labels],
        ncol=1,
        bbox_to_anchor=[1, 1],
        loc="upper left",
    )

    fig.tight_layout()

    if fig_name is not None:
        fig.savefig(fig_name, transparent=config["transparent"])


def plot_pathway_capacities(
    file_list: list, config: dict, plot_heat=True, plot_h2=True, fig_name=None
):
    """Plot the capacities

    Args:
        file_list (list): the input csvs from make_summary
        config (dict): the configuration for plotting (snakemake.config["plotting"])
        fig_name (os.PathLike, optional): the figure name. Defaults to None.
        plot_heat (bool, optional): plot heat capacities. Defaults to True.
        plot_h2 (bool, optional): plot hydrogen capacities. Defaults to True.
    """

    caps_heat, caps_h2, caps_ac, caps_stores = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    # loop over each year result
    for results_file in file_list:
        cap_df = pd.read_csv(results_file, index_col=list(range(4)), header=[1, 2])
        # format table
        cap_df.index.names = ["component", "carrier", "bus_carrier", "end_carrier"]
        year = cap_df.columns.get_level_values(0)[0]
        cap_df = cap_df.droplevel(0, axis=1).rename(columns={"Unnamed: 4_level_1": year})
        cap_df /= PLOT_CAP_UNITS
        if "Load Shedding" in cap_df.index.get_level_values("carrier"):
            cap_df.drop("Load Shedding", level="carrier", inplace=True)

        # get stores relevant for reporting according to config, use later
        stores = (
            cap_df[
                (cap_df.index.get_level_values(0) == "Store")
                & (cap_df.index.get_level_values(1).isin(config["capacity_tracking"]["stores"]))
            ]
            .groupby(level=1)
            .sum()
        )

        # drop stores from cap df
        cap_df.drop(cap_df[cap_df.index.get_level_values(0) == "Store"].index, inplace=True)
        # drop charger/dischargers for stores
        cap_df.drop(
            cap_df[
                (cap_df.index.get_level_values(0) == "Link")
                & (cap_df.index.get_level_values(1).isin(config["capacity_tracking"]["drop_links"]))
            ].index,
            inplace=True,
        )

        # select AC (important for links) and group
        cap_ac = cap_df.reset_index().query(
            "bus_carrier == 'AC' | carrier =='AC' | end_carrier =='AC'"
        )
        cap_ac = cap_ac.groupby("carrier").sum()[year]

        cap_h2 = pd.DataFrame()
        if plot_h2:
            cap_h2 = cap_df.reset_index().query(
                "bus_carrier == 'H2' | carrier =='H2' | end_carrier =='H2'"
            )
            cap_h2 = cap_h2.groupby("carrier").sum()[year]
            if caps_h2.empty:
                caps_h2 = cap_h2
            else:
                caps_h2 = pd.concat([caps_h2, cap_h2], axis=1).fillna(0)
        if plot_heat:
            # TODO issue for CHP in case of several end buses. Bus2 will not be caught
            cap_heat = cap_df.reset_index().query(
                "bus_carrier == 'heat' | carrier =='heat' | end_carrier =='heat'"
            )
            cap_heat = cap_heat.groupby("carrier").sum()[year]
            if caps_heat.empty:
                caps_heat = cap_heat
            else:
                caps_heat = pd.concat([caps_heat, cap_h2], axis=1).fillna(0)

        caps_stores = pd.concat([stores, caps_stores], axis=1).fillna(0)
        if caps_ac.empty:
            caps_ac = cap_ac
        else:
            caps_ac = pd.concat([cap_ac, caps_ac], axis=1).fillna(0)

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches((14, 15))

    for i, capacity_df in enumerate([caps_ac, caps_heat, caps_stores, caps_h2]):
        if capacity_df.empty:
            continue
        if isinstance(capacity_df, pd.Series):
            capacity_df = capacity_df.to_frame()
        k, j = divmod(i, 2)
        ax = axes[k, j]
        preferred_order = pd.Index(config["preferred_order"])
        new_index = preferred_order.intersection(capacity_df.index).append(
            capacity_df.index.difference(preferred_order)
        )
        new_columns = capacity_df.columns.sort_values()

        logger.debug(capacity_df.loc[new_index, new_columns])

        capacity_df.loc[new_index, new_columns].T.plot(
            kind="bar",
            ax=ax,
            stacked=True,
            color=[config["tech_colors"][i] for i in new_index],
        )

        handles, labels = ax.get_legend_handles_labels()

        handles.reverse()
        labels.reverse()

        if capacity_df.index.difference(caps_stores.index).empty:
            ax.set_ylabel(f"Installed Storage Capacity [{PLOT_CAP_LABEL}h]")
        else:
            ax.set_ylabel(f"Installed Capacity [{PLOT_CAP_LABEL}]")
        ax.set_ylim([0, capacity_df.sum(axis=0).max() * 1.1])
        ax.set_xlabel("")
        ax.grid(axis="y")
        # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))
        ax.legend(handles, labels, ncol=2, bbox_to_anchor=(0.5, -0.15), loc="upper center")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.42)

    if fig_name is not None:
        fig.savefig(fig_name, transparent=config["transparent"])

    return fig, axes


def plot_expanded_capacities(
    file_list: list, config: dict, plot_heat=False, plot_h2=True, fig_name=None
):
    """Plot the expanded capacities

    Args:
        file_list (list): the input csvs from make_summary
        config (dict): the configuration for plotting (snakemake.config["plotting"])
        fig_name (os.PathLike, optional): the figure name. Defaults to None.
        plot_heat (bool, optional): plot heat capacities. Defaults to True.
        plot_h2 (bool, optional): plot hydrogen capacities. Defaults to True.
    """

    fig, axes = plot_pathway_capacities(file_list, config, plot_heat, plot_h2, fig_name=None)
    for i, ax in enumerate(axes.flat):
        ylabel = ax.get_ylabel()
        if "Installed" in ylabel:
            ax.set_ylabel(ylabel.replace("Installed", "Additional"))

    if fig_name is not None:
        fig.savefig(fig_name, transparent=config["transparent"])


def plot_energy(file_list: list, config: dict, fig_name=None):
    """Plot the energy production and consumption

    Args:
        file_list (list): the input csvs
        config (dict): the configuration for plotting (snamkemake.config["plotting"])
        fig_name (os.PathLike, optional): the figure name. Defaults to None.
    """
    energy_df = pd.DataFrame()
    for results_file in file_list:
        en_df = pd.read_csv(results_file, index_col=list(range(2)), header=[1])
        df_ = en_df.groupby(en_df.index.get_level_values(1)).sum()
        # do this here so aggregate costs of small items only for that year
        # convert MWh to TWh
        df_ = df_ / PLOT_SUPPLY_UNITS
        df_ = df_.groupby(df_.index.map(rename_techs)).sum()
        to_drop = df_.index[df_.max(axis=1) < config["energy_threshold"] / PLOT_SUPPLY_UNITS]
        df_.loc["Other"] = df_.loc[to_drop].sum(axis=0)
        df_ = df_.drop(to_drop)

        energy_df = pd.concat([df_, energy_df], axis=1)
    energy_df.fillna(0, inplace=True)
    energy_df.sort_index(axis=1, inplace=True)

    logger.info(f"Total energy of {round(energy_df.sum()[0])} {PLOT_SUPPLY_LABEL}/a")
    preferred_order = pd.Index(config["preferred_order"])
    new_index = preferred_order.intersection(energy_df.index).append(
        energy_df.index.difference(preferred_order)
    )
    new_columns = energy_df.columns.sort_values()

    fig, ax = plt.subplots()
    fig.set_size_inches((12, 8))

    logger.debug(energy_df.loc[new_index, new_columns])

    energy_df.loc[new_index, new_columns].T.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=[config["tech_colors"][i] for i in new_index],
    )

    handles, labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_ylim([0, energy_df.sum(axis=0).max() * 1.1])
    ax.set_ylabel(f"Energy [{PLOT_SUPPLY_LABEL}/a]")
    ax.set_xlabel("")
    ax.grid(axis="y")
    ax.legend(handles, labels, ncol=1, bbox_to_anchor=[1, 1], loc="upper left")
    fig.tight_layout()

    if fig_name is not None:
        fig.savefig(fig_name, transparent=config["transparent"])


def plot_electricty_heat_balance(
    file_list: list[os.PathLike], config: dict, fig_dir=None, plot_heat=True
):
    """Plot the energy production and consumption

    Args:
        file_list (list): the input csvs  from make_dirs([year/supply_energy.csv])
        config (dict): the configuration for plotting (snamkemake.config["plotting"])
        fig_dir (os.PathLike, optional): the figure name. Defaults to None.
        plot_heat (bool, optional): plot heat balances. Defaults to True.
    """
    elec_df = pd.DataFrame()
    heat_df = pd.DataFrame()

    for results_file in file_list:
        balance_df = pd.read_csv(results_file, index_col=list(range(2)), header=[1])
        elec = balance_df.loc["AC"].copy()
        elec.set_index(elec.columns[0], inplace=True)
        elec.rename(index={"-": "electric load"}, inplace=True)
        elec.index.rename("carrier", inplace=True)
        # this groups subgroups of the same carrier. For example, baseyar hydro = link from dams
        # but new hydro is generator from province
        elec = elec.groupby(elec.index).sum()
        to_drop = elec.index[
            elec.max(axis=1).abs() < config["energy_threshold"] / PLOT_SUPPLY_UNITS
        ]
        elec.loc["Other"] = elec.loc[to_drop].sum(axis=0)
        elec.drop(to_drop, inplace=True)
        elec_df = pd.concat([elec, elec_df], axis=1)

        if plot_heat:
            heat = balance_df.loc["heat"].copy()
            heat.set_index(heat.columns[0], inplace=True)
            heat.rename(index={"-": "heat load"}, inplace=True)
            heat.index.rename("carrier", inplace=True)
            heat = heat.groupby(heat.index).sum()
            to_drop = heat.index[
                heat.max(axis=1).abs() < config["energy_threshold"] / PLOT_SUPPLY_UNITS
            ]
            heat.loc["Other"] = heat.loc[to_drop].sum(axis=0)
            heat.drop(to_drop, inplace=True)
            heat_df = pd.concat([heat, heat_df], axis=1)
        else:
            heat_df = pd.DataFrame()

    elec_df.fillna(0, inplace=True)
    elec_df.sort_index(axis=1, inplace=True, ascending=True)
    elec_df = elec_df / PLOT_SUPPLY_UNITS

    heat_df.fillna(0, inplace=True)
    heat_df.sort_index(axis=1, inplace=True, ascending=True)
    heat_df = heat_df / PLOT_SUPPLY_UNITS

    # # split into consumption and generation
    el_gen = elec_df.where(elec_df >= 0).dropna(axis=0, how="all").fillna(0)
    el_con = elec_df.where(elec_df < 0).dropna(axis=0, how="all").fillna(0)
    heat_gen = heat_df.where(heat_df > 0).dropna(axis=0, how="all").fillna(0)
    heat_con = heat_df.where(heat_df < 0).dropna(axis=0, how="all").fillna(0)

    # group identical values
    el_con = el_con.groupby(el_con.index).sum()
    el_gen = el_gen.groupby(el_gen.index).sum()
    heat_con = heat_con.groupby(heat_con.index).sum()
    heat_gen = heat_gen.groupby(heat_gen.index).sum()

    logger.info(f"Total energy of {round(elec_df.sum()[0])} TWh/a")

    # ===========        electricity =================
    fig, ax = plt.subplots()
    fig.set_size_inches((12, 8))

    preferred_order = pd.Index(config["preferred_order"])
    for df in [el_gen, el_con]:
        new_index = preferred_order.intersection(df.index).append(
            df.index.difference(preferred_order)
        )
        logger.info(
            f"Missing technologies in preferred order: {df.index.difference(preferred_order)}"
        )

        colors = pd.DataFrame(
            new_index.map(config["tech_colors"]), index=new_index, columns=["color"]
        )
        colors.fillna(NAN_COLOR, inplace=True)
        df.loc[new_index].T.plot(
            kind="bar",
            ax=ax,
            stacked=True,
            color=colors["color"],
        )

        handles, labels = ax.get_legend_handles_labels()
        handles.reverse()
        labels.reverse()

    ax.set_ylim([el_con.sum(axis=0).min() * 1.1, el_gen.sum(axis=0).max() * 1.1])
    ax.set_ylabel("Energy [TWh/a]")
    ax.set_xlabel("")
    ax.grid(axis="y")
    ax.legend(
        handles,
        [l.title() for l in labels],
        ncol=1,
        bbox_to_anchor=[1, 1],
        loc="upper left",
    )

    if config.get("add_bar_labels", False):
        label_stacked_bars(ax, len(el_gen.columns))

    fig.tight_layout()

    if fig_dir is not None:
        fig.savefig(os.path.join(fig_dir, "elec_balance.png"), transparent=config["transparent"])

    # =================     heat     =================
    fig, ax = plt.subplots()
    fig.set_size_inches((12, 8))

    for df in [heat_gen, heat_con]:
        if not plot_heat:
            break

        preferred_order = pd.Index(config["preferred_order"])
        new_index = preferred_order.intersection(df.index).append(
            df.index.difference(preferred_order)
        )
        colors = pd.DataFrame(
            new_index.map(config["tech_colors"]), index=new_index, columns=["color"]
        )
        colors.fillna(NAN_COLOR, inplace=True)
        df.loc[new_index].T.plot(
            kind="bar",
            ax=ax,
            stacked=True,
            color=colors["color"],
        )

        handles, labels = ax.get_legend_handles_labels()
        handles.reverse()
        labels.reverse()

    if plot_heat:
        ax.set_ylim([heat_con.sum(axis=0).min() * 1.1, heat_gen.sum(axis=0).max() * 1.1])
        ax.set_ylabel("Energy [TWh/a]")
        ax.set_xlabel("")
        ax.grid(axis="y")
        ax.legend(handles, labels, ncol=1, bbox_to_anchor=[1, 1], loc="upper left")
        fig.tight_layout()

        if fig_dir is not None:
            fig.savefig(
                os.path.join(fig_dir, "heat_balance.png"),
                transparent=config["transparent"],
            )


def plot_capacity_factors(
    file_list: list, config: dict, techs: list, fig_name=None, ax: object = None
):
    """Plot evolution of capacity factors for the given technologies

    Args:
        file_list (list): the input csvs from make_summary
        config (dict): the configuration for plotting (snakemake.config["plotting"])
        techs (list): the technologies to plot
        fig_name (os.PathLike, optional): the figure name. Defaults to None.
        ax (matplotlib.axes.Axes, optional): the axes to plot on. Defaults to None.
    """

    capfacs_df = pd.DataFrame()
    for results_file in file_list:
        df_year = pd.read_csv(results_file, index_col=list(range(2)), header=[1]).T
        capfacs_df = pd.concat([df_year, capfacs_df])

    if ("links", "battery") in capfacs_df.columns:
        capfacs_df.loc[:, ("links", "battery charger")] = capfacs_df.loc[:, ("links", "battery")]
        capfacs_df.drop(columns=("links", "battery"), inplace=True)

    capfacs_df = capfacs_df.droplevel(0, axis=1).fillna(0)
    capfacs_df.sort_index(axis=0, inplace=True)

    invalid = [t for t in techs if t not in capfacs_df.columns]
    logger.warning(f"Technologies {invalid} not found in capacity factors data. Skipping them.")
    valid_techs = [t for t in techs if t in capfacs_df.columns]
    capfacs_df = capfacs_df[valid_techs]

    if not ax:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    fig.set_size_inches((12, 8))

    colors = pd.Series(config["tech_colors"], index=capfacs_df.columns)
    # missing color may have had nice name, else NAN default
    nice_name_colors = pd.Series(
        config["tech_colors"], index=capfacs_df.columns.map(config["nice_names"])
    ).dropna()
    colors = colors.fillna(nice_name_colors).fillna(NAN_COLOR)

    capfacs_df.plot(
        ax=ax,
        kind="line",
        color=colors,
        linewidth=3,
        marker="o",
    )
    ax.set_ylim([0, capfacs_df.max().max() * 1.1])
    ax.set_ylabel("capacity factor")
    ax.set_xlabel("")
    ax.grid(axis="y")

    handles, labels = ax.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    ax.legend(handles, labels, ncol=1, bbox_to_anchor=[1, 1], loc="upper left")
    fig.tight_layout()

    if fig_name is not None:
        fig.savefig(fig_name, transparent=False)


def plot_prices(
    file_list: list,
    config: dict,
    fig_name=None,
    absolute=False,
    ax: object = None,
    unit="€/MWh",
    **kwargs,
):
    """Plot the prices

    Args:
        file_list (list): the input csvs from make_summary
        config (dict): the configuration for plotting (snakemake.config["plotting"])
        fig_name (os.PathLike, optional): the figure name. Defaults to None.
        absolute (bool, optional): plot absolute prices. Defaults to False.
        ax (matplotlib.axes.Axes, optional): the axes to plot on. Defaults to None.
        unit (str, optional): the unit of the prices. Defaults to "€/MWh".
    """
    prices_df = pd.DataFrame()
    for results_file in file_list:
        df_year = pd.read_csv(results_file, index_col=list(range(1)), header=[1]).T

        prices_df = pd.concat([df_year, prices_df])
    prices_df.sort_index(axis=0, inplace=True)
    if not ax:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    fig.set_size_inches((12, 8))

    colors = config["tech_colors"]

    if absolute:
        prices_df = prices_df.abs()

    defaults = {"lw": 3, "marker": "o", "markersize": 5, "alpha": 0.8}
    if "linewidth" in kwargs:
        kwargs["lw"] = kwargs.pop("linewidth")
    defaults.update(kwargs)
    prices_df.plot(
        ax=ax,
        kind="line",
        color=[colors[k] if k in colors else "k" for k in prices_df.columns],
        **defaults,
    )
    min_ = prices_df.min().min()
    if np.sign(min_) < 0:
        min_ *= 1.1
    else:
        min_ *= 0.9
    ax.set_ylim([min_, prices_df.max().max() * 1.1])
    ax.set_ylabel(f"Prices [{unit}]")
    ax.set_xlabel("")
    ax.grid(axis="y")

    handles, labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()
    ax.legend(handles, labels, ncol=1, bbox_to_anchor=[1, 1], loc="upper left")
    fig.tight_layout()

    if fig_name is not None:
        fig.savefig(fig_name, transparent=False)


def plot_pathway_co2(file_list: list, config: dict, fig_name=None):
    """Plot the CO2 pathway balance and totals

    Args:
        file_list (list): the input csvs
        config (dict): the plotting configuration
        fig_name (_type_, optional): _description_. Defaults to None.
    """

    co2_balance_df = pd.DataFrame()
    for results_file in file_list:
        df_year = pd.read_csv(results_file, index_col=list(range(1)), header=[1]).T
        co2_balance_df = pd.concat([df_year, co2_balance_df])

    co2_balance_df.sort_index(axis=0, inplace=True)

    fig, ax = plt.subplots()
    bar_width = 0.6
    colors = co2_balance_df.T.index.map(config["tech_colors"]).values
    co2_balance_df = co2_balance_df / PLOT_CO2_UNITS
    co2_balance_df.plot(
        kind="bar",
        stacked=True,
        width=bar_width,
        color=pd.Series(colors).fillna(NAN_COLOR),
        ax=ax,
    )
    bar_centers = np.unique([patch.get_x() + bar_width / 2 for patch in ax.patches])
    ax.plot(
        bar_centers,
        co2_balance_df.sum(axis=1).values,
        color="black",
        marker="D",
        markersize=10,
        lw=3,
        label="Total",
    )
    ax.set_ylabel(PLOT_CO2_LABEL)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=1, bbox_to_anchor=[1, 1], loc="upper left")
    ax.set_ylim([0, co2_balance_df.sum(axis=1).max() * 1.1])
    fig.tight_layout()
    if fig_name is not None:
        fig.savefig(fig_name, transparent=config["transparent"])


def plot_co2_prices(co2_prices: dict, config: dict, fig_name=None):
    """Plot the CO2 prices
    Args:
        co2_prices (dict): the CO2 prices per year (from the config)
        config (dict): the plotting configuration
        fig_name (os.PathLike, optional): the figure name. Defaults to None.
    """
    fig, ax = plt.subplots()
    fig.set_size_inches((12, 8))

    ax.plot(
        co2_prices.keys(),
        np.abs(list(co2_prices.values())),
        marker="o",
        color="black",
        lw=2,
    )
    ax.set_ylabel("CO2 price")
    ax.set_xlabel("Year")
    ax.plot(co2_prices.keys(), co2_prices.values(), marker="o", color="black", lw=2)

    fig.tight_layout()
    if fig_name is not None:
        fig.savefig(fig_name, transparent=config["transparent"])


def plot_co2_shadow_price(file_list: list, config: dict, fig_name=None):
    """Plot the co2 price

    Args:
        file_list (list): the input csvs from make_summaries
        config (dict): the snakemake configuration
        fig_name (os.PathLike, optional): the figure name. Defaults to None.
    """
    co2_prices = {}
    co2_budget = {}
    for i, results_file in enumerate(file_list):
        df_metrics = pd.read_csv(results_file, index_col=list(range(1)), header=[1])
        co2_prices.update(dict(df_metrics.loc["co2_shadow"]))
        co2_budget.update(dict(df_metrics.loc["co2_budget"]))

    fig, ax = plt.subplots()
    fig.set_size_inches((12, 8))

    ax.plot(
        co2_prices.keys(),
        np.abs(list(co2_prices.values())),
        marker="o",
        color="black",
        lw=2,
    )
    ax.set_ylabel("CO2 Shadow price")
    ax.set_xlabel("Year")

    ax2 = ax.twinx()
    ax2.plot(
        co2_budget.keys(),
        [v / PLOT_CO2_UNITS for v in co2_budget.values()],
        marker="D",
        color="blue",
        lw=2,
    )
    ax2.set_ylabel(f"CO2 Budget [{PLOT_CO2_LABEL}]", color="blue")
    ax2.tick_params(axis="y", colors="blue")

    fig.tight_layout()

    if fig_name is not None:
        fig.savefig(fig_name, transparent=config["transparent"])


def plot_investments(file_list: list, config: dict, fig_name=None, ax: object = None):
    pass


# TODO move to a separate rule
def write_data(data_paths: dict, outp_dir: os.PathLike):
    """Write some selected data

    Args:
        data_paths (dict): the paths to the summary data (different per year and type)
        outp_dir (os.PathLike): target file (summary dir)
    """
    # make a summary of the co2 prices
    co2_prices = {}
    co2_budget = {}
    for i, results_file in enumerate(data_paths["co2_price"]):
        df_metrics = pd.read_csv(results_file, index_col=list(range(1)), header=[1])
        co2_prices.update(dict(df_metrics.loc["co2_shadow"]))
        co2_budget.update(dict(df_metrics.loc["co2_budget"]))
    years = list(co2_budget.keys())
    co2_df = pd.DataFrame(
        {
            "Year": years,
            "CO2 Budget": [co2_budget[year] for year in years],
            "CO2 Shadow Price": [co2_prices[year] * -1 for year in years],
        }
    )
    outp_p = os.path.join(outp_dir, "co2_prices.csv")
    co2_df.to_csv(outp_p, index=False)

    df = pd.DataFrame()
    for results_file in data_paths["costs"]:
        cost_df = pd.read_csv(results_file, index_col=list(range(3)), header=[1])
        df_ = cost_df.groupby(level=[1, 2]).sum()
        df_ = df_ * COST_UNIT / PLOT_COST_UNITS
        df = pd.concat([df_, df], axis=1)
    df.to_csv(os.path.join(outp_dir, "pathway_costs_not_discounted.csv"))

    prices_df = pd.DataFrame()
    for results_file in data_paths["weighted_prices"]:
        df_year = pd.read_csv(results_file, index_col=list(range(1)), header=[1]).T

        prices_df = pd.concat([df_year, prices_df])
    prices_df.to_csv(os.path.join(outp_dir, "weighted_prices.csv"))


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_summary",
            topology="current+FCG",
            # co2_pathway="exp175default",
            co2_pathway="SSP2-PkBudg1000-pseudo-coupled",
            heating_demand="positive",
            configfiles="resources/tmp/pseudo_coupled_cg.yml",
        )

    configure_logging(snakemake)
    set_plot_test_backend(snakemake.config)
    logger.info(snakemake.input)

    set_plot_style(
        style_config_file="./config/plotting_styles/default_style.mplstyle",
        base_styles=["ggplot"],
    )

    config = snakemake.config
    wildcards = snakemake.wildcards
    logs = snakemake.log
    output_paths = snakemake.output
    paths = snakemake.input

    co2_pathway = config["co2_scenarios"][wildcards.co2_pathway]
    if co2_pathway["control"] == "price":
        co2_prices = co2_pathway["pathway"]
    else:
        co2_prices = None

    plot_heat = config.get("heat_coupling", False)
    plot_h2 = config.get("h2_coupling", True)
    NAN_COLOR = config["plotting"]["nan_color"]
    data_paths = {
        "energy": [os.path.join(p, "energy.csv") for p in paths],
        "costs": [os.path.join(p, "costs.csv") for p in paths],
        "co2_price": [os.path.join(p, "metrics.csv") for p in paths],
        "time_averaged_prices": [os.path.join(p, "time_averaged_prices.csv") for p in paths],
        "weighted_prices": [os.path.join(p, "weighted_prices.csv") for p in paths],
        "co2_balance": [os.path.join(p, "co2_balance.csv") for p in paths],
        "energy_supply": [os.path.join(p, "supply_energy.csv") for p in paths],
        "capacity": [os.path.join(p, "capacities.csv") for p in paths],
        "expanded_capacity": [os.path.join(p, "capacities_expanded.csv") for p in paths],
        "capacity_factors": [os.path.join(p, "cfs.csv") for p in paths],
    }

    sdr = float(config["costs"]["social_discount_rate"])
    plot_capacity_factors(
        data_paths["capacity_factors"],
        config["plotting"],
        techs=[
            "solar",
            "onwind",
            "offwind",
            # "battery",
            "battery discharger",
            "coal",
            "coal-CCS",
            "hydroelectricity",
            "gas OCGT",
            "CCGT-CCS",
            "H2 Electrolysis",
        ],
        fig_name=os.path.dirname(output_paths.costs) + "/capacity_factors.png",
    )
    plot_pathway_costs(
        data_paths["costs"],
        config["plotting"],
        social_discount_rate=sdr,
        fig_name=output_paths.costs,
    )
    plot_pathway_capacities(
        data_paths["capacity"],
        config["plotting"],
        fig_name=os.path.dirname(output_paths.costs) + "/capacities.png",
        plot_heat=plot_heat,
        plot_h2=plot_h2,
    )
    plot_expanded_capacities(
        data_paths["expanded_capacity"],
        config["plotting"],
        fig_name=os.path.dirname(output_paths.costs) + "/capacities_expanded.png",
        plot_heat=plot_heat,
        plot_h2=plot_h2,
    )
    # plot_energy(data_paths["energy"], config["plotting"], fig_name=output_paths.energy)
    plot_electricty_heat_balance(
        data_paths["energy_supply"],
        config["plotting"],
        fig_dir=os.path.dirname(output_paths.costs),
        plot_heat=plot_heat,
    )
    plot_prices(
        data_paths["time_averaged_prices"],
        config["plotting"],
        fig_name=os.path.dirname(output_paths.costs) + "/time_averaged_prices.png",
    )

    plot_prices(
        data_paths["weighted_prices"],
        config["plotting"],
        fig_name=os.path.dirname(output_paths.costs) + "/weighted_prices.png",
        absolute=True,
    )
    plot_co2_shadow_price(
        data_paths["co2_price"],
        config["plotting"],
        fig_name=os.path.dirname(output_paths.costs) + "/co2_shadow_prices.png",
    )

    if co2_prices is not None:
        plot_co2_prices(
            {k: v for k, v in co2_prices.items() if k in config["scenario"]["planning_horizons"]},
            config["plotting"],
            fig_name=os.path.dirname(output_paths.costs) + "/co2_prices.png",
        )

    plot_pathway_co2(
        data_paths["co2_balance"],
        config["plotting"],
        fig_name=os.path.dirname(output_paths.costs) + "/co2_balance.png",
    )

    logger.info(f"Successfully plotted summary for {wildcards}")

    data_dir = os.path.dirname(paths[0])
    write_data(data_paths, data_dir)
