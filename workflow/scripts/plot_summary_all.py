# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

# BEWARE YEARS ARE HARDCODED BASED ON THE PLANNING_HORIZONS LINE IN THE MAKESUMMARY OUTPUT....

"""
Plots energy and cost summaries for solved networks.
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from _helpers import configure_logging, mock_snakemake, set_plot_test_backend
from constants import (
    PLOT_COST_UNITS,
    COST_UNIT,
    PLOT_CO2_UNITS,
    PLOT_CO2_LABEL,
    PLOT_SUPPLY_UNITS,
    PLOT_SUPPLY_LABEL,
    PLOT_CAP_UNITS,
    PLOT_CAP_LABEL,
)
from _plot_utilities import set_plot_style

logger = logging.getLogger(__name__)

set_plot_test_backend()


# consolidate and rename
def rename_techs(label):
    prefix_to_remove = [
        "central ",
        "decentral ",
    ]

    rename_if_contains_dict = {"water tanks": "hot water storage", "H2": "H2", "coal cc": "CC"}
    rename_if_contains = ["gas", "coal"]
    rename = {
        "solar": "solar PV",
        "Sabatier": "methanation",
        "offwind": "offshore wind",
        "onwind": "onshore wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "hydroelectricity",
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
    file_list: list, config: dict, social_discount_rate=0.0, fig_name: os.PathLike = None
):
    """plot the costs

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
        # TODO centralise unit
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
        df = df.apply(lambda x: x / (1 + social_discount_rate) ** (int(x.name) - base_year), axis=0)
    elif social_discount_rate < 0:
        raise ValueError("Social discount rate must be positive")

    preferred_order = pd.Index(config["preferred_order"])
    new_index = preferred_order.intersection(df.index).append(df.index.difference(preferred_order))

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
        f"Total cost in bn Eur: {df.sum().sum()*5:.2f}",
        xy=(0.75, 0.9),
        color="darkgray",
        xycoords="axes fraction",
        ha="right",
        va="top",
    )

    ax.legend(handles, labels, ncol=1, bbox_to_anchor=[1, 1], loc="upper left")

    fig.tight_layout()

    if fig_name is not None:
        fig.savefig(fig_name, transparent=True)


def plot_pathway_capacities(file_list: list, config: dict, fig_name=None):
    """plot the capacities

    Args:
        file_list (list): the input csvs from make_summary
        config (dict): the configuration for plotting (snakemake.config["plotting"])
        fig_name (os.PathLike, optional): the figure name. Defaults to None.
    """

    caps_heat, caps_h2, caps_ac = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for results_file in file_list:
        cap_df = pd.read_csv(results_file, index_col=list(range(3)), header=[1])
        # cap_df.drop(index="component", level=0, inplace=True)
        cap_df /= PLOT_CAP_UNITS

        # drop charger/dischargers for stores
        cap_df.drop(
            cap_df[
                (cap_df.index.get_level_values(0) == "Link")
                & (cap_df.index.get_level_values(1).isin(config["capacity_tracking"]["drop_links"]))
            ].index,
            inplace=True,
        )

        # convert stores to relevant carrier
        ac_stores = (
            cap_df[
                (cap_df.index.get_level_values(0) == "Store")
                & (cap_df.index.get_level_values(1).isin(config["capacity_tracking"]["ac_stores"]))
            ]
            .groupby(level=1)
            .sum()
        )
        heat_stores = (
            cap_df[
                (cap_df.index.get_level_values(0) == "Store")
                & (
                    cap_df.index.get_level_values(1).isin(
                        config["capacity_tracking"]["heat_stores"]
                    )
                )
            ]
            .groupby(level=1)
            .sum()
        )
        # convert to GW
        cap_ac = cap_df.loc[cap_df.index.get_level_values(2) == "AC"].groupby(level=1).sum()
        cap_ac = pd.concat([cap_ac, ac_stores])
        cap_h2 = cap_df.loc[cap_df.index.get_level_values(2) == "H2"].groupby(level=1).sum()
        cap_heat = cap_df.loc[cap_df.index.get_level_values(2) == "heat"].groupby(level=1).sum()
        cap_heat = pd.concat([cap_heat, heat_stores])

        caps_heat = pd.concat([cap_heat, caps_heat], axis=1)
        caps_h2 = pd.concat([cap_h2, caps_h2], axis=1)
        caps_ac = pd.concat([cap_ac, caps_ac], axis=1)

    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches((14, 8))

    for i, capacity_df in enumerate([caps_ac, caps_heat, caps_h2]):
        ax = axes[i]
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

        ax.set_ylim([0, capacity_df.sum(axis=0).max() * 1.1])
        ax.set_ylabel(f"Installed Capacity [{PLOT_CAP_LABEL}]")
        ax.set_xlabel("")
        ax.grid(axis="y")
        # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))
        ax.legend(handles, labels, ncol=2, bbox_to_anchor=(0.5, -0.15), loc="upper center")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.42)

    if fig_name is not None:
        fig.savefig(fig_name, transparent=True)


def plot_energy(file_list: list, config: dict, fig_name=None):
    """plot the energy production and consumption

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
        fig.savefig(fig_name, transparent=True)


def plot_electricty_heat_balance(file_list: list[os.PathLike], config: dict, fig_dir=None):
    """plot the energy production and consumption

    Args:
        file_list (list): the input csvs  from make_dirs([year/supply_energy.csv])
        config (dict): the configuration for plotting (snamkemake.config["plotting"])
        fig_dir (os.PathLike, optional): the figure name. Defaults to None.
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

        heat = balance_df.loc["heat"].copy()
        heat.set_index(heat.columns[0], inplace=True)
        heat.rename(index={"-": "heat load"}, inplace=True)
        heat.index.rename("carrier", inplace=True)
        heat = heat.groupby(heat.index).sum()

        to_drop = elec.index[
            elec.max(axis=1).abs() < config["energy_threshold"] / PLOT_SUPPLY_UNITS
        ]
        elec.loc["Other"] = elec.loc[to_drop].sum(axis=0)
        elec.drop(to_drop, inplace=True)

        to_drop = heat.index[
            heat.max(axis=1).abs() < config["energy_threshold"] / PLOT_SUPPLY_UNITS
        ]
        heat.loc["Other"] = heat.loc[to_drop].sum(axis=0)
        heat.drop(to_drop, inplace=True)

        elec_df = pd.concat([elec, elec_df], axis=1)
        heat_df = pd.concat([heat, heat_df], axis=1)

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

    for df in [el_gen, el_con]:
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

    ax.set_ylim([el_con.sum(axis=0).min() * 1.1, el_gen.sum(axis=0).max() * 1.1])
    ax.set_ylabel("Energy [TWh/a]")
    ax.set_xlabel("")
    ax.grid(axis="y")
    ax.legend(handles, labels, ncol=1, bbox_to_anchor=[1, 1], loc="upper left")
    fig.tight_layout()

    if fig_dir is not None:
        fig.savefig(os.path.join(fig_dir, "elec_balance.png"), transparent=True)

    # =================     heat     =================
    fig, ax = plt.subplots()
    fig.set_size_inches((12, 8))

    for df in [heat_gen, heat_con]:
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

    ax.set_ylim([heat_con.sum(axis=0).min() * 1.1, heat_gen.sum(axis=0).max() * 1.1])
    ax.set_ylabel("Energy [TWh/a]")
    ax.set_xlabel("")
    ax.grid(axis="y")
    ax.legend(handles, labels, ncol=1, bbox_to_anchor=[1, 1], loc="upper left")
    fig.tight_layout()

    if fig_dir is not None:
        fig.savefig(os.path.join(fig_dir, "heat_balance.png"), transparent=True)


def plot_prices(file_list: list, config: dict, fig_name=None):
    """plot the prices

    Args:
        file_list (list): the input csvs from make_summary
        config (dict): the configuration for plotting
        fig_name (os.PathLike, optional): the figure name. Defaults to None.
    """
    prices_df = pd.DataFrame()
    for results_file in file_list:
        df_year = pd.read_csv(results_file, index_col=list(range(1)), header=[1]).T

        prices_df = pd.concat([df_year, prices_df])
    prices_df.sort_index(axis=0, inplace=True)
    fig, ax = plt.subplots()
    fig.set_size_inches((12, 8))

    colors = config["plotting"]["tech_colors"]

    prices_df.plot(
        ax=ax,
        kind="line",
        color=[colors[k] if k in colors else "k" for k in prices_df.columns],
        linewidth=3,
    )
    ax.set_ylim([prices_df.min().min() * 1.1, prices_df.max().max() * 1.1])
    ax.set_ylabel("prices [X/UNIT]")
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
    colors = co2_balance_df.T.index.map(config["plotting"]["tech_colors"]).values
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

    fig.tight_layout()
    if fig_name is not None:
        fig.savefig(fig_name, transparent=True)


def plot_co2_shadow_price(file_list: list, config: dict, fig_name=None):
    """plot the co2 price

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

    ax.plot(co2_prices.keys(), np.abs(list(co2_prices.values())), marker="o", color="black", lw=2)
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
        fig.savefig(fig_name, transparent=True)


if __name__ == "__main__":
    if "snakemake" not in globals():

        snakemake = mock_snakemake(
            "plot_summary",
            topology="current+FCG",
            pathway="exp175",
            heating_demand="positive",
            planning_horizons=[
                "2020",
                "2025",
                "2030",
                "2035",
                "2040",
                "2045",
                "2050",
                "2055",
                "2060",
            ],
        )

    configure_logging(snakemake)
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
    }

    sdr = float(config["costs"]["social_discount_rate"])
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
    )
    plot_energy(data_paths["energy"], config["plotting"], fig_name=output_paths.energy)
    plot_electricty_heat_balance(
        data_paths["energy_supply"],
        config["plotting"],
        fig_dir=os.path.dirname(output_paths.costs),
    )
    plot_prices(
        data_paths["time_averaged_prices"],
        config,
        fig_name=os.path.dirname(output_paths.costs) + "/time_averaged_prices.png",
    )

    plot_prices(
        data_paths["weighted_prices"],
        config,
        fig_name=os.path.dirname(output_paths.costs) + "/weighted_prices.png",
    )
    plot_co2_shadow_price(
        data_paths["co2_price"],
        config,
        fig_name=os.path.dirname(output_paths.costs) + "/co2_shadow_prices.png",
    )

    plot_pathway_co2(
        data_paths["co2_balance"],
        config,
        fig_name=os.path.dirname(output_paths.costs) + "/co2_balance.png",
    )

    logger.info(f"Successfully plotted summary for {wildcards}")
