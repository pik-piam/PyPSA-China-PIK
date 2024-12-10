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

from _helpers import configure_logging, mock_snakemake
from constants import PLOT_COST_UNITS, COST_UNIT
from _plot_utilities import set_plot_style

logger = logging.getLogger(__name__)


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
        results_file (list): the input csvs
        config (dict): the configuration for plotting (snakemake.config["plotting"])
        social_discount_rate (float, optional): the social discount rate (2pc->0.02). Defaults to 0.0.
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
        to_drop = df_.index[df_.max(axis=1) < config["costs_threshold"]]
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


def plot_energy(file_list: list, config: dict, fig_name=None):
    """plot the energy production and consumption

    Args:
        results_file (list): the input csvs
        config (dict): the configuration for plotting (snamkemake.config["plotting"])
        fig_name (os.PathLike, optional): the figure name. Defaults to None.
    """
    energy_df = pd.DataFrame()
    for results_file in file_list:
        cost_df = pd.read_csv(results_file, index_col=list(range(2)), header=[1])
        df_ = cost_df.groupby(cost_df.index.get_level_values(1)).sum()
        # do this here so aggregate costs of small items only for that year
        # TODO centralise unit
        # convert MWh to TWh
        df_ = df_ / 1e6
        df_ = df_.groupby(df_.index.map(rename_techs)).sum()
        to_drop = df_.index[df_.max(axis=1) < config["energy_threshold"]]
        df_.loc["Other"] = df_.loc[to_drop].sum(axis=0)
        df_ = df_.drop(to_drop)

        energy_df = pd.concat([df_, energy_df], axis=1)
    energy_df.fillna(0, inplace=True)
    energy_df.sort_index(axis=1, inplace=True)

    logger.info(f"Total energy of {round(energy_df.sum()[0])} TWh/a")
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
    ax.set_ylabel("Energy [TWh/a]")
    ax.set_xlabel("")
    ax.grid(axis="y")
    ax.legend(handles, labels, ncol=1, bbox_to_anchor=[1, 1], loc="upper left")
    fig.tight_layout()

    if fig_name is not None:
        fig.savefig(fig_name, transparent=True)


def plot_prices(file_list: list, config: dict, fig_name=None):
    """plot the prices

    Args:
        results_file (list): the input csvs
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

    fig, ax = plt.subplots()
    bar_width = 0.6
    co2_balance_df.T.plot(
        kind="bar",
        stacked=True,
        width=bar_width,
        color=co2_balance_df.index.map(n.carriers.color),
        ax=ax,
    )
    bar_centers = np.unique([patch.get_x() + bar_width / 2 for patch in ax.patches])
    handles, labels = ax.get_legend_handles_labels()
    ax.plot(
        bar_centers,
        co2_balance_df.sum(axis=0).values,
        color="black",
        marker="D",
        markersize=10,
        lw=3,
    )
    ax.set_ylabel("Mt CO2")
    ax.legend(handles, labels, ncol=1, bbox_to_anchor=[1, 1], loc="upper left")


def plot_co2_shadow_price(file_list: list, config: dict, fig_name=None):
    """plot the co2 price

    Args:
        results_file (list): the input csvs
        config (dict): the snakemake configuration
        fig_name (os.PathLike, optional): the figure name. Defaults to None.
    """
    co2_prices = {}

    for i, results_file in enumerate(file_list):
        df_metrics = pd.read_csv(results_file, index_col=list(range(1)), header=[1])
        co2_prices.update(dict(df_metrics.loc["co2_shadow"]))

    fig, ax = plt.subplots()
    fig.set_size_inches((12, 8))

    ax.plot(co2_prices.keys(), co2_prices.values(), marker="o", color="black")
    ax.set_ylabel("CO2 Shadow price")
    ax.set_xlabel("Year")
    fig.tight_layout()

    if fig_name is not None:
        fig.savefig(fig_name, transparent=True)


if __name__ == "__main__":
    if "snakemake" not in globals():

        snakemake = mock_snakemake(
            "plot_summary",
            opts="ll",
            topology="current+Neighbor",
            pathway="exponential175",
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

    ## ONE OFF HACK
    base = "/home/ivanra/documents/PyPSA-China-PIK/results/version-0325.175.1H/summary/postnetworks/positive"
    paths = [os.path.join(base, p) for p in os.listdir(base)]

    data_paths = {
        "energy": [os.path.join(p, "energy.csv") for p in paths],
        "costs": [os.path.join(p, "costs.csv") for p in paths],
        "co2_price": [os.path.join(p, "metrics.csv") for p in paths],
        "prices": [os.path.join(p, "prices.csv") for p in paths],
        "co2_balance": [os.path.join(p, "co2_balance.csv") for p in paths],
    }

    sdr = float(config["costs"]["discountrate"])
    plot_pathway_costs(
        data_paths["costs"],
        config["plotting"],
        social_discount_rate=sdr,
        fig_name=output_paths.costs,
    )
    plot_energy(data_paths["energy"], config["plotting"], fig_name=output_paths.energy)
    plot_prices(
        data_paths["prices"],
        config,
        fig_name=os.path.dirname(output_paths.costs) + "/prices.png",
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
