import pypsa
import logging
import matplotlib.pyplot as plt

from _plot_utilities import get_stat_colors, set_plot_style
from _helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


def plot_electricity_balance(
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

    p = (
        n.statistics.energy_balance(aggregate_time=False, bus_carrier=bus_carrier)
        .dropna(how="all")
        .groupby("carrier")
        .sum()
        .div(1e3)
        # .drop("-")
        .T
    )
    p.rename(columns={"-": "Load"}, inplace=True)
    p = p.loc[start_date:end_date]

    p["coal"] = p[[c for c in p.columns if c.find("coal") >= 0]].sum(axis=1)
    p["gas"] = p[[c for c in p.columns if c.find("gas") >= 0]].sum(axis=1)
    p.drop(columns=[c for c in p.columns if c.find("coal") >= 0], inplace=True)
    p.drop(columns=[c for c in p.columns if c.find("gas") >= 0], inplace=True)

    color_series = get_stat_colors(
        p, n, plot_config, extra_colors={"Load": plot_config["tech_colors"]["electric load"]}
    )
    color_series.rename(plot_config["nice_names"], inplace=True)

    supply = p.where(p >= 0).dropna(axis=1)
    # TODO make robust
    preferred_order = plot_config["preferred_order"]
    plot_order = [name for name in preferred_order if name in supply.columns] + [
        name for name in supply.columns if name not in preferred_order
    ]
    supply = supply.reindex(columns=plot_order)

    charge = p.where(p < 0).dropna(how="all", axis=1)
    # charge.rename(columns=tech_names_map, inplace=True)
    plot_order = [name for name in preferred_order if name in charge.columns] + [
        name for name in charge.columns if name not in preferred_order
    ]
    charge = charge.reindex(columns=plot_order)

    if not charge.empty:
        charge.plot.area(ax=ax, linewidth=0, color=color_series.loc[charge.columns])
    supply.plot.area(
        ax=ax,
        linewidth=0,
        color=color_series.loc[supply.columns],
    )
    if add_load_line:
        charge["load_pos"] = charge["Load"] * -1
        charge["load_pos"].plot(linewidth=2, color="black", label="Load", ax=ax, linestyle="--")

    ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=16)
    ax.set_ylabel("GW")
    ax.set_ylim(charge.sum(axis=1).min() * 1.14, supply.sum(axis=1).max() * 1.13)
    ax.grid(axis="y")

    return ax


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_time_series",
        )
    configure_logging(snakemake)

    set_plot_style(
        style_config_file=snakemake.config["plotting"]["network_style_config_file"],
        base_styles=["classic", "seaborn-v0_8-white"],
    )

    config = snakemake.config

    n = pypsa.Network(snakemake.input.network)

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_electricity_balance(
        n, config["plotting"], start_date="2060-03-31 21:00", end_date="2060-04-06 12:00:00", ax=ax
    )
    fig.savefig(snakemake.output.spring)

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_electricity_balance(
        n, config["plotting"], start_date="2060-12-10 21:00", end_date="2060-12-17 12:00:00", ax=ax
    )
    fig.savefig(snakemake.output.winter)

    logger.info("Network successfully plotted")
