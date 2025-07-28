import seaborn as sns
import pandas as pd
import pypsa
import logging
import matplotlib.pyplot as plt

from _helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


def set_plot_style():
    """apply plotting style to all figures"""
    plt.style.use(
        [
            "classic",
            "seaborn-v0_8-white",
            {
                "axes.grid": False,
                "grid.linestyle": "--",
                "grid.color": "0.6",
                "hatch.color": "white",
                "patch.linewidth": 0.5,
                "font.size": 12,
                "legend.fontsize": "medium",
                "lines.linewidth": 1.5,
                "pdf.fonttype": 42,
            },
        ]
    )


def creat_df(n, tech):
    base = abs(n.stores_t.p.filter(like=tech).sum(axis=1)).max()
    df = (n.stores_t.p.filter(like=tech).sum(axis=1)) / base
    df = df.to_frame()
    df.reset_index(inplace=True)
    renames = {0: "p_store"}
    df.rename(columns=renames, inplace=True)
    date = n.stores_t.p.filter(like="water").index
    date = date.tz_localize("utc")
    # date = date.tz_convert("Asia/Shanghai")
    df["Hour"] = date.hour
    df["Day"] = date.strftime("%m-%d")
    summary = pd.pivot_table(data=df, index="Hour", columns="Day", values="p_store")
    summary = summary.fillna(0)
    return summary, base


def plot_heatmap(n, config):
    techs = ["H2", "battery", "water"]
    freq = config["freq"]
    planning_horizon = snakemake.wildcards.planning_horizons
    for tech in techs:
        fig, ax = plt.subplots(figsize=map_figsize)
        df, base = creat_df(n, tech)
        base = str(int(base / 1e3))
        sns.heatmap(df, ax=ax, cmap="coolwarm", cbar_kws={"label": "pu"}, vmin=-1.0, vmax=1.0)
        ax.set_title(
            tech
            + " heatmap with "
            + freq
            + " resolution in "
            + planning_horizon
            + " P_base = "
            + base
            + " GW"
        )
        fig.savefig(snakemake.output[tech], dpi=150, bbox_inches="tight")


def plot_water_store(n):
    planning_horizon = snakemake.wildcards.planning_horizons
    fig, ax = plt.subplots(figsize=map_figsize)
    (
        n.stores_t.e.filter(like="water").sum(axis=1)
        / n.stores.e_nom_opt.filter(like="water").sum()
    ).plot(ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_title(" water tank storage in " + planning_horizon)
    fig.savefig(snakemake.output["water_store"], dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_heatmap",
            opts="ll",
            topology="current+Neighbor",
            co2_pathway="exp175default",
            planning_horizons="2020",
            heating_demand="positive",
        )
    configure_logging(snakemake, logger=logger)

    set_plot_style()
    config = snakemake.config

    map_figsize = config["plotting"]["map"]["figsize"]
    map_boundaries = config["plotting"]["map"]["boundaries"]

    n = pypsa.Network(snakemake.input.network)

    plot_heatmap(n, config)
    plot_water_store(n)

    logger.info("Heatmap successfully plotted")
