import logging
import pypsa
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import os

# from make_summary import assign_carriers
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches
from plot_summary import preferred_order, rename_techs
from _plot_utilities import assign_location, set_plot_style
from _helpers import configure_logging, get_supply, mock_snakemake, calc_component_capex
from constants import PLOT_COST_UNITS, PLOT_CAP_UNITS, PLOT_SUPPLY_UNITS


logger = logging.getLogger(__name__)


def make_cost_pies(ntwk: pypsa.Network, cost_df: pd.DataFrame, tech_colors: dict) -> pd.DataFrame:
    """Make cost pies for plotting

    Args:
        ntwk (pypsa.Network): the network
        cost_df (pd.DataFrame): the costs
        tech_colors (dict): the tech color config

    Returns:
        pd.DataFrame: the cost pies per bus location
    """

    costs = cost_df.T.groupby(cost_df.columns).sum().T
    # drop empty columns
    costs.drop(list(costs.columns[(costs == 0.0).all()]), axis=1, inplace=True)
    reordered = preferred_order.intersection(costs.columns).append(
        costs.columns.difference(preferred_order)
    )
    costs = costs[reordered]
    # deal with missing colors
    missing_colors = costs.columns.difference(tech_colors)
    if len(missing_colors) > 0:
        logger.warning(f"Missing colors in plot config for {missing_colors}")
    dict(zip(missing_colors, [tech_colors.get("other", "pink")] * len(missing_colors)))
    costs = costs.stack()  # .sort_index()
    to_drop = costs.index.levels[0].symmetric_difference(ntwk.buses.index)
    costs.drop(to_drop, level=0, inplace=True, axis=0, errors="ignore")
    # make sure they are removed from index
    costs.index = pd.MultiIndex.from_tuples(costs.index.values)
    return costs


def annualised_network_capex(
    ntwk: pypsa.Network,
    components_list: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """sum component costs [WARNING COSTS ONLY ANNUALISED IF THE PERIOD IS ONE YEAR, OTHERWISE THEY ARE PER PERIOD]

    Args:
        ntwk (pypsa.Network): the network
        components_list (list): the component type list

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: the total and pathway costs
    """

    costs_add = pd.DataFrame(index=ntwk.buses.index)
    costs_nom = pd.DataFrame(index=ntwk.buses.index)

    for comp in components_list:
        df_c = getattr(ntwk, comp)

        if df_c.empty:
            continue

        df_c["nice_group"] = df_c.carrier.map(rename_techs)

        cap_name = "e_nom_opt" if comp == "stores" else "p_nom_opt"
        cap_before_ext = "e_nom" if comp == "stores" else "p_nom"

        costs_total = calc_component_capex(df_c, cap_name)
        costs_before_ext = calc_component_capex(df_c, cap_before_ext)
        costs_diff = costs_total - costs_before_ext

        costs_add = pd.concat([costs_add, costs_diff], axis=1)
        costs_nom = pd.concat([costs_nom, costs_total], axis=1)

    return costs_add, costs_nom


def add_cost_pannel(
    df, fig: plt.Figure, preferred_order, tech_colors: dict, ax_loc=[-0.09, 0.28, 0.09, 0.45]
) -> None:
    """Add a cost pannel to the figure

    Args:
        df (_type_): _description_
        fig (plt.Figure): the figure object to which the cost pannel will be added
        preferred_order (_type_): index, the order in whiich to plot
        tech_colors (dict): the tech colors
        ax_loc (list, optional): _description_. Defaults to [-0.09, 0.28, 0.09, 0.45].
    """
    ax3 = fig.add_axes(ax_loc)
    reordered = preferred_order.intersection(df.index).append(df.index.difference(preferred_order))

    df.loc[reordered, df.columns].T.plot(
        kind="bar",
        ax=ax3,
        stacked=True,
        color=[tech_colors[i] for i in reordered],
    )
    percent = round((df.sum()["added"] / df.sum()["total"]) * 100)
    ax3.legend().remove()
    ax3.set_ylabel("annualized system cost bEUR/a")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation="horizontal")
    ax3.grid(axis="y")
    ax3.set_ylim([0, df.sum().max() * 1.1])
    # add label
    ax3.text(0.85, (df.sum()["added"] + 15), str(percent) + "%", color="black")

    fig.tight_layout()


def plot_map(
    network: pypsa.Network,
    tech_colors: dict,
    edge_widths: pd.Series,
    bus_colors: pd.Series,
    bus_sizes: pd.Series,
    edge_colors: pd.Series | str = "black",
    add_ref_edge_sizes=True,
    add_ref_bus_sizes=True,
    add_legend=True,
    bus_unit_conv=PLOT_COST_UNITS,
    edge_unit_conv=PLOT_CAP_UNITS,
    ax=None,
    **kwargs,
) -> plt.Axes:
    """Plot the network on a map

    Args:
        network (pypsa.Network): the pypsa network (filtered to contain only relevant buses & links)
        tech_colors (dict): config mapping
        edge_colors (pd.Series|str): the series of edge colors
        edge_widths (pd.Series): the edge widths
        bus_colors (pd.Series): the series of bus colors
        bus_sizes (pd.Series): the series of bus sizes
        add_ref_edge_sizes (bool, optional): add reference line sizes in legend (requires edge_colors=True). Defaults to True.
        add_ref_bus_sizes (bool, optional): add reference bus sizes in legend. Defaults to True.
        ax (plt.Axes, optional): the plotting ax. Defaults to None (new figure).
    """

    if not ax:
        fig, ax = plt.subplots()

    network.plot(
        bus_sizes=bus_sizes,
        bus_colors=bus_colors,
        line_colors=edge_colors,
        link_colors=edge_colors,
        line_widths=edge_widths,
        link_widths=edge_widths,
        ax=ax,
        color_geomap=True,
        boundaries=kwargs.get("boundaries", None),
    )

    if add_legend:
        carriers = bus_sizes.index.get_level_values(1).unique().tolist()
        colors = [tech_colors[c] for c in carriers]
        if isinstance(edge_colors, str):
            colors += [edge_colors]
            labels = carriers + ["HVDC or HVAC link"]
        else:
            colors += edge_colors.values.to_list()
            labels = carriers + edge_colors.index.to_list()
        leg_opt = {"bbox_to_anchor": (1.42, 1.04), "frameon": False}
        add_legend_patches(ax, colors, labels, legend_kw=leg_opt)

    if add_ref_edge_sizes & (type(edge_colors) == str):
        ref_unit = kwargs.get("ref_edge_unit", "GW")
        size_factor = float(kwargs.get("linewidth_factor", 1e5))
        ref_sizes = kwargs.get("ref_edge_sizes", [1e5, 5e5])
        labels = [f"{float(s)/edge_unit_conv} {ref_unit}" for s in ref_sizes]
        ref_sizes = list(map(lambda x: float(x) / size_factor, ref_sizes))
        legend_kw = dict(
            loc="upper left",
            bbox_to_anchor=(0.25, 1.0),
            frameon=False,
            labelspacing=0.8,
            handletextpad=1,
            title=kwargs.get("edge_ref_title", "Grid cap."),
        )
        add_legend_lines(
            ax, ref_sizes, labels, patch_kw=dict(color=edge_colors), legend_kw=legend_kw
        )

    # add reference bus sizes ferom the units
    if add_ref_bus_sizes:
        ref_unit = kwargs.get("ref_bus_unit", "bEUR/a")
        size_factor = float(kwargs.get("bus_size_factor", 1e10))
        ref_sizes = kwargs.get("ref_bus_sizes", [2e10, 1e10, 5e10])
        labels = [f"{float(s)/bus_unit_conv:.0f} {ref_unit}" for s in ref_sizes]
        ref_sizes = list(map(lambda x: float(x) / size_factor, ref_sizes))

        legend_kw = {
            "loc": "upper left",
            "bbox_to_anchor": (0.0, 1.0),
            "labelspacing": 0.8,
            "frameon": False,
            "handletextpad": 0,
            "title": kwargs.get("bus_ref_title", "System cost (CAPEX)"),
        }

        add_legend_circles(
            ax,
            ref_sizes,
            labels,
            srid=network.srid,
            patch_kw=dict(facecolor="lightgrey"),
            legend_kw=legend_kw,
        )

    return ax


def plot_capex_map(
    network: pypsa.Network,
    planning_horizon: int,
    discount_rate: float,
    opts: dict,
    components=["generators", "links", "stores", "storage_units"],
    base_year=2020,
    cost_pannel=True,
    save_path: os.PathLike = None,
):
    tech_colors = opts["tech_colors"]
    plot_ntwk = network.copy()
    # Drop non-electric buses so they don't clutter the plot
    plot_ntwk.buses.drop(plot_ntwk.buses.index[plot_ntwk.buses.carrier != "AC"], inplace=True)

    assign_location(plot_ntwk)

    plot_ntwk.links.drop(
        plot_ntwk.links.index[plot_ntwk.links.length == 0],
        inplace=True,
    )

    costs_pathway, costs_nom = annualised_network_capex(plot_ntwk, components)
    cost_pie = make_cost_pies(plot_ntwk, costs_pathway, tech_colors)
    cost_pie_nom = make_cost_pies(plot_ntwk, costs_nom, tech_colors)

    # TODO aggregate costs below threshold into "other" -> requires messing with network

    # get all carrier types
    carriers = (
        cost_pie.index.get_level_values(1)
        .unique()
        .union(cost_pie_nom.index.get_level_values(1).unique())
    )
    carriers = carriers.tolist()

    line_lower_threshold = 500.0
    line_upper_threshold = 1e4

    # Make figure with two pannels
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": ccrs.PlateCarree()})
    fig.set_size_inches(opts["cost_map"]["figsize"])

    # Add the total costs
    bus_size_factor = opts["cost_map"]["bus_size_factor"]
    linewidth_factor = opts["cost_map"]["linewidth_factor"]
    edge_widths = (
        pd.concat([plot_ntwk.lines.s_nom_opt, plot_ntwk.links.p_nom_opt])
        .clip(line_lower_threshold, line_upper_threshold)
        .replace(line_lower_threshold, 0)
    )
    plot_map(
        plot_ntwk,
        tech_colors=tech_colors,
        edge_widths=edge_widths / linewidth_factor,
        bus_colors=tech_colors,
        bus_sizes=cost_pie_nom / bus_size_factor,
        edge_colors="indigo",
        ax=ax1,
        add_legend=False,
        **opts["cost_map"],
    )

    # Add the added pathway costs
    edge_widths_added = pd.concat(
        [plot_ntwk.lines.s_nom_opt, plot_ntwk.links.p_nom_opt]
    ) - pd.concat([plot_ntwk.lines.s_nom, plot_ntwk.links.p_nom])
    plot_map(
        plot_ntwk,
        tech_colors=tech_colors,
        edge_widths=edge_widths_added / linewidth_factor,
        bus_colors=tech_colors,
        bus_sizes=cost_pie / bus_size_factor,
        edge_colors="rosybrown",
        ax=ax2,
        add_legend=True,
        **opts["cost_map"],
    )

    # Add the optional cost pannel
    if cost_pannel:
        df = pd.DataFrame(index=carriers, columns=["total", "added"])
        df["total"] = cost_pie_nom.groupby(level=1).sum()
        df["added"] = cost_pie.groupby(level=1).sum()
        df = df.fillna(0)
        df = df / PLOT_COST_UNITS
        df = df / (1 + discount_rate) ** (int(planning_horizon) - base_year)
        add_cost_pannel(df, fig, preferred_order, tech_colors, ax_loc=[-0.09, 0.28, 0.09, 0.45])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, transparent=True, bbox_inches="tight")


def plot_energy_map(
    network: pypsa.Network,
    opts: dict,
    energy_pannel=True,
    save_path=None,
):
    plot_ntwk = network.copy()
    plot_ntwk.buses.drop(plot_ntwk.buses.index[plot_ntwk.buses.carrier != "AC"], inplace=True)

    assign_location(plot_ntwk)

    plot_ntwk.links.drop(
        plot_ntwk.links.index[plot_ntwk.links.length == 0],
        inplace=True,
    )

    energy_supply = get_supply(plot_ntwk, bus_carrier="AC", components_list=["Generator"])
    supply_pies = energy_supply.droplevel(0)

    # TODO aggregate costs below threshold into "other" -> requires messing with network

    # get all carrier types
    carriers = supply_pies.index.get_level_values(1).unique()
    carriers = carriers.tolist()

    # TODO make line handling nicer
    line_lower_threshold = 500.0
    line_upper_threshold = 1e4
    # Make figure
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    fig.set_size_inches(opts["energy_map"]["figsize"])
    # get colors
    bus_colors = plot_ntwk.carriers.loc[plot_ntwk.carriers.nice_name.isin(carriers), "color"]
    bus_colors.rename(opts["nice_names"], inplace=True)
    # Add the total costs
    bus_size_factor = opts["energy_map"]["bus_size_factor"]
    linewidth_factor = opts["energy_map"]["linewidth_factor"]
    edge_widths = (
        pd.concat([plot_ntwk.lines.s_nom_opt, plot_ntwk.links.p_nom_opt])
        .clip(line_lower_threshold, line_upper_threshold)
        .replace(line_lower_threshold, 0)
    )
    preferred_order = pd.Index(opts["preferred_order"])
    reordered = preferred_order.intersection(bus_colors.index).append(
        bus_colors.index.difference(preferred_order)
    )

    plot_map(
        plot_ntwk,
        tech_colors=plot_ntwk.carriers.color,
        edge_widths=edge_widths / linewidth_factor,
        bus_colors=bus_colors.loc[reordered],
        bus_sizes=supply_pies / bus_size_factor,
        edge_colors="indigo",
        ax=ax,
        edge_unit_conv=PLOT_CAP_UNITS,
        bus_unit_conv=PLOT_SUPPLY_UNITS,
        add_legend=False,
        **opts["energy_map"],
    )
    # # Add the optional cost pannel
    if energy_pannel:
        df = supply_pies.groupby(level=1).sum().to_frame()
        df = df.fillna(0)
        add_energy_pannel(df, fig, preferred_order, bus_colors, ax_loc=[-0.09, 0.28, 0.09, 0.45])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, transparent=True, bbox_inches="tight")


def add_energy_pannel(
    df, fig: plt.Figure, preferred_order, colors: pd.Series, ax_loc=[-0.09, 0.28, 0.09, 0.45]
) -> None:
    """Add a cost pannel to the figure

    Args:
        df (_type_): _description_
        fig (plt.Figure): the figure object to which the cost pannel will be added
        preferred_order (_type_): index, the order in whiich to plot
        colors (pd.Series): the colors for the techs, with the correct index and no extra techs
        ax_loc (list, optional): _description_. Defaults to [-0.09, 0.28, 0.09, 0.45].
    """
    ax3 = fig.add_axes(ax_loc)
    reordered = preferred_order.intersection(df.index).append(df.index.difference(preferred_order))
    df = df / PLOT_SUPPLY_UNITS
    # only works if colors has correct index
    df.loc[reordered, df.columns].T.plot(
        kind="bar",
        ax=ax3,
        stacked=True,
        color=colors[reordered],
    )

    ax3.legend().remove()
    ax3.set_ylabel("Electricity supply [TWh]")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation="horizontal")
    ax3.grid(axis="y")
    ax3.set_ylim([0, df.sum().max() * 1.1])

    fig.tight_layout()


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_network",
            opts="ll",
            topology="current+Neighbor",
            pathway="exponential175",
            planning_horizons="2050",
            heating_demand="positive",
        )

    configure_logging(snakemake, logger=logger)

    set_plot_style(
        style_config_file=snakemake.config["plotting"]["network_style_config_file"],
        base_styles=["classic", "seaborn-v0_8-white"],
    )

    config = snakemake.config

    n = pypsa.Network(snakemake.input.network)

    # check the timespan
    timespan = n.snapshots.max() - n.snapshots.min()
    if not 365 <= timespan.days <= 366:
        logger.warning(
            "Network timespan is not one year, this may cause issues with the CAPEX calculation,"
            + " which is referenced to the time period and not directly annualised"
        )
    plot_capex_map(
        n,
        planning_horizon=snakemake.wildcards.planning_horizons,
        discount_rate=config["costs"]["discountrate"],
        opts=config["plotting"],
        components=["generators", "links", "stores", "storage_units"],
        save_path=snakemake.output.cost_map,
        **config["plotting"]["cost_map"],
    )

    logger.info("Network successfully plotted")
