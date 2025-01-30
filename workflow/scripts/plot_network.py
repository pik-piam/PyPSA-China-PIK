import logging
import pypsa
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# from make_summary import assign_carriers
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches
from _plot_utilities import (
    assign_location,
    set_plot_style,
    fix_network_names_colors,
    aggregate_small_pie_vals,
)
from _helpers import configure_logging, get_supply, mock_snakemake, calc_component_capex
from constants import PLOT_COST_UNITS, PLOT_CAP_UNITS, PLOT_SUPPLY_UNITS


logger = logging.getLogger(__name__)


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
        carriers = bus_sizes.index.get_level_values(1).unique()
        colors = carriers.intersection(tech_colors).map(tech_colors).to_list()

        if isinstance(edge_colors, str):
            colors += [edge_colors]
            labels = carriers.to_list() + ["HVDC or HVAC link"]
        else:
            colors += edge_colors.values.to_list()
            labels = carriers.to_list() + edge_colors.index.to_list()
        leg_opt = {"bbox_to_anchor": (1.42, 1.04), "frameon": False}
        add_legend_patches(ax, colors, labels, legend_kw=leg_opt)

    if add_ref_edge_sizes & isinstance(edge_colors, str):
        ref_unit = kwargs.get("ref_edge_unit", "GW")
        size_factor = float(kwargs.get("linewidth_factor", 1e5))
        ref_sizes = kwargs.get("ref_edge_sizes", [1e5, 5e5])

        labels = [f"{float(s)/edge_unit_conv} {ref_unit}" for s in ref_sizes]
        ref_sizes = list(map(lambda x: float(x) / size_factor, ref_sizes))
        legend_kw = dict(
            loc="upper left",
            bbox_to_anchor=(0.26, 1.0),
            frameon=False,
            labelspacing=0.8,
            handletextpad=2,
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
            "title": kwargs.get("bus_ref_title", "UNDEFINED TITLE"),
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


def add_cost_pannel(
    df: pd.DataFrame,
    fig: plt.Figure,
    preferred_order: pd.Index,
    tech_colors: dict,
    plot_additions: bool,
    ax_loc=[-0.09, 0.28, 0.09, 0.45],
) -> None:
    """Add a cost pannel to the figure

    Args:
        df (pd.DataFrame): the cost data to plot
        fig (plt.Figure): the figure object to which the cost pannel will be added
        preferred_order (pd.Index): index, the order in whiich to plot
        tech_colors (dict): the tech colors
        plot_additions (bool): plot the additions
        ax_loc (list, optional): the location of the cost pannel. Defaults to [-0.09, 0.28, 0.09, 0.45].
    """
    ax3 = fig.add_axes(ax_loc)
    reordered = preferred_order.intersection(df.index).append(df.index.difference(preferred_order))
    colors = {k.lower(): v for k, v in tech_colors.items()}
    df.loc[reordered, df.columns].T.plot(
        kind="bar",
        ax=ax3,
        stacked=True,
        color=[colors[k.lower()] for k in reordered],
    )
    ax3.legend().remove()
    ax3.set_ylabel("annualized system cost bEUR/a")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation="horizontal")
    ax3.grid(axis="y")
    ax3.set_ylim([0, df.sum().max() * 1.1])
    if plot_additions:
        # add label
        percent = np.round((df.sum()["added"] / df.sum()["total"]) * 100)
        ax3.text(0.85, (df.sum()["added"] + 15), str(percent) + "%", color="black")

    fig.tight_layout()


def plot_cost_map(
    network: pypsa.Network,
    planning_horizon: int,
    discount_rate: float,
    opts: dict,
    base_year=2020,
    plot_additions=True,
    capex_only=False,
    cost_pannel=True,
    save_path: os.PathLike = None,
):
    """Plot the cost of each node on a map as well as the line capacities

    Args:
        network (pypsa.Network): the network object
        planning_horizon (int): the year reoresebtubg investment period
        discount_rate (float): the social discount rate, applied to the cost pannel only
        opts (dict): the plotting config (snakemake.config["plotting"])
        components (list, optional): the components to plot. Defaults to ["generators", "links", "stores", "storage_units"].
        base_year (int, optional): the base year (for cost delta). Defaults to 2020.
        capex_only (bool, optional): do not plot VOM (FOM is in CAPEX). Defaults to False.
        plot_additions (bool, optional): plot a map of investments (p_nom_opt vs p_nom). Defaults to True.
        cost_pannel (bool, optional): add a bar graph with costs. Defaults to True.
        save_path (os.PathLike, optional): save figure to path (or not if None). Defaults to None.
    raises:
        ValueError: if plot_additions and not capex_only
    """

    if plot_additions and not capex_only:
        raise ValueError("Cannot plot additions without capex only")

    tech_colors = opts["tech_colors"]
    plot_ntwk = network.copy()
    # add regions & flag buses not assigned to a region/provinced as not plottable
    assign_location(plot_ntwk)

    # calc costs & sum over component types to keep bus & carrier
    costs = plot_ntwk.statistics.capex(groupby=["location", "carrier"])
    costs = costs.groupby(level=[1, 2]).sum()
    # add marginal (excluding quasi fixed) to costs if desired
    if not capex_only:
        opex = (
            plot_ntwk.statistics.opex(groupby=["location", "carrier"]).groupby(level=[1, 2]).sum()
        )
        cost_pies = costs + opex.reindex(costs.index, fill_value=0)

    cost_pies = costs.fillna(0)
    cost_pies.index.names = ["bus", "carrier"]
    carriers = cost_pies.index.get_level_values(1).unique()

    # TODO fix or delete
    # cost_pies = aggregate_small_pie_vals(cost_pies, opts["costs_threshold"])
    # plot_ntwk.add("Carrier", "Other")
    # plot_ntwk.add("Generator", [" ".join(x) for x in cost_pies.index.to_flat_index()])
    # reordered = preferred_order.intersection(cost_pies.columns).append(
    #     costs.columns.difference(preferred_order)
    # )
    # cost_pies = cost_pies.loc[(slice(None), reordered)]

    preferred_order = pd.Index(opts["preferred_order"])

    if plot_additions:
        installed = (
            plot_ntwk.statistics.installed_capex(groupby=["location", "carrier"])
            .groupby(level=[1, 2])
            .sum()
        )
        costs_additional = costs - installed.reindex(costs.index, fill_value=0)
        cost_pies_additional = costs_additional.fillna(0)
        cost_pies_additional.index.names = ["bus", "carrier"]

        # get all carrier types
        carriers = carriers.union(cost_pies_additional.index.get_level_values(1).unique())

    carriers = carriers.tolist()

    # TODO more robust to calc edge width as in energy map
    plot_ntwk.links.drop(
        plot_ntwk.links.index[plot_ntwk.links.length == 0],
        inplace=True,
    )
    plot_ntwk.links.drop(
        plot_ntwk.links.index[plot_ntwk.links.carrier != "AC"],
        inplace=True,
    )
    line_lower_threshold = opts.get("min_edge_capacity", 0)

    # Make figure with right number of pannels
    if plot_additions:
        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": ccrs.PlateCarree()})
        fig.set_size_inches(opts["cost_map"]["figsize_w_additions"])
    else:
        fig, ax1 = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        fig.set_size_inches(opts["cost_map"]["figsize"])

    # TODO scale edges by cost from capex summary

    # Add the total costs
    bus_size_factor = opts["cost_map"]["bus_size_factor"]
    linewidth_factor = opts["cost_map"]["linewidth_factor"]
    edges = pd.concat([plot_ntwk.lines.s_nom_opt, plot_ntwk.links.p_nom_opt])
    edge_widths = edges.clip(line_lower_threshold, edges.max()).replace(line_lower_threshold, 0)
    plot_map(
        plot_ntwk,
        tech_colors=tech_colors,
        edge_widths=edge_widths / linewidth_factor,
        bus_colors=tech_colors,
        bus_sizes=cost_pies / bus_size_factor,
        edge_colors="indigo",
        ax=ax1,
        add_legend=not plot_additions,
        bus_ref_title=f"System costs{ ('CAPEX') if capex_only else ''}",
        **opts["cost_map"],
    )
    # TODO check edges is working
    # Add the added pathway costs
    if plot_additions:
        edge_widths_added = pd.concat(
            [plot_ntwk.lines.s_nom_opt, plot_ntwk.links.p_nom_opt]
        ) - pd.concat([plot_ntwk.lines.s_nom, plot_ntwk.links.p_nom])
        plot_map(
            plot_ntwk,
            tech_colors=tech_colors,
            edge_widths=edge_widths_added / linewidth_factor,
            bus_colors=tech_colors,
            bus_sizes=cost_pies_additional / bus_size_factor,
            edge_colors="rosybrown",
            ax=ax2,
            bus_ref_title=f"Added costs{ ('CAPEX') if capex_only else ''}",
            add_legend=True,
            **opts["cost_map"],
        )

    # Add the optional cost pannel
    # TODO fix with opex
    if cost_pannel:
        df = pd.DataFrame(columns=["total"])
        df["total"] = network.statistics.capex(nice_names=False).groupby(level=1).sum()
        if not capex_only:
            df["opex"] = network.statistics.opex(nice_names=False).groupby(level=1).sum()
            df.rename(columns={"total": "capex"})
        elif plot_additions:
            df["added"] = (
                df["total"]
                - network.statistics.installed_capex(nice_names=False).groupby(level=1).sum()
            )

        df.fillna(0, inplace=True)
        df = df / PLOT_COST_UNITS
        # df = df / (1 + discount_rate) ** (int(planning_horizon) - base_year)
        add_cost_pannel(
            df, fig, preferred_order, tech_colors, plot_additions, ax_loc=[-0.09, 0.28, 0.09, 0.45]
        )

    fig.set_size_inches(opts["cost_map"][f"figsize{'_w_additions' if plot_additions else ''}"])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, transparent=False, bbox_inches="tight")


def plot_energy_map(
    network: pypsa.Network,
    opts: dict,
    energy_pannel=True,
    save_path: os.PathLike = None,
    carrier="AC",
    plot_ac_imports=False,
    components=["Generator", "Link"],
):
    """A map plot of energy, either AC or heat

    Args:
        network (pypsa.Network): the pyPSA network object
        opts (dict): the plotting options (snakemake.config["plotting"])
        energy_pannel (bool, optional): add an anergy pie to the left. Defaults to True.
        save_path (os.PathLike, optional): Fig outp path. Defaults to None (no save).
        carrier (str, optional): the energy carrier. Defaults to "AC".
        plot_ac_imports (bool, optional): plot electricity imports. Defaults to False.
        components (list, optional): the components to plot. Defaults to ["Generator", "Link"].
    raises:
        ValueError: if carrier is not AC or heat
    """
    if carrier not in ["AC", "heat"]:
        raise ValueError("Carrier must be either 'AC' or 'heat'")

    # THIS IS INEFFICIENT (3s copy), is there a better way?
    plot_ntwk = network.copy()

    # add regions & flag buses not assigned to a region/provinced as not plottable
    assign_location(plot_ntwk)

    # make the statistics. Buses not assigned to a region will be included
    # if they are linked to a region (e.g. turbine link w carrier = hydroelectricity)
    energy_supply = plot_ntwk.statistics.supply(
        groupby=pypsa.statistics.get_bus_and_carrier,
        bus_carrier=carrier,
        comps=components,
    )
    supply_pies = energy_supply.droplevel(0)

    # remove imports from supply pies
    if carrier == "AC" and not plot_ac_imports:
        supply_pies = supply_pies.loc[supply_pies.index.get_level_values(1) != "AC"]
    if "plottable" in plot_ntwk.links.columns:
        plot_ntwk.links.drop(
            plot_ntwk.links.index[plot_ntwk.links.plottable == False],
            inplace=True,
        )

    # TODO aggregate costs below threshold into "other" -> requires messing with network
    plot_ntwk.add("Carrier", "Other")

    # get all carrier types
    carriers_list = supply_pies.index.get_level_values(1).unique()
    carriers_list = carriers_list.tolist()

    # TODO make line handling nicer
    line_lower_threshold = opts.get("min_edge_capacity", 500)
    # Make figur
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    fig.set_size_inches(opts["energy_map"]["figsize"])
    # get colors
    bus_colors = plot_ntwk.carriers.loc[plot_ntwk.carriers.nice_name.isin(carriers_list), "color"]
    bus_colors.rename(opts["nice_names"], inplace=True)

    # Add the total costs
    bus_size_factor = opts["energy_map"]["bus_size_factor"]
    linewidth_factor = opts["energy_map"]["linewidth_factor"]

    preferred_order = pd.Index(opts["preferred_order"])
    reordered = preferred_order.intersection(bus_colors.index).append(
        bus_colors.index.difference(preferred_order)
    )

    colors = plot_ntwk.carriers.color.copy()
    colors.index = colors.index.map(opts["nice_names"])

    # make sure plot isnt overpopulated
    def calc_plot_width(row, carrier="AC"):
        if row.length == 0:
            return 0
        elif row.carrier != carrier:
            return 0
        else:
            return row.p_nom_opt

    edge_carrier = "H2" if carrier == "heat" else "AC"
    link_plot_w = plot_ntwk.links[["p_nom_opt", "length", "carrier"]].apply(
        lambda row: calc_plot_width(row, edge_carrier), axis=1
    )
    edges = pd.concat([plot_ntwk.lines.s_nom_opt, link_plot_w])
    edge_widths = edges.clip(line_lower_threshold, edges.max()).replace(line_lower_threshold, 0)
    # plot_ntwk.links.drop(
    #     plot_ntwk.links.index[plot_ntwk.links.length == 0],
    #     inplace=True,
    # )
    # )
    # plot_ntwk.links.drop(
    #     plot_ntwk.links.index[plot_ntwk.links.carrier != edge_carrier],
    #     inplace=True,
    # )

    plot_map(
        plot_ntwk,
        tech_colors=colors.to_dict(),
        edge_widths=edge_widths / linewidth_factor,
        bus_colors=bus_colors.loc[reordered],
        bus_sizes=supply_pies / bus_size_factor,
        edge_colors="indigo",
        ax=ax,
        edge_unit_conv=PLOT_CAP_UNITS,
        bus_unit_conv=PLOT_SUPPLY_UNITS,
        add_legend=True,
        **opts["energy_map"],
    )
    # # Add the optional cost pannel
    if energy_pannel:
        df = supply_pies.groupby(level=1).sum().to_frame()
        df = df.fillna(0)
        add_energy_pannel(df, fig, preferred_order, bus_colors, ax_loc=[-0.09, 0.28, 0.09, 0.45])

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=1, bbox_to_anchor=[1, 1], loc="upper left")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, transparent=True, bbox_inches="tight")


def add_energy_pannel(
    df: pd.DataFrame,
    fig: plt.Figure,
    preferred_order: pd.Index,
    colors: pd.Series,
    ax_loc=[-0.09, 0.28, 0.09, 0.45],
) -> None:
    """Add a cost pannel to the figure

    Args:
        df (pd.DataFrame): the statistics supply output by carrier (from plot_energy map)
        fig (plt.Figure): the figure object to which the cost pannel will be added
        preferred_order (pd.Index): index, the order in whiich to plot
        colors (pd.Series): the colors for the techs, with the correct index and no extra techs
        ax_loc (list, optional): the pannel location. Defaults to [-0.09, 0.28, 0.09, 0.45].
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


def plot_nodal_prices(
    network: pypsa.Network,
    opts: dict,
    energy_pannel=True,
    save_path: os.PathLike = None,
    carrier="AC",
    plot_ac_imports=False,
    components=["Generator", "Link"],
):
    """A map plot of energy, either AC or heat

    Args:
        network (pypsa.Network): the pyPSA network object
        opts (dict): the plotting options (snakemake.config["plotting"])
        energy_pannel (bool, optional): add an anergy pie to the left. Defaults to True.
        save_path (os.PathLike, optional): Fig outp path. Defaults to None (no save).
        carrier (str, optional): the energy carrier. Defaults to "AC".
        plot_ac_imports (bool, optional): plot electricity imports. Defaults to False.
        components (list, optional): the components to plot. Defaults to ["Generator", "Link"].
    raises:
        ValueError: if carrier is not AC or heat
    """
    if carrier not in ["AC", "heat"]:
        raise ValueError("Carrier must be either 'AC' or 'heat'")

    # THIS IS INEFFICIENT (3s copy), is there a better way?
    plot_ntwk = network.copy()

    # add regions & flag buses not assigned to a region/provinced as not plottable
    assign_location(plot_ntwk)

    # make the statistics. Buses not assigned to a region will be included
    # if they are linked to a region (e.g. turbine link w carrier = hydroelectricity)
    energy_consum = plot_ntwk.statistics.withdrawal(
        groupby=pypsa.statistics.get_bus_and_carrier,
        bus_carrier=carrier,
        comps=components,
    )
    consum_pies = energy_consum.droplevel(0).groupby(level=2).sum()

    if "plottable" in plot_ntwk.links.columns:
        plot_ntwk.links.drop(
            plot_ntwk.links.index[plot_ntwk.links.plottable == False],
            inplace=True,
        )

    # TODO aggregate costs below threshold into "other" -> requires messing with network
    plot_ntwk.add("Carrier", "Other")

    # get all carrier type

    # TODO make line handling nicer
    line_lower_threshold = opts.get("min_edge_capacity", 500)
    # Make figurfor
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    fig.set_size_inches(opts["energy_map"]["figsize"])
    # get colors
    bus_colors = plot_ntwk.carriers.loc[plot_ntwk.carriers.nice_name.isin(carriers_list), "color"]
    bus_colors.rename(opts["nice_names"], inplace=True)
    # Add the total costs
    bus_size_factor = opts["energy_map"]["bus_size_factor"]
    linewidth_factor = opts["energy_map"]["linewidth_factor"]
    edges = pd.concat([plot_ntwk.lines.s_nom_opt, plot_ntwk.links.p_nom_opt])
    edge_widths = edges.clip(line_lower_threshold, edges.max()).replace(line_lower_threshold, 0)
    preferred_order = pd.Index(opts["preferred_order"])
    reordered = preferred_order.intersection(bus_colors.index).append(
        bus_colors.index.difference(preferred_order)
    )

    colors = plot_ntwk.carriers.color.copy()
    colors.index = colors.index.map(opts["nice_names"])

    plot_map(
        plot_ntwk,
        tech_colors=colors.to_dict(),
        edge_widths=edge_widths / linewidth_factor,
        bus_colors=bus_colors.loc[reordered],
        bus_sizes=consum_pies / bus_size_factor,
        edge_colors="indigo",
        ax=ax,
        edge_unit_conv=PLOT_CAP_UNITS,
        bus_unit_conv=PLOT_SUPPLY_UNITS,
        add_legend=True,
        **opts["energy_map"],
    )
    # # Add the optional cost pannel
    if energy_pannel:
        df = supply_pies.groupby(level=1).sum().to_frame()
        df = df.fillna(0)
        add_energy_pannel(df, fig, preferred_order, bus_colors, ax_loc=[-0.09, 0.28, 0.09, 0.45])

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=1, bbox_to_anchor=[1, 1], loc="upper left")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, transparent=True, bbox_inches="tight")


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_network",
            opts="ll",
            topology="current+FCG",
            pathway="exp175",
            planning_horizons="2045",
            heating_demand="positive",
        )

    configure_logging(snakemake, logger=logger)

    set_plot_style(
        style_config_file=snakemake.config["plotting"]["network_style_config_file"],
        base_styles=["classic", "seaborn-v0_8-white"],
    )

    config = snakemake.config

    n = pypsa.Network(snakemake.input.network)
    # backward compatibility for old network files
    fix_network_names_colors(n, config)

    # check the timespan
    timespan = n.snapshots.max() - n.snapshots.min()
    if not 365 <= timespan.days <= 366:
        logger.warning(
            "Network timespan is not one year, this may cause issues with the CAPEX calculation,"
            + " which is referenced to the time period and not directly annualised"
        )
    plot_cost_map(
        n,
        planning_horizon=snakemake.wildcards.planning_horizons,
        discount_rate=config["costs"]["discountrate"],
        opts=config["plotting"],
        save_path=snakemake.output.cost_map,
        capex_only=False,
        plot_additions=False,
    )
    p = snakemake.output.cost_map.replace(".pdf", "_additions.pdf")
    plot_cost_map(
        n,
        planning_horizon=snakemake.wildcards.planning_horizons,
        discount_rate=config["costs"]["discountrate"],
        opts=config["plotting"],
        save_path=p,
        capex_only=False,
        plot_additions=False,
    )
    plot_energy_map(
        n,
        opts=config["plotting"],
        save_path=snakemake.output.el_supply_map,
        carrier="AC",
        energy_pannel=True,
    )

    logger.info("Network successfully plotted")
