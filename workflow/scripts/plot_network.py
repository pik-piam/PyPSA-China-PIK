import logging
import pypsa
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# from make_summary import assign_carriers
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches
from _plot_utilities import (
    set_plot_style,
    fix_network_names_colors,
    determine_plottable,
    make_nice_tech_colors,
    # aggregate_small_pie_vals,
)
from _pypsa_helpers import get_location_and_carrier
from _helpers import (
    configure_logging,
    mock_snakemake,
    set_plot_test_backend,
)
from constants import PLOT_COST_UNITS, PLOT_CAP_UNITS, PLOT_SUPPLY_UNITS, CURRENCY


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
        add_ref_edge_sizes (bool, optional): add reference line sizes in legend
            (requires edge_colors=True). Defaults to True.
        add_ref_bus_sizes (bool, optional): add reference bus sizes in legend.
            Defaults to True.
        ax (plt.Axes, optional): the plotting ax. Defaults to None (new figure).
    """

    if not ax:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

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

    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray")
    states_provinces = cfeature.NaturalEarthFeature(
        category="cultural", name="admin_1_states_provinces_lines", scale="50m", facecolor="none"
    )
    # Add our states feature.
    ax.add_feature(states_provinces, edgecolor="lightgray", alpha=0.7)

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

    fig.tight_layout()

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
        ax_loc (list, optional): the location of the cost pannel.
            Defaults to [-0.09, 0.28, 0.09, 0.45].
    """
    ax3 = fig.add_axes(ax_loc)
    reordered = preferred_order.intersection(df.index).append(df.index.difference(preferred_order))
    colors = {k.lower(): v for k, v in tech_colors.items()}
    print("DEBUG df in add_cost_pannel:\n", df)
    print("DEBUG reordered:\n", reordered)
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


# TODO fix args unused
def plot_cost_map(
    network: pypsa.Network,
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
        opts (dict): the plotting config (snakemake.config["plotting"])
        base_year (int, optional): the base year (for cost delta). Defaults to 2020.
        capex_only (bool, optional): do not plot VOM (FOM is in CAPEX). Defaults to False.
        plot_additions (bool, optional): plot a map of investments (p_nom_opt vs p_nom).
              Defaults to True.
        cost_pannel (bool, optional): add a bar graph with costs. Defaults to True.
        save_path (os.PathLike, optional): save figure to path (or not if None). Defaults to None.
    raises:
        ValueError: if plot_additions and not capex_only
    """

    if plot_additions and not capex_only:
        raise ValueError("Cannot plot additions without capex only")

    tech_colors = make_nice_tech_colors(opts["tech_colors"], opts["nice_names"])

    # TODO scale edges by cost from capex summary
    def calc_link_plot_width(row, carrier="AC", additions=False):
        if row.length == 0 or row.carrier != carrier or not row.plottable:
            return 0
        elif additions:
            return row.p_nom
        else:
            return row.p_nom_opt

    # ============ === Stats by bus ===
    # calc costs & sum over component types to keep bus & carrier (remove no loc)
    costs = network.statistics.capex(groupby=["location", "carrier"])
    costs = costs.groupby(level=[1, 2]).sum()
    if "" in costs.index:
        costs = costs.drop("")    
    # we miss some buses by grouping epr location, fill w 0s
    bus_idx = pd.MultiIndex.from_product([network.buses.index, ["AC"]])
    costs = costs.reindex(bus_idx.union(costs.index), fill_value=0)
    # add marginal (excluding quasi fixed) to costs if desired
    if not capex_only:
        opex = network.statistics.opex(groupby=["location", "carrier"])
        opex = opex.groupby(level=[1, 2]).sum()
        cost_pies = costs + opex.reindex(costs.index, fill_value=0)

    # === make map components: pies and edges
    cost_pies = costs.fillna(0)
    cost_pies.index.names = ["bus", "carrier"]
    carriers = cost_pies.index.get_level_values(1).unique()
    # map edges
    link_plot_w = network.links.apply(lambda row: calc_link_plot_width(row, carrier="AC"), axis=1)
    edges = pd.concat([network.lines.s_nom_opt, link_plot_w]).groupby(level=0).sum()
    line_lower_threshold = opts.get("min_edge_capacity", 0)
    edge_widths = edges.clip(line_lower_threshold, edges.max()).replace(line_lower_threshold, 0)

    # === Additions ===
    # for pathways sometimes interested in additions from last time step
    if plot_additions:
        installed = (
            network.statistics.installed_capex(groupby=["location", "carrier"])
            .groupby(level=[1, 2])
            .sum()
        )
        costs_additional = costs - installed.reindex(costs.index, fill_value=0)
        cost_pies_additional = costs_additional.fillna(0)
        cost_pies_additional.index.names = ["bus", "carrier"]

        link_additions = network.links.apply(
            lambda row: calc_link_plot_width(row, carrier="AC", additions=True), axis=1
        )
        added_links = link_plot_w - link_additions.reindex(link_plot_w.index, fill_value=0)
        added_lines = network.lines.s_nom_opt - network.lines.s_nom.reindex(
            network.lines.index, fill_value=0
        )
        edge_widths_added = pd.concat([added_links, added_lines]).groupby(level=0).sum()

        # add to carrier types
        carriers = carriers.union(cost_pies_additional.index.get_level_values(1).unique())

    preferred_order = pd.Index(opts["preferred_order"])
    carriers = carriers.tolist()

    # Make figure with right number of pannels
    if plot_additions:
        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": ccrs.PlateCarree()})
        fig.set_size_inches(opts["cost_map"]["figsize_w_additions"])
    else:
        fig, ax1 = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        fig.set_size_inches(opts["cost_map"]["figsize"])

    # Add the total costs
    bus_size_factor = opts["cost_map"]["bus_size_factor"]
    linewidth_factor = opts["cost_map"]["linewidth_factor"]
    plot_map(
        network,
        tech_colors=tech_colors,
        edge_widths=edge_widths / linewidth_factor,
        bus_colors=tech_colors,
        bus_sizes=cost_pies / bus_size_factor,
        edge_colors=opts["cost_map"]["edge_color"],
        ax=ax1,
        add_legend=not plot_additions,
        bus_ref_title=f"System costs{' (CAPEX)'if capex_only else ''}",
        **opts["cost_map"],
    )

    # TODO check edges is working
    # Add the added pathway costs
    if plot_additions:
        plot_map(
            network,
            tech_colors=tech_colors,
            edge_widths=edge_widths_added / linewidth_factor,
            bus_colors=tech_colors,
            bus_sizes=cost_pies_additional / bus_size_factor,
            edge_colors="rosybrown",
            ax=ax2,
            bus_ref_title=f"Added costs{' (CAPEX)' if capex_only else ''}",
            add_legend=True,
            **opts["cost_map"],
        )

    # Add the optional cost pannel
    if cost_pannel:
        df = pd.DataFrame(columns=["total"])
        df["total"] = network.statistics.capex(nice_names=False).groupby(level=1).sum()
        if not capex_only:
            df["opex"] = network.statistics.opex(nice_names=False).groupby(level=1).sum()
            df.rename(columns={"total": "capex"}, inplace=True)
        elif plot_additions:
            df["added"] = (
                df["total"]
                - network.statistics.installed_capex(nice_names=False).groupby(level=1).sum()
            )

        df.fillna(0, inplace=True)
        df = df / PLOT_COST_UNITS
        # TODO decide discount
        # df = df / (1 + discount_rate) ** (int(planning_horizon) - base_year)
        add_cost_pannel(
            df, fig, preferred_order, tech_colors, plot_additions, ax_loc=[-0.09, 0.28, 0.09, 0.45]
        )

    fig.set_size_inches(opts["cost_map"][f"figsize{'_w_additions' if plot_additions else ''}"])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, transparent=opts["transparent"], bbox_inches="tight")


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

    # make the statistics. Buses not assigned to a region will be included
    # if they are linked to a region (e.g. turbine link w carrier = hydroelectricity)
    energy_supply = network.statistics.supply(
        groupby=get_location_and_carrier,
        bus_carrier=carrier,
        comps=components,
    )
    # get rid of components
    supply_pies = energy_supply.groupby(level=[1, 2]).sum()

    # TODO fix  this for heat
    # # calc costs & sum over component types to keep bus & carrier (remove no loc)
    # energy_supply = network.statistics.capex(groupby=["location", "carrier"])
    # energy_supply = energy_supply.groupby(level=[1, 2]).sum().drop("")
    # # we miss some buses by grouping epr location, fill w 0s
    # bus_idx = pd.MultiIndex.from_product([network.buses.index, ["AC"]])
    # supply_pies = energy_supply.reindex(bus_idx.union(energy_supply.index), fill_value=0)

    # remove imports from supply pies
    if carrier == "AC" and not plot_ac_imports:
        supply_pies = supply_pies.loc[supply_pies.index.get_level_values(1) != "AC"]

    # TODO aggregate costs below threshold into "other" -> requires messing with network
    # network.add("Carrier", "Other")

    # get all carrier types
    carriers_list = supply_pies.index.get_level_values(1).unique()
    carriers_list = carriers_list.tolist()

    # TODO make line handling nicer
    line_lower_threshold = opts.get("min_edge_capacity", 500)
    # Make figur
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    fig.set_size_inches(opts["energy_map"]["figsize"])
    # get colors
    bus_colors = network.carriers.loc[network.carriers.nice_name.isin(carriers_list), "color"]
    bus_colors.rename(opts["nice_names"], inplace=True)

    preferred_order = pd.Index(opts["preferred_order"])
    reordered = preferred_order.intersection(bus_colors.index).append(
        bus_colors.index.difference(preferred_order)
    )
    # TODO there'sa  problem with network colors when using heat, pies aren't grouped by location
    colors = network.carriers.color.copy()
    colors.index = colors.index.map(opts["nice_names"])
    tech_colors = make_nice_tech_colors(opts["tech_colors"], opts["nice_names"])

    # make sure plot isnt overpopulated
    def calc_link_plot_width(row, carrier="AC"):
        if row.length == 0 or row.carrier != carrier or not row.plottable:
            return 0
        else:
            return row.p_nom_opt

    edge_carrier = "H2 pipeline" if carrier == "heat" else "AC"
    link_plot_w = network.links.apply(lambda row: calc_link_plot_width(row, edge_carrier), axis=1)
    edges = pd.concat([network.lines.s_nom_opt, link_plot_w])
    edge_widths = edges.clip(line_lower_threshold, edges.max()).replace(line_lower_threshold, 0)

    opts_plot = opts["energy_map"].copy()
    if carrier == "heat":
        opts_plot["ref_bus_sizes"] = opts_plot["ref_bus_sizes_heat"]
        opts_plot["ref_edge_sizes"] = opts_plot["ref_edge_sizes_heat"]
        opts_plot["linewidth_factor"] = opts_plot["linewidth_factor_heat"]
        opts_plot["bus_size_factor"] = opts_plot["bus_size_factor_heat"]
    plot_map(
        network,
        tech_colors=tech_colors,  # colors.to_dict(),
        edge_widths=edge_widths / opts_plot["linewidth_factor"],
        bus_colors=bus_colors.loc[reordered],
        bus_sizes=supply_pies / opts_plot["bus_size_factor"],
        edge_colors=opts_plot["edge_color"],
        ax=ax,
        edge_unit_conv=PLOT_CAP_UNITS,
        bus_unit_conv=PLOT_SUPPLY_UNITS,
        add_legend=True,
        **opts_plot,
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
        fig.savefig(save_path, transparent=opts["transparent"], bbox_inches="tight")


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
    carrier="AC",
    save_path: os.PathLike = None,
):
    """A map plot of energy, either AC or heat

    Args:
        network (pypsa.Network): the pyPSA network object
        opts (dict): the plotting options (snakemake.config["plotting"])
        save_path (os.PathLike, optional): Fig outp path. Defaults to None (no save).
        carrier (str, optional): the energy carrier. Defaults to "AC".
    raises:
        ValueError: if carrier is not AC or heat
    """
    if carrier not in ["AC", "heat"]:
        raise ValueError("Carrier must be either 'AC' or 'heat'")

    # demand weighed prices per node
    nodal_prices = (
        network.statistics.revenue(
            groupby=pypsa.statistics.get_bus_and_carrier_and_bus_carrier,
            comps="Load",
            bus_carrier=carrier,
        )
        / network.statistics.withdrawal(
            comps="Load",
            groupby=pypsa.statistics.get_bus_and_carrier_and_bus_carrier,
            bus_carrier=carrier,
        )
        * -1
    )
    # drop the carrier and bus_carrier, map to colors
    nodal_prices = nodal_prices.droplevel(1).droplevel(1)
    norm = plt.Normalize(vmin=nodal_prices.min(), vmax=nodal_prices.max())
    cmap = plt.get_cmap("plasma")
    bus_colors = nodal_prices.map(lambda x: cmap(norm(x)))

    energy_consum = network.statistics.withdrawal(
        groupby=pypsa.statistics.get_bus_and_carrier,
        bus_carrier=carrier,
        comps=["Load"],
    )
    consum_pies = energy_consum.groupby(level=1).sum()

    # Make figure
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    fig.set_size_inches(opts["price_map"]["figsize"])
    # get colors

    # TODO make line handling nicer
    # make sure plot isnt overpopulated
    def calc_plot_width(row, carrier="AC"):
        if row.length == 0:
            return 0
        elif row.carrier != carrier:
            return 0
        else:
            return row.p_nom_opt

    line_lower_threshold = opts.get("min_edge_capacity", 500)
    edge_carrier = "H2" if carrier == "heat" else "AC"
    link_plot_w = network.links.apply(lambda row: calc_plot_width(row, edge_carrier), axis=1)
    edges = pd.concat([network.lines.s_nom_opt, link_plot_w])
    edge_widths = edges.clip(line_lower_threshold, edges.max()).replace(line_lower_threshold, 0)

    bus_size_factor = opts["price_map"]["bus_size_factor"]
    linewidth_factor = opts["price_map"][f"linewidth_factor{"_heat" if carrier == 'heat' else ''}"]
    plot_map(
        network,
        tech_colors=None,
        edge_widths=edge_widths / linewidth_factor,
        bus_colors=bus_colors,
        bus_sizes=consum_pies / bus_size_factor,
        edge_colors=opts["price_map"]["edge_color"],
        ax=ax,
        edge_unit_conv=PLOT_CAP_UNITS,
        bus_unit_conv=PLOT_SUPPLY_UNITS,
        add_legend=False,
        **opts["price_map"],
    )

    # Add colorbar based on bus_colors
    # fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.87, ax.get_position().y0, 0.02, ax.get_position().height])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label(f"Nodal Prices ${CURRENCY}/MWh")

    if save_path:
        fig.savefig(save_path, transparent=opts["transparent"], bbox_inches="tight")


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_network",
            topology="current+FCG",
            co2_pathway="exp175default",
            planning_horizons="2060",
            heating_demand="positive",
        )
    set_plot_test_backend(snakemake.config)
    configure_logging(snakemake, logger=logger)

    set_plot_style(
        style_config_file=snakemake.config["plotting"]["network_style_config_file"],
        base_styles=["classic", "seaborn-v0_8-white"],
    )

    config = snakemake.config

    n = pypsa.Network(snakemake.input.network)
    # determine whether links correspond to network nodes
    determine_plottable(n)
    # from _helpers import assign_locations
    # assign_locations(n)

    # TODO remove
    # backward compatibility for old network files
    fix_network_names_colors(n, config)

    # check the timespan
    timespan = n.snapshots.max() - n.snapshots.min()
    if not 365 <= timespan.days <= 366:
        logger.warning(
            "Network timespan is not one year, this may cause issues with the CAPEX calculation,"
            + " which is referenced to the time period and not directly annualised"
        )
    additions = True if config["foresight"] != "overnight" else False
    plot_cost_map(
        n,
        opts=config["plotting"],
        save_path=snakemake.output.cost_map,
        capex_only=not additions,
        plot_additions=additions,
    )
    p = snakemake.output.cost_map.replace(".png", "_additions.png")
    plot_cost_map(
        n,
        opts=config["plotting"],
        save_path=p,
        capex_only=not additions,
        plot_additions=additions,
    )
    plot_energy_map(
        n,
        opts=config["plotting"],
        save_path=snakemake.output.el_supply_map,
        carrier="AC",
        energy_pannel=True,
        components=["Generator", "Link"],
    )

    if config["heat_coupling"]:
        p = snakemake.output.cost_map.replace("el_supply.png", "heat_supply.png")
        plot_energy_map(
            n,
            opts=config["plotting"],
            save_path=p,
            carrier="heat",
            energy_pannel=True,
            components=["Generator", "Link"],
        )

    p = snakemake.output.cost_map.replace("cost.png", "nodal_prices.png")
    plot_nodal_prices(
        n,
        carrier="AC",
        opts=config["plotting"],
        save_path=snakemake.output.cost_map,
    )

    logger.info("Network successfully plotted")
