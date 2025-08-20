# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

# coding: utf-8
"""Functions to add constraints and prepare the network for the solver.
Associated with the `solve_network_myopic` rule in the Snakefile.
To be merged/consolidated with the `solve_network` script.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
from _helpers import (
    configure_logging,
    mock_snakemake,
    setup_gurobi_tunnel_and_env,
)

logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


def prepare_network(
    n,
    config: dict,
    solve_opts=None,
):
    """Prepare the network for the solver
    Args:
        n (pypsa.Network): the pypsa network object
        solve_opts (dict): the solving options
    """
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    load_shedding = solve_opts.get("load_shedding")
    if load_shedding:
        # intersect between macroeconomic and surveybased willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
        # TODO: retrieve color and nice name from config
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.query("carrier == 'AC'").index
        if not np.isscalar(load_shedding):
            # TODO: do not scale via sign attribute (use Eur/MWh instead of Eur/kWh)
            load_shedding = 1e2  # Eur/kWh

        n.madd(
            "Generator",
            buses_i,
            " load",
            bus=buses_i,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=load_shedding,  # Eur/kWh
            p_nom=1e9,  # kW
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (np.random.random(len(t.df)) - 0.5)

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)) * t.df[
                "length"
            ]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if config["existing_capacities"].get("add", True):
        add_land_use_constraint(n)

    return n


def add_land_use_constraint(n: pypsa.Network, planning_horizons: str | int) -> None:
    """
    Add land use constraints for renewable energy potential. This ensures that the brownfield + greenfield
     vre installations for each generator technology do not exceed the technical potential.

    Args:
        n (pypsa.Network): the network object to add the constraints to
        planning_horizons (str | int): the planning horizon year as string
    """
    # warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'

    for carrier in [
        "solar",
        "solar rooftop",
        "solar-hsat",
        "onwind",
        "offwind",
        "offwind-ac",
        "offwind-dc",
        "offwind-float",
    ]:
        ext_i = (n.generators.carrier == carrier) & ~n.generators.p_nom_extendable
        grouper = n.generators.loc[ext_i].index.str.replace(f" {carrier}.*$", "", regex=True)
        existing = n.generators.loc[ext_i, "p_nom"].groupby(grouper).sum()
        existing.index += f" {carrier}-{planning_horizons}"
        n.generators.loc[existing.index, "p_nom_max"] -= existing

    # check if existing capacities are larger than technical potential
    existing_large = n.generators[n.generators["p_nom_min"] > n.generators["p_nom_max"]].index
    if len(existing_large):
        logger.warning(
            f"Existing capacities larger than technical potential for {existing_large},\
                        adjust technical potential to existing capacities"
        )
        n.generators.loc[existing_large, "p_nom_max"] = n.generators.loc[
            existing_large, "p_nom_min"
        ]

    n.generators["p_nom_max"] = n.generators["p_nom_max"].clip(lower=0)


def add_battery_constraints(n):
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("battery discharger")
    charger_bool = n.links.index.str.contains("battery charger")

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = n.model["Link-p_nom"].loc[chargers_ext] - n.model["Link-p_nom"].loc[dischargers_ext] * eff

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")


def add_chp_constraints(n):
    """Add constraints to couple the heat and electricity output of CHP plants
         (using the cb and cv parameter). See the DEA technology cataloge

    Args:
        n (pypsa.Network): the pypsa network object to which's model the constraints are added
    """
    electric = n.links.index.str.contains("CHP") & n.links.index.str.contains("generator")
    heat = n.links.index.str.contains("CHP") & n.links.index.str.contains("boiler")

    electric_ext = n.links[electric].query("p_nom_extendable").index
    heat_ext = n.links[heat].query("p_nom_extendable").index

    electric_fix = n.links[electric].query("~p_nom_extendable").index
    heat_fix = n.links[heat].query("~p_nom_extendable").index

    p = n.model["Link-p"]  # dimension: [time, link]

    # output ratio between heat and electricity and top_iso_fuel_line for extendable
    if not electric_ext.empty:
        p_nom = n.model["Link-p_nom"]

        lhs = (
            p_nom.loc[electric_ext]
            * (n.links.p_nom_ratio * n.links.efficiency)[electric_ext].values
            - p_nom.loc[heat_ext] * n.links.efficiency[heat_ext].values
        )
        n.model.add_constraints(lhs == 0, name="chplink-fix_p_nom_ratio")

        rename = {"Link-ext": "Link"}
        lhs = p.loc[:, electric_ext] + p.loc[:, heat_ext] - p_nom.rename(rename).loc[electric_ext]
        n.model.add_constraints(lhs <= 0, name="chplink-top_iso_fuel_line_ext")

    # top_iso_fuel_line for fixed
    if not electric_fix.empty:
        lhs = p.loc[:, electric_fix] + p.loc[:, heat_fix]
        rhs = n.links.p_nom[electric_fix]
        n.model.add_constraints(lhs <= rhs, name="chplink-top_iso_fuel_line_fix")

    # back-pressure
    if not n.links[electric].index.empty:
        lhs = (
            p.loc[:, heat] * (n.links.efficiency[heat] * n.links.c_b[electric].values)
            - p.loc[:, electric] * n.links.efficiency[electric]
        )
        n.model.add_constraints(lhs <= 0, name="chplink-backpressure")


def add_transmission_constraints(n):
    """
    Add constraint ensuring that transmission lines p_nom are the same for both directions, i.e.
    p_nom positive = p_nom negative
    """
    if not n.links.p_nom_extendable.any():
        return

    positive_bool = n.links.index.str.contains("positive")
    negative_bool = n.links.index.str.contains("reversed")

    positive_ext = n.links[positive_bool].query("p_nom_extendable").index
    negative_ext = n.links[negative_bool].query("p_nom_extendable").index

    lhs = n.model["Link-p_nom"].loc[positive_ext] - n.model["Link-p_nom"].loc[negative_ext]

    n.model.add_constraints(lhs == 0, name="Link-transimission")


def add_retrofit_constraints(n):
    """
    Add constraints to ensure retrofit capacity is linked to the original capacity
    Args:
        n (pypsa.Network): the pypsa network object to which's model the constraints are added
    """
    p_nom_max = pd.read_csv("resources/data/p_nom/p_nom_max_cc.csv", index_col=0)
    p_nom_max = pd.read_csv("resources/data/p_nom/p_nom_max_cc.csv", index_col=0)
    planning_horizon = snakemake.wildcards.planning_horizons
    for year in range(int(planning_horizon) - 40, 2021, 5):
        coal = (
            n.generators[
                (n.generators.carrier == "coal power plant") & (n.generators.build_year == year)
            ]
            .query("p_nom_extendable")
            .index
        )
        Bus = (
            n.generators[
                (n.generators.carrier == "coal power plant") & (n.generators.build_year == year)
            ]
            .query("p_nom_extendable")
            .bus.values
        )
        coal_retrofit = (
            n.generators[
                n.generators.index.str.contains("retrofit")
                & (n.generators.build_year == year)
                & n.generators.bus.isin(Bus)
            ]
            .query("p_nom_extendable")
            .index
        )
        coal_retrofitted = (
            n.generators[
                n.generators.index.str.contains("retrofit")
                & (n.generators.build_year == year)
                & n.generators.bus.isin(Bus)
            ]
            .query("~p_nom_extendable")
            .groupby("bus")
            .sum()
            .p_nom_opt
        )

        lhs = (
            n.model["Generator-p_nom"].loc[coal]
            + n.model["Generator-p_nom"].loc[coal_retrofit]
            - (
                p_nom_max[str(year)].loc[Bus]
                - coal_retrofitted.reindex(p_nom_max[str(year)].loc[Bus].index, fill_value=0)
            ).values
        )

        n.model.add_constraints(lhs == 0, name="Generator-coal-retrofit-" + str(year))


def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to ``pypsa.linopf.network_lopf``.
    If you want to enforce additional custom constraints, this is a good location to add them.
    The arguments ``opts`` and ``snakemake.config`` are expected to be attached to the network.
    """

    add_chp_constraints(n)
    add_battery_constraints(n)
    add_transmission_constraints(n)
    if snakemake.wildcards.planning_horizons != "2020":
        add_retrofit_constraints(n)


def solve_network(n: pypsa.Network, config: dict, solving, opts="", **kwargs):
    """Perform the optimisation
    Args:
        n (pypsa.Network): the pypsa network object
        config (dict): the configuration dictionary
        solving (dict): the solving configuration dictionary
        opts (str): optional wildcards such as ll (not used in pypsa-china)
    """
    set_of_options = solving["solver"]["options"]
    solver_options = solving["solver_options"][set_of_options] if set_of_options else {}
    solver_name = solving["solver"]["name"]
    cf_solving = solving["options"]
    track_iterations = cf_solving.get("track_iterations", False)
    min_iterations = cf_solving.get("min_iterations", 4)
    max_iterations = cf_solving.get("max_iterations", 6)
    transmission_losses = cf_solving.get("transmission_losses", 0)

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    skip_iterations = cf_solving.get("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    if skip_iterations:
        status, condition = n.optimize(
            solver_name=solver_name,
            transmission_losses=transmission_losses,
            extra_functionality=extra_functionality,
            **solver_options,
            **kwargs,
        )
    else:
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            solver_name=solver_name,
            track_iterations=track_iterations,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            transmission_losses=transmission_losses,
            extra_functionality=extra_functionality,
            **solver_options,
            **kwargs,
        )

    if status != "ok":
        logger.warning(f"Solving status '{status}' with termination condition '{condition}'")
    if "infeasible" in condition:
        raise RuntimeError("Solving status 'infeasible'")

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "solve_network_myopic",
            topology="current+Neighbor",
            co2_pathway="exp175default",
            co2_reduction="0.0",
            planning_horizons="2025",
            heat_demand="positive",
        )

    configure_logging(snakemake)

    # deal with the gurobi license activation, which requires a tunnel to the login nodes
    solver_config = snakemake.config["solving"]["solver"]
    gurobi_license_config = snakemake.config["solving"].get("gurobi_hpc_tunnel", None)
    logger.info(f"Solver config {solver_config} and license cfg {gurobi_license_config}")
    if (solver_config["name"] == "gurobi") & (gurobi_license_config is not None):
        setup_gurobi_tunnel_and_env(gurobi_license_config, logger=logger)

    opts = snakemake.wildcards.get("opts", "")
    if "sector_opts" in snakemake.wildcards.keys():
        opts += "-" + snakemake.wildcards.sector_opts
    opts = [o for o in opts.split("-") if o != ""]
    solve_opts = snakemake.params.solving["options"]

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)

    n = prepare_network(n, solve_opts, snakemake.config)

    n = solve_network(
        n,
        config=snakemake.config,
        solving=snakemake.params.solving,
        opts=opts,
        log_fn=snakemake.log.solver,
    )

    # n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.links_t.p2 = n.links_t.p2.astype(float)
    outp = snakemake.output.network_name
    compression = snakemake.config.get("io", None)
    if compression:
        compression = compression.get("nc_compression", None)
    n.export_to_netcdf(outp, compression=compression)

    logger.info(f"Network successfully solved for {snakemake.wildcards.planning_horizons}")

    logger.info(f"Network successfully solved for {snakemake.wildcards.planning_horizons}")
