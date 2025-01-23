# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

# coding: utf-8

import logging

from pypsa.descriptors import get_switchable_as_dense as get_as_dense
import numpy as np
import pypsa

from _helpers import (
    configure_logging,
    override_component_attrs,
    mock_snakemake,
    setup_gurobi_tunnel_and_env,
)

pypsa.pf.logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def prepare_network(n, solve_opts):

    if "clip_p_max_pu" in solve_opts:
        for df in (n.generators_t.p_max_pu, n.storage_units_t.inflow):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if solve_opts.get("load_shedding"):
        n.add("Carrier", "Load")
        buses_i = n.buses.query("carrier == 'AC'").index
        n.madd(
            "Generator",
            buses_i,
            " load",
            bus=buses_i,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=1e2,  # Eur/kWh
            # intersect between macroeconomic and surveybased
            # willingness to pay
            # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
            p_nom=1e9,  # kW
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components(n.one_port_components):
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

    return n


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
        n.model.add_constraints(lhs <= rhs, name="chplink-backpressure")


def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to ``pypsa.linopf.network_lopf``.
    If you want to enforce additional custom constraints, this is a good location to add them.
    The arguments ``opts`` and ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    add_battery_constraints(n)
    add_chp_constraints(n)


def solve_network(n, config, solving, opts="", **kwargs):
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
            "solve_networks",
            co2_reduction="0.0",
            # opts="ll",
            planning_horizons=2020,
            pathway="exponential-175",
            topology="current+Neighbour",
            heating_demand="positive",
        )
    configure_logging(snakemake)

    # deal with the gurobi license activation, which requires a tunnel to the login nodes
    solver_config = snakemake.config["solving"]["solver"]
    gurobi_license_config = snakemake.config["solving"].get("gurobi_hpc_tunnel", None)
    logger.info(f"Solver config {solver_config} and license cfg {gurobi_license_config}")
    if (solver_config["name"] == "gurobi") & (gurobi_license_config is not None):
        tunnel = setup_gurobi_tunnel_and_env(gurobi_license_config, logger=logger)

    opts = snakemake.wildcards.get("opts", "")
    if "sector_opts" in snakemake.wildcards.keys():
        opts += "-" + snakemake.wildcards.sector_opts
    opts = [o for o in opts.split("-") if o != ""]
    solve_opts = snakemake.params.solving["options"]

    if "overrides" in snakemake.input.keys():
        overrides = override_component_attrs(snakemake.input.overrides)
        n = pypsa.Network(snakemake.input.network_name, override_component_attrs=overrides)
    else:
        n = pypsa.Network(snakemake.input.network_name)

    n = prepare_network(n, solve_opts)
    if (solver_config["name"] == "gurobi") & (gurobi_license_config is not None):
        logger.info(f"tunnel process alive? {tunnel.poll()}")
    n = solve_network(
        n,
        config=snakemake.config,
        solving=snakemake.params.solving,
        opts=opts,
        log_fn=snakemake.log.solver,
    )

    # n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.links_t.p2 = n.links_t.p2.astype(float)
    n.export_to_netcdf(snakemake.output[0])

    logger.info(f"Network successfully solved for {snakemake.wildcards.planning_horizons}")
    if (solver_config["name"] == "gurobi") & (gurobi_license_config is not None):
        logger.info(f"tunnel alive? {tunnel.poll()}")

        tunnel.kill()
        logger.info(f"tunnel alive after kill? {tunnel.poll()}")
