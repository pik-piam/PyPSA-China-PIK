# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

# coding: utf-8
"""Functions to add constraints and prepare the network for the solver.
Associated with the `solve_networks` rule in the Snakefile.
"""
import logging
import numpy as np
import pypsa
import xarray as xr
import pandas as pd
import os
from pandas import DatetimeIndex


from _helpers import configure_logging, mock_snakemake, setup_gurobi_tunnel_and_env, ConfigManager
from _pypsa_helpers import mock_solve
from constants import YEAR_HRS

pypsa.pf.logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def set_transmission_limit(n: pypsa.Network, kind: str, factor: float, n_years=1):
    """
    Set global transimission limit constraints - adapted from pypsa-eur

    Args:
        n (pypsa.Network): the network object
        kind (str): the kind of limit to set, either 'c' for cost or 'v' for volume or l for length
        factor (float or str): the factor to apply to the base year quantity, per year
        n_years (int, optional): the number of years to consider for the limit. Defaults to 1.
    """
    logger.info(
        f"Setting global transmission limit for {kind} with factor {factor}/year and {n_years} years"
    )
    links_dc = n.links.query("carrier in ['AC','DC']").index
    # links_dc_rev = n.links.query("carrier in ['AC','DC'] & Link.str.contains('reverse')").index

    _lines_s_nom = (
        np.sqrt(3)
        * n.lines.type.map(n.line_types.i_nom)
        * n.lines.num_parallel
        * n.lines.bus0.map(n.buses.v_nom)
    )
    lines_s_nom = n.lines.s_nom.where(n.lines.type == "", _lines_s_nom)

    col = "capital_cost" if kind == "c" else "length"
    ref = lines_s_nom @ n.lines[col] + n.links.loc[links_dc, "p_nom"] @ n.links.loc[links_dc, col]

    if factor == "opt" or float(factor) ** n_years > 1.0:
        n.lines["s_nom_min"] = lines_s_nom
        n.lines["s_nom_extendable"] = True

        n.links.loc[links_dc, "p_nom_extendable"] = True

    elif float(factor) ** n_years == 1.0:
        # if factor is 1.0, then we do not need to extend
        n.lines["s_nom_min"] = lines_s_nom
        n.lines["s_nom_extendable"] = False
        n.links.loc[links_dc, "p_nom_extendable"] = False

        # factor = 1 + 1e-7  # to avoid numerical issues with the constraints

    elif float(factor) ** n_years < 1.0:
        n.lines["s_nom_min"] = 0
        n.links.loc[links_dc, "p_nom_min"] = 0
        # n.links.loc[links_dc_rev, "p_nom_min"] = 0

    if factor != "opt":
        con_type = "expansion_cost" if kind == "c" else "volume_expansion"
        rhs = float(factor) ** n_years * ref
        logger.info(
            f"Setting global transmission limit for {kind} to {float(factor) ** n_years} current value"
        )
        n.add(
            "GlobalConstraint",
            f"l{kind}_limit",
            type=f"transmission_{con_type}_limit",
            sense="<=",
            constant=rhs,
            carrier_attribute="AC, DC",
        )


def add_emission_prices(n: pypsa.Network, emission_prices={"co2": 0.0}, exclude_co2=False):
    """from pypsa-eur: add GHG price to marginal costs of generators and storage units

    Args:
        n (pypsa.Network): the pypsa network
        emission_prices (dict, optional): emission prices per GHG. Defaults to {"co2": 0.0}.
        exclude_co2 (bool, optional): do not charge for CO2 emissions. Defaults to False.
    """
    if exclude_co2:
        emission_prices.pop("co2")
    em_price = (
        pd.Series(emission_prices).rename(lambda x: x + "_emissions")
        * n.carriers.filter(like="_emissions")
    ).sum(axis=1)

    n.meta.update({"emission_prices": emission_prices})

    gen_em_price = n.generators.carrier.map(em_price) / n.generators.efficiency

    n.generators["marginal_cost"] += gen_em_price
    n.generators_t["marginal_cost"] += gen_em_price[n.generators_t["marginal_cost"].columns]
    # storage units su
    su_em_price = n.storage_units.carrier.map(em_price) / n.storage_units.efficiency_dispatch
    n.storage_units["marginal_cost"] += su_em_price

    logger.info("Added emission prices to marginal costs of generators and storage units")
    logger.info(f"\tEmission prices: {emission_prices}")


def add_co2_constraints_prices(network: pypsa.Network, co2_control: dict):
    """Add co2 constraints or prices

    Args:
        network (pypsa.Network): the network to which prices or constraints are to be added
        co2_control (dict): the config

    Raises:
        ValueError: unrecognised co2 control option
    """

    if co2_control["control"] is None:
        pass
    elif co2_control["control"] == "price":
        logger.info("Adding CO2 price to marginal costs of generators and storage units")
        add_emission_prices(network, emission_prices={"co2": co2_control["co2_pr_or_limit"]})

    elif co2_control["control"].startswith("budget"):
        co2_limit = co2_control["co2_pr_or_limit"]
        logger.info("Adding CO2 constraint based on scenario {co2_limit}")
        network.add(
            "GlobalConstraint",
            "co2_limit",
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=co2_limit,
        )
    else:
        logger.error(f"Unhandled CO2 control config {co2_control} due to unknown control.")
        raise ValueError(f"Unhandled CO2 config {config['scenario']['co2_reduction']}")

def freeze_components(n: pypsa.Network, config: dict, exclude: list = ["H2 turbine"]):
    """Set p_nom_extendable=False for the components in the network.
    Applies to vre_techs and conventional technologies not in the exclude list.

    Args:
        n (pypsa.Network): the network object
        config (dict): the configuration dictionary
        exclude (list, optional): list of technologies to exclude from freezing. Defaults to ["OCGT"]
    """

    # Freeze VRE and conventional techs
    freeze = config["Techs"]["vre_techs"] + config["Techs"]["conv_techs"]
    freeze = [f for f in freeze if not f in exclude]
    if "coal boiler" in freeze:
        freeze += ["coal boiler central", "coal boiler decentral"]
    if "gas boiler" in freeze:
        freeze += ["gas boiler central", "gas boiler decentral"]

    # very ugly -> how to make more robust?
    to_fix = {
        "OCGT": "gas OCGT",
        "CCGT": "gas CCGT",
        "CCGT-CCS": "gas ccs",
        "coal power plant": "coal",
        "coal-CCS": "coal ccs",
    }
    freeze += [to_fix[k] for k in to_fix if k in freeze]

    for comp in ["generators", "links"]:
        query = "carrier in @freeze & p_nom_extendable == True"
        components = getattr(n, comp)
        # p_nom_max_rcl.isna(): exclude paid_off as needed
        if "p_nom_max_rcl" in components.columns:
            query += " & p_nom_max_rcl.isna()"
        mask = components.query(query).index
        components.loc[mask, "p_nom_extendable"] = False


def prepare_network(
    n: pypsa.Network, solve_opts: dict, config: dict, plan_year: int, co2_pathway: str
) -> pypsa.Network:
    """prepare the network for the solver,
    Args:
        n (pypsa.Network): the network object to optimize
        solve_opts (dict): solving options
        config (dict): the snakemake configuration dictionary
        plan_year (int): planning horizon year for which network is solved
        co2_pathway (str): the CO2 pathway name to use

    Returns:
        pypsa.Network: network object with additional constraints
    """

    co2_opts = ConfigManager(config).fetch_co2_restriction(co2_pathway, int(plan_year))
    add_co2_constraints_prices(n, co2_opts)

    if "clip_p_max_pu" in solve_opts:
        for df in (n.generators_t.p_max_pu, n.storage_units_t.inflow):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    # TODO duplicated with freeze components
    if solve_opts.get("load_shedding"):
        n.add("Carrier", "Load Shedding")
        buses_i = n.buses.query("carrier == 'AC'").index
        n.add(
            "Generator",
            buses_i,
            " load",
            bus=buses_i,
            carrier="Load Shedding",
            marginal_cost=solve_opts.get("voll", 1e5),  # EUR/MWh
            # intersect between macroeconomic and surveybased
            # willingness to pay
            # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
            p_nom=1e6,  # MW
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

    if config["run"].get("is_remind_coupled", False) & (
        config["existing_capacities"].get("freeze_new", False)
    ):
        freeze_components(
            n,
            config,
            exclude=config["existing_capacities"].get("never_freeze", []),
        )
        
    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = YEAR_HRS / nhours

    if config["existing_capacities"].get("add", False):
        add_land_use_constraint(n, plan_year)

    return n


def add_battery_constraints(n: pypsa.Network):
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


def add_chp_constraints(n: pypsa.Network):
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
        "solar thermal",
        "onwind",
        "offwind",
        "offwind-ac",
        "offwind-dc",
        "offwind-float",
    ]:
        ext_i = (n.generators.carrier == carrier) & ~n.generators.p_nom_extendable
        grouper = n.generators.loc[ext_i].index.str.replace(f" {carrier}.*$", "", regex=True)
        existing = n.generators.loc[ext_i, "p_nom"].groupby(grouper).sum()
        existing.index += f" {carrier}"
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


def add_transmission_constraints(n: pypsa.Network):
    """
    Add constraint ensuring that transmission lines p_nom are the same for both directions, i.e.
    p_nom positive = p_nom negative

    Args:
        n (pypsa.Network): the network object to optimize
    """

    if not n.links.p_nom_extendable.any():
        return

    positive_bool = n.links.index.str.contains("positive")
    negative_bool = n.links.index.str.contains("reversed")

    positive_ext = n.links[positive_bool].query("p_nom_extendable").index
    negative_ext = n.links[negative_bool].query("p_nom_extendable").index

    lhs = n.model["Link-p_nom"].loc[positive_ext]
    rhs = n.model["Link-p_nom"].loc[negative_ext]

    n.model.add_constraints(lhs == rhs, name="Link-transmission")


def add_remind_paid_off_constraints(n: pypsa.Network) -> None:
    """
    Paid-off components can be placed wherever PyPSA wants but have a total limit.

    Add constraints to ensure that the paid off capacity from REMIND is not
    exceeded across the network & that it does not exceed the technical potential.

    Args:
        n (pypsa.Network): the network object to which's model the constraints are added
    """

    if not n.config["run"].get("is_remind_coupled", False):
        logger.info("Skipping paid off constraints as REMIND is not coupled")
        return

    # In coupled-mode components (Generators, Links,..) have limits p/e_nom_rcl & a tech_group
    # These columns are added by `add_existing_baseyear.add_paid_off_capacity`.
    # p/e_nom_rcl is the availale paid-off capacity per tech group and is nan for non paid-off (usual) generators.
    # rcl is a legacy name from Aodenweller
    for component in ["Generator", "Link", "Store"]:

        prefix = "e" if component == "Store" else "p"
        paid_off_col = f"{prefix}_nom_max_rcl"

        paid_off = getattr(n, component.lower() + "s").copy()
        # if there are no paid_off components
        if paid_off_col not in paid_off.columns:
            continue
        else:
            paid_off.dropna(subset=[paid_off_col], inplace=True)

        paid_off_totals = paid_off.set_index("tech_group")[paid_off_col].drop_duplicates()

        # LHS: p_nom per technology grp < totals
        groupers = [paid_off["tech_group"]]
        grouper_lhs = xr.DataArray(pd.MultiIndex.from_arrays(groupers), dims=[f"{component}-ext"])
        p_nom_groups = (
            n.model[f"{component}-{prefix}_nom"].loc[paid_off.index].groupby(grouper_lhs).sum()
        )

        # get indices to sort RHS. the grouper is multi-indexed (legacy from PyPSA-Eur)
        idx = p_nom_groups.indexes["group"]
        idx = [x[0] for x in idx]

        # Add constraint
        if not p_nom_groups.empty():
            n.model.add_constraints(
                p_nom_groups <= paid_off_totals[idx].values,
                name=f"paidoff_cap_totals_{component.lower()}",
            )

    # === ensure normal e/p_nom_max is respected for (paid_off + normal) components
    # e.g. if PV has 100MW tech potential at nodeA, paid_off+normal p_nom_opt <100MW
    for component in ["Generator", "Link", "Store"]:
        paidoff_comp = getattr(n, component.lower() + "s").copy()

        prefix = "e" if component == "Store" else "p"
        paid_off_col = f"{prefix}_nom_max_rcl"
        # if there are no paid_off components
        if paid_off_col not in paidoff_comp.columns:
            continue
        else:
            paidoff_comp.dropna(subset=[paid_off_col], inplace=True)

        # techs that only exist as paid-off don't have usual counterparts
        remind_only_techs = n.config["existing_capacities"].get("remind_only_tech_groups", [])
        paidoff_comp = paidoff_comp.query("tech_group not in @remind_only_techs")

        if paidoff_comp.empty:
            continue

        # find equivalent usual (not paid-off) components
        usual_comps_idx = paidoff_comp.index.str.replace("_paid_off", "")
        usual_comps = getattr(n, component.lower() + "s").loc[usual_comps_idx].copy()
        usual_comps = usual_comps[~usual_comps.p_nom_max.isin([np.inf, np.nan])]

        paid_off_wlimits = paidoff_comp.loc[usual_comps.index + "_paid_off"]
        to_constrain = pd.concat([usual_comps, paid_off_wlimits], axis=0)
        to_constrain.rename_axis(index=f"{component}-ext", inplace=True)
        # otherwise n.model query will fail. This is needed in case freeze_compoents was used
        # it is fine so long as p_nom is zero for the frozen components
        to_constrain = to_constrain.query("p_nom_extendable==True")
        to_constrain["grouper"] = to_constrain.index.str.replace("_paid_off", "")

        grouper = xr.DataArray(to_constrain.grouper, dims=[f"{component}-ext"])

        lhs = n.model[f"{component}-{prefix}_nom"].loc[to_constrain.index].groupby(grouper).sum()
        # RHS
        idx = lhs.indexes["grouper"]

        if not lhs.empty():
            n.model.add_constraints(
                lhs <= usual_comps.loc[idx].p_nom_max.values,
                name=f"constrain_paidoff&usual_{component}_potential",
            )


def extra_functionality(n: pypsa.Network, snapshots: DatetimeIndex) -> None:
    """
    Add supplementary constraints to the network model. ``pypsa.linopf.network_lopf``.
    If you want to enforce additional custom constraints, this is a good location to add them.
    The arguments ``opts`` and ``snakemake.config`` are expected to be attached to the network.

    Args:
        n (pypsa.Network): the network object to optimize
        snapshots (DatetimeIndex): the time index of the network
        config (dict): the configuration dictionary
    """
    config = n.config
    add_battery_constraints(n)
    add_transmission_constraints(n)
    add_chp_constraints(n)
    if config["run"].get("is_remind_coupled", False):
        logger.info("Adding remind paid off constraints")
        add_remind_paid_off_constraints(n)


def solve_network(
    n: pypsa.Network, config: dict, solving: dict, opts: str = "", **kwargs
) -> tuple[pypsa.Network, dict]:
    """perform the optimisation
    Args:
        n (pypsa.Network): the pypsa network object
        config (dict): the configuration dictionary
        solving (dict): the solving configuration dictionary
        opts (str): optional wildcards such as ll (not used in pypsa-china)
        
    Returns:
        tuple: (network, dual_data)
    """
    set_of_options = solving["solver"]["options"]
    solver_options = solving["solver_options"][set_of_options] if set_of_options else {}
    solver_name = solving["solver"]["name"]
    cf_solving = solving["options"]
    track_iterations = cf_solving.get("track_iterations", False)
    min_iterations = cf_solving.get("min_iterations", 4)
    max_iterations = cf_solving.get("max_iterations", 6)
    transmission_losses = cf_solving.get("transmission_losses", 0)
    export_duals = solving["solver_options"].get("export_duals", False)

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

    # Collect dual data if enabled
    dual_data = {}
    if export_duals and hasattr(n, "model") and hasattr(n.model, "dual"):
        # Process dual variables and add them to network object
        from _pypsa_helpers import process_dual_variables
        process_dual_variables(n)
        
        # Collect dual data before model might be destroyed
        if n.model.dual:
            dual_data.update(dict(n.model.dual))
        if hasattr(n, 'duals') and n.duals:
            dual_data.update(n.duals)
    elif export_duals:
        logger.warning("Network model does not have dual variables. Dual export will be skipped.")

    return n, dual_data


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "solve_networks",
            co2_pathway="SSP2-PkBudg1000-CHA-pypsaelh2",
            planning_horizons="2025",
            topology="current+FCG",
            # heating_demand="positive",
            configfiles="resources/tmp/remind_coupled_cg.yaml",
        )
    configure_logging(snakemake)

    opts = snakemake.wildcards.get("opts", "")
    if "sector_opts" in snakemake.wildcards.keys():
        opts += "-" + snakemake.wildcards.sector_opts
    opts = [o for o in opts.split("-") if o != ""]
    solve_opts = snakemake.params.solving["options"]
    co2_pathway = snakemake.wildcards.co2_pathway

    # deal with the gurobi license activation, which requires a tunnel to the login nodes
    solver_config = snakemake.config["solving"]["solver"]
    gurobi_tnl_cfg = snakemake.config["solving"].get("gurobi_hpc_tunnel", None)
    logger.info(f"Solver config {solver_config} and license cfg {gurobi_tnl_cfg}")
    if (solver_config["name"] == "gurobi") & (gurobi_tnl_cfg is not None):
        tunnel = setup_gurobi_tunnel_and_env(gurobi_tnl_cfg, logger=logger)
        logger.info(tunnel)
    else:
        tunnel = None

    n = pypsa.Network(snakemake.input.network_name)

    n = prepare_network(
        n, solve_opts, snakemake.config, snakemake.wildcards.planning_horizons, co2_pathway
    )

    line_exp_limits = snakemake.config["lines"].get(
        "expansion", {"transmission_limit": "copt", "base_year": 2020}
    )
    transmission_limit = line_exp_limits.get("transmission_limit", "copt")
    exp_years = int(snakemake.wildcards.planning_horizons) - int(
        line_exp_limits.get("base_year", 2020)
    )
    # TODO split copt, c1.05 into c , opt etc
    set_transmission_limit(
        n, kind=transmission_limit[0], factor=transmission_limit[1:], n_years=exp_years
    )
    # # TODO: remove ugly hack
    # n.storage_units.p_nom_max = n.storage_units.p_nom * 1.05**exp_years

    if tunnel:
        logger.info(f"tunnel process alive? {tunnel.poll()}")

    # HACK to replace pytest monkeypatch
    # which doesn't work as snakemake is a subprocess
    is_test = snakemake.config["run"].get("is_test", False)
    if not is_test:
        n, dual_data = solve_network(
            n,
            config=snakemake.config,
            solving=snakemake.params.solving,
            opts=opts,
            log_fn=snakemake.log.solver,
        )
        
        # Export dual variables in main (not in solve function)
        if dual_data and snakemake.params.solving["solver_options"].get("export_duals", False):
            current_year = snakemake.wildcards.planning_horizons
            results_dir = os.path.dirname(os.path.dirname(snakemake.output.network_name))
            dual_output_dir = os.path.join(results_dir, 'dual', f"dual_values_raw_{current_year}")
            
            from pathlib import Path
            from _pypsa_helpers import export_duals_simple
            export_duals_simple(dual_data, Path(dual_output_dir))
    else:
        logging.info("Mocking the solve step")
        n = mock_solve(n)
        dual_data = {}

    if "p2" in n.links_t:
        n.links_t.p2 = n.links_t.p2.astype(float)

    n.meta.update(dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards))))
    compression = snakemake.config.get("io", None)
    if compression:
        compression = compression.get("nc_compression", None)
    n.export_to_netcdf(snakemake.output.network_name, compression=compression)

    logger.info(f"Network successfully solved for {snakemake.wildcards.planning_horizons}")

    if tunnel:
        logger.info(f"tunnel alive? {tunnel.poll()}")
        tunnel.kill()
        logger.info(f"tunnel alive after kill? {tunnel.poll()}")
