# SPDX-FileCopyrightText: : 2022 The PyPSA-Eur Authors
# SPDX-License-Identifier: MIT
import os
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from constants import SNAKEFILE_CHOICES

# from pypsa.descriptors import Dict
import pypsa
from pypsa.components import components, component_attrs


def override_component_attrs(directory):
    """Tell PyPSA that links can have multiple outputs by
    overriding the component_attrs. This can be done for
    as many buses as you need with format busi for i = 2,3,4,5,....
    See https://pypsa.org/doc/components.html#link-with-multiple-outputs-or-inputs
    Parameters
    ----------
    directory : string
        Folder where component attributes to override are stored
        analogous to ``pypsa/component_attrs``, e.g. `links.csv`.
    Returns
    -------
    Dictionary of overriden component attributes.
    """

    attrs = {k: v.copy() for k, v in component_attrs.items()}

    for component, list_name in components.list_name.items():
        fn = f"{directory}/{list_name}.csv"
        if os.path.isfile(fn):
            overrides = pd.read_csv(fn, index_col=0, na_values="n/a")
            attrs[component] = overrides.combine_first(attrs[component])

    return attrs


def configure_logging(snakemake, skip_handlers=False):
    """
    Configure the basic behaviour for the logging module.
    Note: Must only be called once from the __main__ section of a script.
    The setup includes printing log messages to STDERR and to a log file defined
    by either (in priority order): snakemake.log.python, snakemake.log[0] or "logs/{rulename}.log".
    Additional keywords from logging.basicConfig are accepted via the snakemake configuration
    file under snakemake.config.logging.
    Parameters
    ----------
    snakemake : snakemake object
        Your snakemake object containing a snakemake.config and snakemake.log.
    skip_handlers : True | False (default)
        Do (not) skip the default handlers created for redirecting output to STDERR and file.
    """

    import logging

    if "snakemake" not in globals():
        return

    kwargs = snakemake.config.get("logging", dict())
    kwargs.setdefault("level", "INFO")

    if skip_handlers is False:
        fallback_path = Path(__file__).parent.joinpath("..", "logs", f"{snakemake.rule}.log")
        logfile = snakemake.log.get("python", snakemake.log[0] if snakemake.log else fallback_path)
        kwargs.update(
            {
                "handlers": [
                    # Prefer the 'python' log, otherwise take the first log for each
                    # Snakemake rule
                    logging.FileHandler(logfile),
                    logging.StreamHandler(),
                ]
            }
        )
    logging.basicConfig(**kwargs)


def mock_snakemake(rule_name, snakefile=None, **wildcards):
    """
    This function is expected to be executed from the 'scripts'-directory of '
    the snakemake project. It returns a snakemake.script.Snakemake object,
    based on the Snakefile.
    If a rule has wildcards, you have to specify them in **wildcards.
    Parameters
    ----------
    rule_name: str
        name of the rule for which the snakemake object should be generated
    **wildcards:
        keyword arguments fixing the wildcards. Only necessary if wildcards are
        needed.
    """
    from pathlib import Path
    import os
    import snakemake
    from snakemake.script import Snakemake
    import snakemake.api as sm_api

    # dumb search
    if snakefile is None:
        project = Path(__file__).parent.parent
        snakes = list(project.rglob("Snakefile"))
        snakes += list(project.rglob("snakefile"))
        snakefile = snakes[0]

    # Load the config files
    configfiles = list(snakefile.parent.parent.joinpath("config").rglob("*.yaml"))

    with sm_api.SnakemakeApi(
        sm_api.OutputSettings(
            verbose=False,
            show_failed_logs=True,
        )
    ) as api:
        config_settings = sm_api.ConfigSettings(configfiles=configfiles)
        workflow = api.workflow(
            snakefile=snakefile,
            config_settings=config_settings,
            resource_settings=sm_api.ResourceSettings(),
            storage_settings=sm_api.StorageSettings(),
        )
        workflow.global_resources = {}
        api.setup_logger()
        rule = workflow._workflow.get_rule(rule_name)
        wc = dict(wildcards)

        def make_accessible(*ios):
            for io in ios:
                for i in range(len(io)):
                    for i in range(len(io)):
                        io[i] = os.path.abspath(io[i])

        make_accessible(rule.input, rule.output, rule.log)

        snakemake = Snakemake(
            rule.input,
            rule.output,
            rule.params,
            wildcards,
            None,
            rule.resources,
            rule.log,
            workflow.config_settings,
            rule.name,
            None,
        )
        # create log and output dir if not existent
        for path in list(snakemake.log) + list(snakemake.output):
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        return snakemake


def load_network_for_plots(fn, tech_costs, config, cost_year, combine_hydro_ps=True):
    import pypsa
    from add_electricity import update_transmission_costs, load_costs

    n = pypsa.Network(fn)

    n.loads["carrier"] = n.loads.bus.map(n.buses.carrier) + " load"
    n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)

    n.links["carrier"] = n.links.bus0.map(n.buses.carrier) + "-" + n.links.bus1.map(n.buses.carrier)
    n.lines["carrier"] = "AC line"
    n.transformers["carrier"] = "AC transformer"

    # n.lines['s_nom'] = n.lines['s_nom_min']
    # n.links['p_nom'] = n.links['p_nom_min']

    if combine_hydro_ps:
        n.storage_units.loc[n.storage_units.carrier.isin({"PHS", "hydro"}), "carrier"] = "hydro+PHS"

    # if the carrier was not set on the heat storage units
    # bus_carrier = n.storage_units.bus.map(n.buses.carrier)
    # n.storage_units.loc[bus_carrier == "heat","carrier"] = "water tanks"

    Nyears = n.snapshot_weightings.objective.sum() / 8760.0
    costs = load_costs(tech_costs, config["costs"], config["electricity"], cost_year, Nyears)
    update_transmission_costs(n, costs)

    return n


def aggregate_p(n):
    return pd.concat(
        [
            n.generators_t.p.sum().groupby(n.generators.carrier).sum(),
            n.storage_units_t.p.sum().groupby(n.storage_units.carrier).sum(),
            n.stores_t.p.sum().groupby(n.stores.carrier).sum(),
            -n.loads_t.p.sum().groupby(n.loads.carrier).sum(),
        ]
    )


def aggregate_costs(n, flatten=False, opts=None, existing_only=False):

    components = dict(
        Link=("p_nom", "p0"),
        Generator=("p_nom", "p"),
        StorageUnit=("p_nom", "p"),
        Store=("e_nom", "p"),
        Line=("s_nom", None),
        Transformer=("s_nom", None),
    )

    costs = {}
    for c, (p_nom, p_attr) in zip(
        n.iterate_components(components.keys(), skip_empty=False), components.values()
    ):
        if c.df.empty:
            continue
        if not existing_only:
            p_nom += "_opt"
        costs[(c.list_name, "capital")] = (
            (c.df[p_nom] * c.df.capital_cost).groupby(c.df.carrier).sum()
        )
        if p_attr is not None:
            p = c.pnl[p_attr].sum()
            if c.name == "StorageUnit":
                p = p.loc[p > 0]
            costs[(c.list_name, "marginal")] = (p * c.df.marginal_cost).groupby(c.df.carrier).sum()
    costs = pd.concat(costs)

    if flatten:
        assert opts is not None
        conv_techs = opts["conv_techs"]

        costs = costs.reset_index(level=0, drop=True)
        costs = costs["capital"].add(
            costs["marginal"].rename({t: t + " marginal" for t in conv_techs}), fill_value=0.0
        )

    return costs


def update_p_nom_max(n):
    # if extendable carriers (solar/onwind/...) have capacity >= 0,
    # e.g. existing assets from the OPSD project are included to the network,
    # the installed capacity might exceed the expansion limit.
    # Hence, we update the assumptions.

    n.generators.p_nom_max = n.generators[["p_nom_min", "p_nom_max"]].max(1)


def define_spatial(nodes, options):
    """
    Namespace for spatial
    Parameters
    ----------
    nodes : list-like
    """

    spatial = SimpleNamespace()

    spatial.nodes = nodes

    # biomass

    spatial.biomass = SimpleNamespace()

    if options["biomass_transport"]:
        spatial.biomass.nodes = nodes + " solid biomass"
        spatial.biomass.locations = nodes
        spatial.biomass.industry = nodes + " solid biomass for industry"
        spatial.biomass.industry_cc = nodes + " solid biomass for industry CC"
    else:
        spatial.biomass.nodes = ["China solid biomass"]
        spatial.biomass.locations = ["China"]
        spatial.biomass.industry = ["solid biomass for industry"]
        spatial.biomass.industry_cc = ["solid biomass for industry CC"]

    spatial.biomass.df = pd.DataFrame(vars(spatial.biomass), index=nodes)

    # co2

    spatial.co2 = SimpleNamespace()

    if options["co2_network"]:
        spatial.co2.nodes = nodes + " co2 stored"
        spatial.co2.locations = nodes
        spatial.co2.vents = nodes + " co2 vent"
    else:
        spatial.co2.nodes = ["co2 stored"]
        spatial.co2.locations = ["China"]
        spatial.co2.vents = ["co2 vent"]

    spatial.co2.df = pd.DataFrame(vars(spatial.co2), index=nodes)

    # gas

    spatial.gas = SimpleNamespace()

    if options["gas_network"]:
        spatial.gas.nodes = nodes + " gas"
        spatial.gas.locations = nodes
        spatial.gas.biogas = nodes + " biogas"
        spatial.gas.industry = nodes + " gas for industry"
        spatial.gas.industry_cc = nodes + " gas for industry CC"
        spatial.gas.biogas_to_gas = nodes + " biogas to gas"
    else:
        spatial.gas.nodes = ["China gas"]
        spatial.gas.locations = ["China"]
        spatial.gas.biogas = ["China biogas"]
        spatial.gas.industry = ["gas for industry"]
        spatial.gas.industry_cc = ["gas for industry CC"]
        spatial.gas.biogas_to_gas = ["China biogas to gas"]

    spatial.gas.df = pd.DataFrame(vars(spatial.gas), index=nodes)

    # oil
    spatial.oil = SimpleNamespace()
    spatial.oil.nodes = ["China oil"]
    spatial.oil.locations = ["China"]

    # uranium
    spatial.uranium = SimpleNamespace()
    spatial.uranium.nodes = ["China uranium"]
    spatial.uranium.locations = ["China"]

    # coal
    spatial.coal = SimpleNamespace()
    spatial.coal.nodes = ["China coal"]
    spatial.coal.locations = ["China"]

    # lignite
    spatial.lignite = SimpleNamespace()
    spatial.lignite.nodes = ["China lignite"]
    spatial.lignite.locations = ["China"]

    return spatial


if __name__ == "__main__":

    mock_snakemake(
        "build_population",
    )
    print("DONE")
