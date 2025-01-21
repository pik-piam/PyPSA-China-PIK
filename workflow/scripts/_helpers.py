# SPDX-FileCopyrightText: : 2022 The PyPSA-Eur Authors
# SPDX-License-Identifier: MIT

# WARNING: DO NOT DO "import snakemake"

import os
import sys
import subprocess
import pandas as pd
import logging
import importlib
import time
import pytz
from pathlib import Path
from types import SimpleNamespace
from numpy.random import default_rng

import pypsa
from pypsa.components import components, component_attrs
from constants import TIMEZONE

# get root logger
logger = logging.getLogger()

DEFAULT_TUNNEL_PORT = 1080


class PathManager:
    """A class to manage paths for the snakemake workflow"""

    def __init__(self, snmk_config):
        self.config = snmk_config

    def _get_version(self) -> str:
        """Hacky solution to get version from workflow pseudo-package"""
        spec = importlib.util.spec_from_file_location(
            "workflow", os.path.abspath("./workflow/__init__.py")
        )
        workflow = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workflow)
        return workflow.__version__

    def _join_scenario_vars(self) -> str:
        # TODO make into a config
        exclude = ["planning_horizons", "co2_reduction"]
        short_names = {
            "planning_horizons": "ph",
            "co2_reduction": "co2",
            "opts": "opts",
            "topology": "topo",
            "pathway": "pthw",
            "heating_demand": "proj",
        }
        # remember need place holders for snakemake
        return "_".join(
            [
                f"{short_names[k] if k in short_names else k}-{{{k}}}"
                for k in self.config["scenario"]
                if k not in exclude
            ]
        )

    def results_dir(self, extra_opts: dict = None):
        run, foresight = self.config["run"]["name"], self.config["foresight"]
        base_dir = "v-" + self._get_version() + "_" + run
        sub_dir = foresight + "_" + self._join_scenario_vars()
        if extra_opts:
            sub_dir += "_" + "".join(extra_opts.values())
        return os.path.join(self.config["results_dir"], base_dir, sub_dir)

    def derived_data_dir(self, shared=False):
        foresight = self.config["foresight"]
        if not shared:
            sub_dir = foresight + "_" + self._join_scenario_vars()
            return os.path.join("resources/derived_data", sub_dir)
        else:
            return "resources/derived_data"

    def logs_dir(self):
        run, foresight = self.config["run"]["name"], self.config["foresight"]
        base_dir = "v-" + self._get_version() + "_" + run
        sub_dir = foresight + "_" + self._join_scenario_vars()
        return os.path.join("logs", base_dir, sub_dir)

    def copy_log(self):
        pass


# ============== HPC helpers ==================


def setup_gurobi_tunnel_and_env(tunnel_config: dict, logger: logging.Logger = None):
    """A utility function to set up the Gurobi environment variables and establish an
    SSH tunnel on HPCs. Otherwise the license check will fail if the compute nodes do
     not have internet access or a token server isn't set up

    Args:
        config (dict): the snakemake pypsa-china configuration
        logger (logging.Logger, optional): Logger. Defaults to None.
    """
    if tunnel_config.get("use_tunnel", False) is False:
        return
    logger.info("setting up tunnel")
    user = os.getenv("USER")  # User is pulled from the environment
    port = tunnel_config.get("port", DEFAULT_TUNNEL_PORT)
    ssh_command = f"ssh -fN -D {port} {user}@login01"

    # random sleep to avoid port conflicts with simultaneous connections
    time.sleep(default_rng().uniform(0, 1))

    try:
        # Run SSH in the background to establish the tunnel
        subprocess.Popen(ssh_command, shell=True)
        logger.info(f"SSH tunnel established on port {port}")
    # TODO don't handle unless neeeded
    except Exception as e:
        logger.error(f"Error starting SSH tunnel: {e}")
        sys.exit(1)

    os.environ["https_proxy"] = f"socks5://127.0.0.1:{port}"
    os.environ["SSL_CERT_FILE"] = "/p/projects/rd3mod/ssl/ca-bundle.pem_2022-02-08"
    os.environ["GRB_CAFILE"] = "/p/projects/rd3mod/ssl/ca-bundle.pem_2022-02-08"

    # Set up Gurobi environment variables
    os.environ["GUROBI_HOME"] = "/p/projects/rd3mod/gurobi1103/linux64"
    os.environ["PATH"] += f":{os.environ['GUROBI_HOME']}/bin"
    os.environ["LD_LIBRARY_PATH"] += f":{os.environ['GUROBI_HOME']}/lib"
    os.environ["GRB_LICENSE_FILE"] = "/p/projects/rd3mod/gurobi_rc/gurobi.lic"
    os.environ["GRB_CURLVERBOSE"] = "1"
    os.environ["GRB_SERVER_TIMEOUT"] = "10"
    # os.environ["https_timeout"] = "10"
    # os.environ["proxy_timeout"] = "10"
    logger.info("Gurobi Environment variables & tunnel set up successfully.")


# =========== PyPSA Helpers =============


def override_component_attrs(directory: os.PathLike) -> dict:
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


def aggregate_p(n: pypsa.Network) -> pd.Series:
    """Make a single series for generators, storage units, loads, and stores power,
    summed over all carriers

    Args:
        n (pypsa.Network): the network object

    Returns:
        pd.Series: the aggregated p data
    """
    return pd.concat(
        [
            n.generators_t.p.sum().groupby(n.generators.carrier).sum(),
            n.storage_units_t.p.sum().groupby(n.storage_units.carrier).sum(),
            n.stores_t.p.sum().groupby(n.stores.carrier).sum(),
            -n.loads_t.p.sum().groupby(n.loads.carrier).sum(),
        ]
    )


# TODO make a standard apply/str op instead ofmap in add_electricity.sanitize_carriers
def rename_techs(label: str, nice_names: dict | pd.Series = None) -> str:
    """Rename technology labels for better readability. Removes some prefixes
        and renames if certain conditions  defined in function body are met.

    Args:
        label (str): original technology label
        nice_names (dict, optional): nice names that will overwrite defaults

    Returns:
        str: renamed tech label
    """

    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral ",
    ]

    rename_if_contains = [
        "CHP",
        "gas boiler",
        "biogas",
        "solar thermal",
        "air heat pump",
        "ground heat pump",
        "resistive heater",
        "Fischer-Tropsch",
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        # "H2 Electrolysis": "hydrogen storage",
        # "H2 Fuel Cell": "hydrogen storage",
        # "H2 pipeline": "hydrogen storage",
        "battery": "battery storage",
        "H2 for industry": "H2 for industry",
        "land transport fuel cell": "land transport fuel cell",
        "land transport oil": "land transport oil",
        "oil shipping": "shipping oil",
        # "CC": "CC"
    }

    for ptr in prefix_to_remove:
        if label[: len(ptr)] == ptr:
            label = label[len(ptr) :]

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old, new in rename_if_contains_dict.items():
        if old in label:
            label = new
    # import here to not mess with snakemake
    from constants import NICE_NAMES_DEFAULT

    names_new = NICE_NAMES_DEFAULT.copy()
    names_new.update(nice_names)
    for old, new in names_new.items():
        if old == label:
            label = new
    return label


def aggregate_costs(
    n: pypsa.Network,
    flatten=False,
    opts: dict = None,
    existing_only=False,
) -> pd.Series | pd.DataFrame:

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
        n.iterate_components(components.keys(), skip_empty=True), components.values()
    ):
        if not existing_only:
            p_nom += "_opt"
        costs[(c.list_name, "capital")] = (
            (c.df[p_nom] * c.df.capital_cost).groupby(c.df.carrier).sum()
        )
        if p_attr is not None:
            p = c.dynamic[p_attr].sum()
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


def calc_component_capex(comp_df: pd.DataFrame, capacity_name="p_nom_opt") -> pd.DataFrame:
    """Annualise the capex costs of the components

    Args:
        comp_df (pd.DataFrame): the component dataframe (from n.itercomponents() or getattr)
        capacity_name (str, optional): the capacity (power, energy,..). Defaults to "p_nom_opt".

    Returns:
        pd.Dataframe: annualised capex
    """

    costs_a = (
        (comp_df.capital_cost * comp_df[capacity_name])
        .groupby([comp_df.location, comp_df.nice_group])
        .sum()
        .unstack()
        .fillna(0.0)
    )

    return costs_a


def calc_atlite_heating_timeshift(date_range: pd.date_range, use_last_ts=False) -> int:
    """Imperfect function to calculate the heating time shift for atlite
    Atlite is in xarray, which does not have timezone handling. Adapting the UTC ERA5 data
    to the network local time, is therefore limited to a single shift, which is based on the first
    entry of the time range. For a whole year, in the northern Hemisphere -> winter

    Args:
        date_range (pd.date_range): the date range for which the shift is calc
        use_last_ts (bool, optional): use last instead of first. Defaults to False.

    Returns:
        int: a single timezone shift to utc in hours
    """
    idx = 0 if not use_last_ts else -1
    return pytz.timezone(TIMEZONE).utcoffset(date_range[idx]).total_seconds() / 3600


def calc_utc_timeshift(snapshot_config: dict, weather_year: int) -> pd.TimedeltaIndex:
    """calculate the timeshift to UTC based on the TIMEZONE constant. This is needed
    to bring the atlite UTC times in line with the network ones.

    A complication is that the planning and weather years are not identical

    Args:
        snapshot_config (dict): the snapshots config from snakemake

    Returns:
        pd.TimedeltaIndex: the shifts to UTC
    """
    weather_snapshots = make_periodic_snapshots(
        year=weather_year,
        freq=snapshot_config["freq"],
        start_day_hour=snapshot_config["start"],
        end_day_hour=snapshot_config["end"],
        bounds=snapshot_config["bounds"],
        tz=TIMEZONE,
        end_year=(None if not snapshot_config["end_year_plus1"] else weather_year + 1),
    )
    # for some reason convert to utc messes up the time delta
    utc_snapshots = make_periodic_snapshots(
        year=weather_year,
        freq=snapshot_config["freq"],
        start_day_hour=snapshot_config["start"],
        end_day_hour=snapshot_config["end"],
        bounds=snapshot_config["bounds"],
        tz="UTC",
        end_year=(None if not snapshot_config["end_year_plus1"] else weather_year + 1),
    )

    # time delta will be added to the utc snapshots from atlite
    return utc_snapshots - weather_snapshots


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


def get_supply(
    n: pypsa.Network, bus_carrier: str = None, components_list=["Generator"]
) -> pd.DataFrame:
    """Get the period supply for all or a certain carrier from the network buses

    Args:
        n (pypsa.Network): the network object
        bus_carrier (str, optional): the carrier for the buses. Defaults to None.
        components_list (list, optional): the list of components. Defaults to ['Generator'].

    Returns:
        pd.DataFrame: the supply for each bus and component type
    """
    return n.statistics.supply(
        groupby=pypsa.statistics.get_bus_and_carrier, bus_carrier=bus_carrier, comps=components_list
    )


def is_leap_year(year: int) -> bool:
    """Determine whether a year is a leap year.
    Args:
        year (int): the year"""
    year = int(year)
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def load_network_for_plots(
    network_file: os.PathLike,
    tech_costs: os.PathLike,
    config: dict,
    cost_year: int,
    combine_hydro_ps=True,
) -> pypsa.Network:
    """load network object

    Args:
        network_file (os.PathLike): the path to the network file
        tech_costs (os.PathLike): the path to the costs file
        config (dict): the snamekake config
        cost_year (int): the year for the costs
        combine_hydro_ps (bool, optional): combine the hydro & PHS carriers. Defaults to True.

    Returns:
        pypsa.Network: the network object
    """

    from add_electricity import update_transmission_costs, load_costs

    n = pypsa.Network(network_file)

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


def make_periodic_snapshots(
    year: int,
    freq: int,
    start_day_hour="01-01 00:00:00",
    end_day_hour="12-31 23:00",
    bounds="both",
    end_year: int = None,
    tz: str = None,
) -> pd.date_range:
    """Centralised function to make regular snapshots.
    REMOVES LEAP DAYS

    Args:
        year (int): start time stamp year (end year if end_year None)
        freq (int): snapshot frequency in hours
        start_day_hour (str, optional): Day and hour. Defaults to "01-01 00:00:00".
        end_day_hour (str, optional): _description_. Defaults to "12-31 23:00".
        bounds (str, optional):  bounds behaviour (pd.data_range) . Defaults to "both".
        tz (str, optional): timezone (UTC, None or a timezone). Defaults to None (naive).
        end_year (int, optional): end time stamp year. Defaults to None (use year).

    Returns:
        pd.date_range: the snapshots for the network
    """
    if not end_year:
        end_year = year
    snapshots = pd.date_range(
        f"{int(year)}-{start_day_hour}",
        f"{int(end_year)}-{end_day_hour}",
        freq=freq,
        inclusive=bounds,
        tz=tz,
    )

    if is_leap_year(int(year)):
        snapshots = snapshots[~((snapshots.month == 2) & (snapshots.day == 29))]
    return snapshots


def shift_profile_to_planning_year(data: pd.DataFrame, planning_yr: int | str) -> pd.DataFrame:
    """Shift the profile to the planning year - this harmonises weather and network timestamps
       which is needed for pandas loc operations
    Args:
        data (pd.DataFrame): profile data, for 1 year
        planning_yr (int): planning year
    Returns:
        pd.DataFrame: shifted profile data
    Raises:
        ValueError: if the profile data crosses years
    """
    years = data.index.year.unique()
    if not len(years) == 1:
        raise ValueError(f"Data should be for one year only but got {years}")

    ref_year = years[0]
    # remove all planning year leap days
    if is_leap_year(ref_year):  # and not is_leap_year(planning_yr):
        data = data.loc[~((data.index.month == 2) & (data.index.day == 29))]

    data.index = data.index.map(lambda t: t.replace(year=int(planning_yr)))

    return data


def update_p_nom_max(n: pypsa.Network) -> None:
    # if extendable carriers (solar/onwind/...) have capacity >= 0,
    # e.g. existing assets from the OPSD project are included to the network,
    # the installed capacity might exceed the expansion limit.
    # Hence, we update the assumptions.

    n.generators.p_nom_max = n.generators[["p_nom_min", "p_nom_max"]].max(1)


# ====== SNAKEMAKE HELPERS =========


def configure_logging(
    snakemake: object, logger: logging.Logger = None, skip_handlers=False, level="INFO"
):
    """Configure the logger or the  behaviour for the logging module.

    Note: Must only be called once from the __main__ section of a script.
    The setup includes printing log messages to STDERR and to a log file defined
    by either (in priority order): snakemake.log.python, snakemake.log[0] or "logs/{rulename}.log".
    Additional keywords from logging.basicConfig are accepted via the snakemake configuration
    file under snakemake.config.logging.

    ISSUE: may not work properly with snakemake logging yaml config [to be solved]

    Args:
        snakemake (object):  snakemake script object
        logger (Logger, optional): the script logger. Defaults to None (Root logger).
            Passing a local logger will apply the configuration to the logger instead of root.
        skip_handlers (bool, optional): Do (not) skip the default handlers
            redirecting output to STDERR and file. Defaults to False.
        level (str, optional): the logging level. Defaults to "INFO".
    """

    if not logger:
        logger = logging.getLogger()
        logger.info("Configuring logging")

    kwargs = snakemake.config.get("logging", dict())
    kwargs.setdefault("level", level)

    if skip_handlers is False:
        fallback_path = Path(__file__).parent.joinpath("../..", "logs", f"{snakemake.rule}.log")
        default_logfile = snakemake.log[0] if snakemake.log else fallback_path
        logfile = snakemake.log.get("python", default_logfile)
        logger.setLevel(kwargs["level"])

        formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")

        if not os.path.exists(logfile):
            with open(logfile, "a"):
                pass
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # make running log easier to read
        logger.info("=========== NEW RUN ===========")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        # Log the exception
        logger = logging.getLogger()
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        sys.excepthook = handle_exception

    sys.excepthook = handle_exception


def mock_snakemake(
    rulename: str,
    configfiles: list | str = None,
    snakefile_path: os.PathLike = None,
    **wildcards,
):
    """A function to enable scripts to run as standalone, giving them access to
     the snakefile rule input, outputs etc

    Args:
        rulename (str): the name of the rule
        configfiles (list or str, optional): the config file or config file list. Defaults to None.
        wildcards (optional):  keyword arguments fixing the wildcards (if any needed)

    Raises:

        FileNotFoundError: Config file not found

    Returns:
        snakemake.script.Snakemake: an object storing all the rule inputs/outputs etc
    """

    import snakemake as sm
    from snakemake.api import Workflow
    from snakemake.common import SNAKEFILE_CHOICES
    from snakemake.script import Snakemake
    from snakemake.settings.types import (
        ConfigSettings,
        DAGSettings,
        ResourceSettings,
        StorageSettings,
        WorkflowSettings,
    )

    # horrible hack
    curr_path = os.getcwd()
    if snakefile_path:
        os.chdir(os.path.dirname(snakefile_path))
    try:
        snakefile = None
        for p in SNAKEFILE_CHOICES:
            if os.path.exists(p):
                snakefile = p
                break

        if snakefile is None:
            raise FileNotFoundError("Snakefile not found.")

        if configfiles is None:
            configfiles = []
        elif isinstance(configfiles, str):
            configfiles = [configfiles]

        resource_settings = ResourceSettings()
        config_settings = ConfigSettings(configfiles=map(Path, configfiles))
        workflow_settings = WorkflowSettings()
        storage_settings = StorageSettings()
        dag_settings = DAGSettings(rerun_triggers=[])
        workflow = Workflow(
            config_settings,
            resource_settings,
            workflow_settings,
            storage_settings,
            dag_settings,
            storage_provider_settings=dict(),
        )
        workflow.include(snakefile)

        if configfiles:
            for f in configfiles:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Config file {f} does not exist.")
                workflow.configfile(f)

        workflow.global_resources = {}
        rule = workflow.get_rule(rulename)
        dag = sm.dag.DAG(workflow, rules=[rule])
        wc = wildcards
        job = sm.jobs.Job(rule, dag, wc)

        def make_accessible(*ios):
            for io in ios:
                for i, _ in enumerate(io):
                    io[i] = os.path.abspath(io[i])

        make_accessible(job.input, job.output, job.log)
        snakemake = Snakemake(
            job.input,
            job.output,
            job.params,
            job.wildcards,
            job.threads,
            job.resources,
            job.log,
            job.dag.workflow.config,
            job.rule.name,
            None,
        )
        # create log and output dir if not existent
        for path in list(snakemake.log) + list(snakemake.output):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise e
    finally:
        os.chdir(curr_path)

    return snakemake
