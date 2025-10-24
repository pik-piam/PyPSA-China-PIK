# SPDX-FileCopyrightText: : 2022 The PyPSA-Eur Authors
# SPDX-License-Identifier: MIT

# WARNING: DO NOT DO "import snakemake"

"""
Helper functions for the PyPSA China workflow including
- HPC helpers (gurobi tunnel setup)
- Snakemake helpers (logging, path management and emulators for testing)
"""

import functools
import importlib
import logging
import multiprocessing
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import gurobipy

# get root logger
logger = logging.getLogger()

DEFAULT_TUNNEL_PORT = 1080
LOGIN_NODE = "01"

# ============== Path & config Management ==================


class ConfigManager:
    """Config manager class for the snakemake configs"""

    def __init__(self, config: dict):
        self._raw_config = deepcopy(config)
        self.config = deepcopy(config)
        self.wildcards = {}

    def handle_scenarios(self) -> dict:
        """Unpack and filter scenarios from the configuration.

        Processes planning horizons by converting them to integers and handles
        GHG scenarios through the GHGConfigHandler. This method modifies the
        internal config state.
        
        Returns:
            dict: The processed configuration with validated scenarios.
            
        Example:
            >>> config_manager = ConfigManager(raw_config)
            >>> processed_config = config_manager.handle_scenarios()
            >>> print(processed_config['scenario']['planning_horizons'])
            [2020, 2030, 2040, 2050]
        """
        self.config["scenario"]["planning_horizons"] = [
            int(v) for v in self._raw_config["scenario"]["planning_horizons"]
        ]
        ghg_handler = GHGConfigHandler(self.config.copy())
        self.config = ghg_handler.handle_ghg_scenarios()

        return self.config

    def fetch_co2_restriction(self, pthw_name: str, year: str) -> dict:
        """Fetch CO2 restriction parameters for a specific scenario and year.
        
        Retrieves the CO2 emission reduction or price limit for a given scenario
        pathway and planning year from the configuration.
        
        Args:
            pthw_name (str): The name of the CO2 scenario pathway (e.g., 'exp175default').
            year (str): The planning year as a string (e.g., '2030').
            
        Returns:
            dict: A dictionary containing:
                - 'co2_pr_or_limit': The CO2 reduction fraction or price limit
                - 'control': The control method ('reduction', 'price', 'budget', etc.)
                
        Raises:
            KeyError: If the pathway name or year is not found in the configuration.
            
        Example:
            >>> config_manager = ConfigManager(config)
            >>> restriction = config_manager.fetch_co2_restriction('exp175default', '2030')
            >>> print(restriction)
            {'co2_pr_or_limit': 0.41175086, 'control': 'reduction'}
        """
        scenario = self.config["co2_scenarios"][pthw_name]
        return {
            "co2_pr_or_limit": scenario["pathway"][year],
            "control": scenario["control"],
        }

    def make_wildcards(self) -> list:
        """Expand wildcards in config"""
        raise NotImplementedError


# TODO add dataclass for Config


class GHGConfigHandler:
    """A class to handle & validate GHG scenarios in the config"""

    def __init__(self, config: dict):
        self.config = deepcopy(config)
        self._raw_config = deepcopy(config)
        self._validate_scenarios()

    def handle_ghg_scenarios(self) -> dict:
        """Handle ghg scenarios (parse, valdiate & unpack to config[scenario])

        Returns:
            dict: validated and parsed
        """
        # HACK for snakemake access
        scripts_dir = os.path.abspath(os.path.dirname(__file__))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        if self.config.get("heat_coupling", False):
            # HACK import here for snakemake access
            from constants import CO2_BASEYEAR_EM as base_year_ems
        else:
            from constants import CO2_EL_2020 as base_year_ems

        for name, co2_scen in self.config["co2_scenarios"].items():
            co2_scen["pathway"] = {int(k): v for k, v in co2_scen.get("pathway", {}).items()}
        self._reduction_to_budget(base_year_ems)
        self._filter_active_scenarios()
        return self.config

    def _filter_active_scenarios(self):
        """Select active ghg scenarios"""
        scenarios = self.config["scenario"].get("co2_pathway", [])
        if not isinstance(scenarios, list):
            scenarios = [scenarios]

        self.config["co2_scenarios"] = {
            k: v for k, v in self.config["co2_scenarios"].items() if k in scenarios
        }

    def _reduction_to_budget(self, base_yr_ems: float):
        """Transform reduction to budget

        Args:
            base_yr_ems (float): Base year emissions value
        """
        for name, co2_scen in self.config["co2_scenarios"].items():
            if co2_scen["control"] == "reduction":
                budget = {yr: base_yr_ems * (1 - redu) for yr, redu in co2_scen["pathway"].items()}
                self.config["co2_scenarios"][name]["pathway"] = budget
                self.config["co2_scenarios"][name]["control"] = "budget_from_reduction"

    # TODO switch to config validate, see
    # https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html#validation
    def _validate_scenarios(self):
        """Validate CO2 scenarios"""

        for name, scen in self._raw_config["co2_scenarios"].items():
            # do not validate if not selected
            if name not in self.config["scenario"]["co2_pathway"]:
                continue

            # check type
            if not isinstance(scen, dict):
                raise ValueError(f"Expected a dictionary for co2 scenario but got {scen}")

            # control type none = free emissions. DOn't validate
            if "control" in set(scen) and scen["control"] is None:
                continue

            # otherwise check expected keys in scenario
            if {"control", "pathway"} - set(scen):
                raise ValueError(f"Scenario {scen} must contain 'control' and 'pathway'")

            ALLOWED = ["price", "reduction", "budget", None]

            if scen["control"] not in ALLOWED:
                err = f"Control must be {','.join([str(x) for x in ALLOWED])} but was {name}:{scen.get('control', 'missing')}"
                raise ValueError(err)

            years_int = set(map(int, self.config["scenario"]["planning_horizons"]))
            missing_yrs = years_int - set(map(int, scen["pathway"]))
            if missing_yrs:
                raise ValueError(f"Years in scenario {scen['pathway']} missing {missing_yrs}")


# TODO return pathlib objects? so can just use / to combine paths?
# TODO unit tests for path manager
class PathManager:
    """Manages file system paths for the Snakemake workflow.
    
    This class provides centralized path management for the PyPSA-China workflow,
    handling different path configurations for production runs vs. CI/CD test runs.
    It constructs paths based on scenario configurations and wildcards.
    
    The class handles the incompatibility between Snakemake and pytest by providing
    different path behaviors when running in test mode. (Temporary workaround)
    
    Attributes:
        config: The Snakemake configuration dictionary.
        root_dir: The root directory of the project.
        
    Example:
        >>> path_manager = PathManager(snakemake_config)
        >>> results_path = path_manager.results_dir()
        >>> cutout_path = path_manager.cutouts_dir()
    """

    def __init__(self, snmk_config: dict, wildcards_map: dict = None):
        """Initialize the PathManager with configuration.
        
        Args:
            snmk_config (dict): The Snakemake configuration containing run settings.
            wildcards_map (dict, optional): mapping of wildcards for path construction.
                Defaults to None.
        """
        self.config = snmk_config
        # HACK for pytests CI, should really be a patch but not possible
        self._is_test_run = self.config["run"].get("is_test", False)

        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    def _get_version(self) -> str:
        """Get version from workflow pseudo-package.
        
        This is a workaround to extract version information from the workflow
        package for path construction purposes.
        
        Returns:
            str: The version string from the workflow package.
            
        Note:
            This is marked as a HACK and should ideally be refactored to use
            a more standard approach for version management.
        """
        spec = importlib.util.spec_from_file_location(
            "workflow", os.path.abspath("./workflow/__init__.py")
        )
        workflow = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workflow)
        return workflow.__version__

    @functools.lru_cache
    def _join_scenario_vars(self) -> str:
        """Join scenario variables into a human-readable string for directory naming.
        
        Creates a compact string representation of scenario parameters by joining
        them with underscores, using abbreviated names where configured.
        
        Returns:
            str: A human-readable string suitable for building directory paths,
                containing abbreviated scenario variable names and values.
                
        Example:
            >>> path_manager._join_scenario_vars()
            'topo_current+FCG_co2pw_exp175default_proj_positive'
        """
        # TODO make into a config
        exclude = ["planning_horizons", "co2_reduction"]
        short_names = {
            "planning_horizons": "yr",
            "topology": "topo",
            "co2_pathway": "co2pw",
            "heating_demand": "proj",
        }

        # remove heating wildcards if not needed
        if not self.config.get("heat_coupling", False):
            if "heating_demand" in self.config["scenario"]:
                _ = self.config["scenario"].pop("heating_demand")

        # remember need place holders for snakemake
        return "_".join(
            [
                f"{short_names[k] if k in short_names else k}-{{{k}}}"
                for k in self.config["scenario"]
                if k not in exclude
            ]
        )

    def results_dir(self, extra_opts: dict = None) -> os.PathLike:
        """Generate the results directory

        Args:
            extra_opts (dict, optional): opt extra args. Defaults to None.

        Returns:
            Pathlike: base directory for reslts
        """
        run, foresight = self.config["run"]["name"], self.config["foresight"]
        base_dir = "v-" + self._get_version() + "_" + run
        sub_dir = foresight + "_" + self._join_scenario_vars()
        if extra_opts:
            sub_dir += "_" + "".join(extra_opts.values())
        return os.path.join(self.config["paths"]["results_dir"], base_dir, sub_dir)

    def derived_data_dir(self, shared=False) -> os.PathLike:
        """Generate the derived data directory path.

        Args:
            shared (bool, optional): If True, return the shared derived data directory.
                         Defaults to False.

        Returns:
            os.PathLike: The path to the derived data directory.
        """

        base_path = "tests" if self._is_test_run else "resources"
        base_path = os.path.abspath(os.path.join(self.root_dir, base_path))

        foresight = self.config["foresight"]
        if not shared:
            sub_dir = foresight + "_" + self._join_scenario_vars()
            return os.path.join(f"{base_path}/derived_data", sub_dir)
        else:
            return f"{base_path}/derived_data"

    def logs_dir(self) -> os.PathLike:
        """Generate logs directory.

        Returns:
            os.PathLike: The path to the derived data directory.
        """
        run, foresight = self.config["run"]["name"], self.config["foresight"]
        base_dir = "v-" + self._get_version() + "_" + run
        sub_dir = foresight + "_" + self._join_scenario_vars()
        return os.path.join("logs", base_dir, sub_dir)

    def cutouts_dir(self) -> os.PathLike:
        """Generate cutouts directory.

        Returns:
            os.PathLike: The path to the cutouts directory.
        """

        if self._is_test_run:
            return "tests/testdata"
        else:
            return "resources/cutouts"

    def landuse_raster_data(self) -> os.PathLike:
        """Generate the landuse raster data directory path.

        Returns:
            os.PathLike: The path to the landuse raster data directory.
        """

        if self._is_test_run:
            return "tests/testdata/landuse_availability"
        else:
            return "resources/data/landuse_availability"

    def profile_base_p(self, technology: str) -> os.PathLike:
        """Generate the profile data directory base path.

        Args:
            technology (str): The technology name.

        Returns:
            os.PathLike: The path to the profile data directory.
        """
        cutout_name = self.config["atlite"]["cutout_name"]
        base_p = self.derived_data_dir(shared=True) + f"/cutout_{cutout_name}/"
        resource_cfg = self.config["renewable"][technology]
        rsrc = "_".join([f"{k}{v}" for k, v in resource_cfg.items()])

        return base_p + rsrc

    def costs_dir(self, ignore_remind=False) -> os.PathLike:
        """Get the costs directory path.
        In case a path was specified in the config, it will be used.
        Otherwise, a default path will be used.

        Args:
            ignore_remind (bool, optional): do not return the remind default,
                even if remind coupling is enabled. Defaults to False.

        Returns:
            os.PathLike: the dirname
        """

        default = "resources/data/costs/default"
        if self.config["run"].get("is_remind_coupled", False) and not ignore_remind:
            default = self.derived_data_dir() + "/remind/costs"

        costs_dir = self.config["paths"].get("costs_dir", default)
        # if not absolute path & rel not recognised by snakemake
        if not costs_dir:
            costs_dir = default
        elif not os.path.exists(costs_dir):
            # if relative path, make it absolute
            costs_dir = os.path.abspath(costs_dir)

        if costs_dir.endswith("/"):
            costs_dir = costs_dir[:-1]
        return costs_dir

    def elec_load(self, ignore_remind=False) -> os.PathLike:
        """Determine the path to the electric load data. If a path
         is specified in the config, it will be used, otherwise the defaukt
        The default path is different for remind coupled & standalone runs.

        Args:
            ignore_remind (bool, optional): use the non-remind default regardless
                of coupling
        """

        default = "resources/data/load/Provincial_Load_2020_2060_MWh.csv"
        # if remind coupling is enabled, use the remind data
        if self.config["run"].get("is_remind_coupled", False) and not ignore_remind:
            default = self.derived_data_dir() + "/remind/ac_load_disagg.csv"

        loads = self.config["paths"].get("yearly_regional_load", {"ac": default})
        if not loads["ac"]:
            loads = {"ac": default}
        elec_load = loads["ac"]
        # if not absolute path and rel not recognised by snakemake
        if not os.path.exists(elec_load):
            # if relative path, make it absolute
            elec_load = os.path.abspath(elec_load)

        return elec_load

    def infrastructure(self, ignore_remind=False) -> os.PathLike:
        """Determine the path to the existing insrastructure data. If a path
         is specified in the config, it will be used, otherwise the defaukt
        The default path is different for remind coupled & standalone runs.

        Args:
            ignore_remind (bool, optional): use the non-remind default regardless
                of coupling
        """
        default = self.derived_data_dir() + "/existing_infrastructure"
        if self.config["run"].get("is_remind_coupled", False) and not ignore_remind:
            default = self.derived_data_dir() + "/remind/harmonized_capacities"

        infra_dir = self.config["paths"].get("existing_infra", default)
        # if not absolute path & rel not recognised by snakemake
        if not infra_dir:
            infra_dir = default
        elif not os.path.exists(infra_dir):
            # if relative path, make it absolute
            infra_dir = os.path.abspath(infra_dir)

        return infra_dir


# ============== HPC helpers ==================


def setup_gurobi_tunnel_and_env(
    tunnel_config: dict, logger: logging.Logger = None, attempts=4
) -> subprocess.Popen:
    """A utility function to set up the Gurobi environment variables and establish an
    SSH tunnel on HPCs. Otherwise the license check will fail if the compute nodes do
     not have internet access or a token server isn't set up

    Args:
        config (dict): the snakemake pypsa-china configuration
        logger (logging.Logger, optional): Logger. Defaults to None.
        attempts (int, optional): ssh connection attemps. Defaults to 4.
    """
    if not tunnel_config.get("use_tunnel", False):
        return
    logger.info("setting up tunnel")
    user = os.getenv("USER")  # User is pulled from the environment
    port = tunnel_config.get("tunnel_port", DEFAULT_TUNNEL_PORT)
    login_node = tunnel_config.get("login_node", LOGIN_NODE)
    timeout = tunnel_config.get("timeout_s", 60)

    # bash commands for tunnel: reduce pipe err severity (too high from snakemake)
    pipe_err = "set -o pipefail; "
    ssh_command = f"ssh -vvv -fN -D {port} -o ConnectTimeout={timeout} {user}@login{login_node}"
    logger.info(f"Attempting ssh tunnel to login node {login_node}")
    # Run SSH in the background to establish the tunnel
    socks_proc = subprocess.Popen(
        pipe_err + ssh_command,
        shell=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    try:
        stdout, stderr = socks_proc.communicate(timeout=timeout + 2)
        err = stderr.decode()
        logger.info(f"ssh err returns {str(err)}")
        logger.info(f"ssh stdout returns {str(stdout)}")
        if err.find("Permission") != -1 or err.find("Could not resolve hostname") != -1:
            socks_proc.kill()
        else:
            logger.info("Gurobi Environment variables & tunnel set up successfully at attempt {i}.")
    except subprocess.TimeoutExpired:
        logger.error("SSH tunnel communication timed out.")

    os.environ["https_proxy"] = f"socks5://127.0.0.1:{port}"
    os.environ["SSL_CERT_FILE"] = tunnel_config.get(
        "ssl_cert", "/p/projects/rd3mod/ssl/ca-bundle.pem_2022-02-08"
    )
    os.environ["GRB_CAFILE"] = tunnel_config.get(
        "grb_cafile", "/p/projects/rd3mod/ssl/ca-bundle.pem_2022-02-08"
    )

    # Set up Gurobi environment variables
    # TODO soft code
    os.environ["GUROBI_HOME"] = tunnel_config.get(
        "gurobi_home", "/p/projects/rd3mod/gurobi1103/linux64"
    )
    os.environ["PATH"] += f":{os.environ['GUROBI_HOME']}/bin"
    if "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] += f":{os.environ['GUROBI_HOME']}/lib"
    os.environ["GRB_LICENSE_FILE"] = tunnel_config.get(
        "license_path", "/p/projects/rd3mod/gurobi_rc/gurobi.lic"
    )
    os.environ["GRB_CURLVERBOSE"] = tunnel_config.get("verbose", "1")
    os.environ["GRB_SERVER_TIMEOUT"] = tunnel_config.get("timeout", "10")

    return socks_proc


def _check_gurobi_license_subprocess() -> bool:
    """
    Subprocess function to check Gurobi license availability.
    This function will start the Gurobi environment to verify if a license is available.

    Returns:
        bool: True if the license check succeeded, False otherwise.
    """
    try:
        env = gurobipy.Env(empty=True)
        env.start()  # Start the Gurobi environment (this will attempt to acquire the license)
        logger.info("Gurobi license is available.")
        env.dispose()  # Dispose of the environment after use
        return True
    except gurobipy.GurobiError as e:
        logger.error(f"Error checking Gurobi license: {e}")
        return False


def check_gurobi_license(attempts=1, timeout=10) -> bool:
    """
    Checks the availability of the Gurobi license in a subprocess with timeout.

    Args:
        attempts (int): Number of attempts.
        timeout (int): Time to wait before retrying (in seconds).

    Returns:
        bool: True if the license is available, False if the check times out.
    """
    logger.info("Checking Gurobi license availability...")

    for _ in range(attempts):
        # Create a multiprocessing Process to check license
        process = multiprocessing.Process(target=_check_gurobi_license_subprocess)
        process.start()

        process.join(timeout=timeout)  # Wait for the process to finish or timeout

        if process.is_alive():
            # If the process is still alive after the timeout, terminate it
            process.terminate()
            process.join()  # Ensure it is properly joined to clean up
            logger.warning("License check timeout. Retrying...")
        else:
            # If the process completed, check the result
            if process.exitcode == 0:
                # License was available
                return True
            else:
                # License was not available
                logger.warning("License not available during subprocess check. Retrying...")

    return False


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


def get_cutout_params(config: dict) -> dict:
    """Get the cutout parameters from the config file

    Args:
        config (dict): the snakemake config
    Raises:
        ValueError: if no parameters are found for the cutout name
        FileNotFoundError: if the cutout is not built & build_cutout is disabled
    Returns:
        dict: the cutout parameters
    """
    cutout_name = config["atlite"]["cutout_name"]
    cutout_params = config["atlite"]["cutouts"].get(cutout_name, None)

    if cutout_params is None:
        err = f"No cutout parameters found for {cutout_name}"
        raise ValueError(err + " in config['atlite']['cutouts'].")
    elif not config["enable"]["build_cutout"]:
        cutouts_dir = PathManager(config).cutouts_dir()
        is_built = os.path.exists(os.path.join(cutouts_dir, f"{cutout_name}.nc"))
        if not is_built:
            err = f"Cutout {cutout_name} not found in {cutouts_dir}, enable build_cutout"
            raise FileNotFoundError(err)
    return cutout_params


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


def set_plot_test_backend(config: dict):
    """Hack to set the matplotlib backend to Agg for testing
    Not possible via normal conftest.py since snakemake is a subprocess

    Args:
        config (dict): the snakemake config
    """
    is_test = config["run"].get("is_test", False)
    if is_test:
        import matplotlib

        matplotlib.use("Agg")
