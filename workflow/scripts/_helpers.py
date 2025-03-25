# SPDX-FileCopyrightText: : 2022 The PyPSA-Eur Authors
# SPDX-License-Identifier: MIT

# WARNING: DO NOT DO "import snakemake"

"""
Helper functions for the PyPSA China workflow including
- HPC helpers (gurobi tunnel setup)
- PyPSA helpers (legacy, time handling, ntwk relabeling)
- Snakemake helpers (logging, path management and emulators for testing)
"""
import os
import sys
import subprocess
import pandas as pd
import logging
import importlib
import time

from pathlib import Path

import pypsa

# get root logger
logger = logging.getLogger()

DEFAULT_TUNNEL_PORT = 1080
LOGIN_NODE = "01"

# TODO return pathlib objects? so can just use / to combine paths?
class PathManager:
    """A class to manage paths for the snakemake workflow

    Returns different paths for CI/CD runs (HACK due to snamekame-pytest incompatibility)
    """

    def __init__(self, snmk_config):
        self.config = snmk_config
        # HACK for pytests CI, should really be a patch but not possible
        self._is_test_run = self.config["run"].get("is_test", False)

    def _get_version(self) -> str:
        """HACK to get version from workflow pseudo-package"""
        spec = importlib.util.spec_from_file_location(
            "workflow", os.path.abspath("./workflow/__init__.py")
        )
        workflow = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workflow)
        return workflow.__version__

    def _join_scenario_vars(self) -> str:
        """Join scenario variables into a human readable string

        Returns:
            str: human readable string to build directories
        """
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

    def results_dir(self, extra_opts: dict = None) -> os.PathLike:
        """generate the results directory

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
        return os.path.join(self.config["results_dir"], base_dir, sub_dir)

    def derived_data_dir(self, shared=False) -> os.PathLike:
        """Generate the derived data directory path.

        Args:
            shared (bool, optional): If True, return the shared derived data directory.
                         Defaults to False.

        Returns:
            os.PathLike: The path to the derived data directory.
        """

        base_path = "tests" if self._is_test_run else "resources"

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
            os.PathLike: The path to the cutouts directory."""

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

    # bash commands for tunnel: reduce pipe err severity (too high from snakemake)
    pipe_err = "set -o pipefail; "
    ssh_command = f"ssh -vvv -fN -D {port} {user}@login{LOGIN_NODE}"
    logger.info(f"Attempting ssh tunnel to login node {LOGIN_NODE}")
    # Run SSH in the background to establish the tunnel
    socks_proc = subprocess.Popen(
        pipe_err + ssh_command, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE
    )
    try:
        time.sleep(0.2)
        # [-1] because ssh is last command
        err = socks_proc.communicate(timeout=2)[-1].decode()
        logger.info(f"ssh err returns {str(err)}")
        if err.find("Permission") != -1 or err.find("Could not resolve hostname") != -1:
            socks_proc.kill()
        else:
            logger.info("Gurobi Environment variables & tunnel set up successfully at attempt {i}.")
    except subprocess.TimeoutExpired:
        logger.info(
            f"SSH tunnel established on port {port} with possible errors (err check timedout)."
        )

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

    return socks_proc



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


def mock_solve(n: pypsa.Network) -> pypsa.Network:
    """Mock the solving step for tests

    Args:
        n (pypsa.Network): the network object
    """
    for c in n.iterate_components(components=["Generator", "Link", "Store", "LineType"]):
        opt_cols = [col for col in c.df.columns if col.endswith("opt")]
        base_cols = [col.split("_opt")[0] for col in opt_cols]
        c.df[opt_cols] = c.df[base_cols]
    return n


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
