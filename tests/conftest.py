"""
FIXTURES & monkey patches for pytests
Note that conftest functions are automatically discovered by pytest
"""

import pathlib
import yaml
import pytest
from os import PathLike
import matplotlib
import shutil

from hashlib import sha256 as hash256
from os import remove

# from filelock import BaseFileLock

from constants import TESTS_RUNNAME, TESTS_CUTOUT
from _helpers import PathManager

DEFAULT_CONFIG = pathlib.Path(pathlib.Path.cwd(), "config", "default_config.yaml")
TECH_CONFIG = pathlib.Path(pathlib.Path.cwd(), "config", "technology_config.yaml")


@pytest.fixture(scope="session", autouse=True)
def set_matplotlib_backend():
    # make sure matplotlib is not plotting (backend with no rendering)
    matplotlib.use("Agg")


def load_config(config_path: PathLike) -> dict:
    """load a config file
    Args:
        config_path (PathLike): the path to the config file

    Returns:
        dict: the config file as a dictionary
    """
    with open(config_path) as file:
        config_dict = yaml.safe_load(file)
    return config_dict


@pytest.fixture(scope="module")
def make_snakemake_test_config(tmp_path_factory) -> dict:
    """make a test config for snamekemake based on the default config
    Example:
        conf_dict = make_snamkemake_test_config({"scenario":{"planning_horizons":2030}})
    Returns:
        dict: the test config
    """

    def make(
        time_res=24, plan_year=2040, start_d="01-02 00:00", end_d="01-02 23:00", **kwargs
    ) -> dict:

        base_config = load_config(DEFAULT_CONFIG)
        # base_config.update(load_config(TECH_CONFIG))
        base_config.update(kwargs)

        test_config = base_config.copy()
        if isinstance(plan_year, int):
            plan_year = [plan_year]
        test_config["scenario"]["planning_horizons"] = plan_year
        test_config["snapshots"]["freq"] = f"{time_res}h"
        test_config["snapshots"]["frequency"] = time_res
        test_config["snapshots"]["start"] = start_d
        test_config["snapshots"]["end"] = end_d
        # do not setup tunnel
        # test_config["solving"]["solver"] = None
        test_config["solving"]["gurobi_hpc_tunnel"]["use_tunnel"] = False
        # do not build rasters
        test_config["enable"] = {k: False for k in test_config["enable"]}
        # set the paths & run name (PathManager reacts to TESTS_RUNNAME)
        test_config["results_dir"] = str(tmp_path_factory.mktemp("results"))
        test_config["run"]["name"] = TESTS_RUNNAME
        test_config["run"]["is_test"] = True
        test_config["paths"]["costs_dir"] = None
        

        # mock the atlite cutout config
        test_config["atlite"]["cutout_name"] = TESTS_CUTOUT
        test_config["atlite"]["cutouts"] = {
            TESTS_CUTOUT: {"weather_year": 2020, "module": "era5", "dx": 5, "dy": 5}
        }

        return test_config

    return make


@pytest.fixture(scope="module")
def make_test_config_file(make_snakemake_test_config, tmpdir_factory, request):
    """Fixture to save a temp config file for testing, return its path,
    and clean up after module.
    """

    # Get parameters passed via pytest.mark.parametrize
    time_res = request.param.get("time_res", 24)
    plan_year = request.param.get("plan_year", 2040)
    kwargs = {k: v for k, v in request.param.items() if k not in ["time_res", "plan_year"]}

    # Helper function to create a unique filename from the config arguments
    def generate_filename(*args, **kwargs):
        config_str = f"{args}_{kwargs}"
        hash_object = hash256(config_str.encode())
        return f"test_config_{hash_object.hexdigest()[:8]}.yaml"

    # Create a temporary directory for the module
    temp_dir = tmpdir_factory.mktemp("config_dir")
    # temp_dir = pathlib.Path("tests/")

    # Generate the test config
    test_config = make_snakemake_test_config(time_res=time_res, plan_year=plan_year, **kwargs)

    # Generate a unique filename based on the arguments
    config_filename = generate_filename(time_res=time_res, plan_year=plan_year)

    # Define the file path for the YAML file
    config_file_path = temp_dir / config_filename

    # Write the test config to the YAML file
    with open(config_file_path, "w") as f:
        yaml.dump(test_config, f, sort_keys=False)

    # Yield the file path for use in tests
    yield str(config_file_path)
