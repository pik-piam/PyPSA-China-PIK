"""
FIXTURES for pytests
Note that conftest functions are automatically discovered by pytest
"""

import pathlib
import yaml
import pytest
from os import PathLike
import os.path
from hashlib import sha256 as hash256

DEFAULT_CONFIG = pathlib.Path(pathlib.Path.cwd(), "config", "default_config.yaml")
TECH_CONFIG = pathlib.Path(pathlib.Path.cwd(), "config", "technology_config.yaml")


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


@pytest.fixture(scope="session")
def make_snakemake_test_config() -> dict:
    """make a test config for snamekemake based on the default config
    Example:
        conf_dict = make_snamkemake_test_config({"scenario":{"planning_horizons":2030}})
    Returns:
        dict: the test config
    """

    def make(time_res=24, plan_year=2040, **kwargs) -> dict:

        base_config = load_config(DEFAULT_CONFIG)
        base_config.update(load_config(TECH_CONFIG))
        base_config.update(kwargs)

        test_config = base_config.copy()
        test_config["scenario"]["planning_horizons"] = plan_year
        test_config["snapshots"]["freq"] = f"{time_res}h"
        test_config["snapshots"]["frequency"] = time_res

        return test_config

    return make


# TODO could change scope to session if wrote a custom tempdir (built-in tmpdir scope is fn)
@pytest.fixture
def config_file(make_snakemake_test_config, tmpdir):
    """Fixture to generate the config file, return its path, and clean up after session."""

    # Helper function to create a unique filename from the config arguments
    def generate_filename(*args, **kwargs):
        # Create a unique identifier based on the arguments (e.g., time_res, plan_year, etc.)
        config_str = f"{args}_{kwargs}"
        hash_object = hash256(config_str.encode())
        return f"test_config_{hash_object.hexdigest()[:8]}.yaml"

    # Generate the test config
    test_config = make_snakemake_test_config(time_res=24, plan_year=2040)

    # Generate a unique filename based on the arguments
    config_filename = generate_filename(time_res=24, plan_year=2040)

    # Define the file path for the YAML file using the tmpdir fixture
    config_file_path = tmpdir.join(config_filename)

    # Write the test config to the YAML file
    with open(config_file_path, "w") as f:
        yaml.dump(test_config, f)

    # Yield the file path for use in tests
    yield str(config_file_path)
