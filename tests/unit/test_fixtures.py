""" 
test the fixture generation
"""

import yaml
import pytest


def test_make_snakemake_test_config(make_snakemake_test_config):
    # Test the fixture generation
    config = make_snakemake_test_config(time_res=48, plan_year=2060)
    assert config["scenario"]["planning_horizons"] == 2060
    assert config["snapshots"]["freq"] == "48h"


# Test the fixture generation
@pytest.mark.parametrize(
    "make_test_config_file, expected_yr, expected_time_res",
    [
        ({"time_res": 48, "plan_year": 2060, "foresight": "overnight"}, 2060, "48h"),
        ({"time_res": 24, "plan_year": 2040, "foresight": "myopic"}, 2040, "24h"),
    ],
    indirect=["make_test_config_file"],
)
def test_write_config(make_test_config_file, expected_yr, expected_time_res):
    with open(make_test_config_file, "r") as f:
        config_data = yaml.load(f, Loader=yaml.SafeLoader)
    # Perform your tests with config_data here
    assert config_data["scenario"]["planning_horizons"] == expected_yr
    assert config_data["snapshots"]["freq"] == expected_time_res


# Test the fixture generation
@pytest.mark.parametrize(
    "make_test_config_args, expected_yr, expected_time_res",
    [
        ({"time_res": 48, "plan_year": 2060, "foresight": "overnight"}, 2060, "48h"),
        ({"time_res": 24, "plan_year": 2040, "foresight": "myopic"}, 2040, "24h"),
    ],
    indirect=["make_test_config_args"],
)
def test_make_test_config_args(make_test_config_args, expected_yr, expected_time_res):
    # Test the fixture generation
    config_args = make_test_config_args
    assert f"'planning_horizons={expected_yr}'" in config_args, config_args
    assert f"'freq={expected_time_res}'" in config_args
