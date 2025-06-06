import os
import pytest
from unittest import mock
import sys

from _helpers import ConfigManager, GHGConfigHandler, PathManager

@pytest.fixture
def sample_config():
    return {
        "scenario": {
            "planning_horizons": [2020, 2030],
            "co2_pathway": ["test_co2_scen"],
            "topology": "test_topo",
            "heating_demand": "test_proj",
            "foresight": "overnight"
        },
        "co2_scenarios": {
            "test_co2_scen": {
                "control": "reduction",
                "pathway": {"2020": 0.1, "2030": 0.2}
            }
        },
        "run": {"name": "testrun", "is_test": True},
        "foresight": "overnight",
        "paths": {
            "results_dir": "results",
            "costs_dir": "",
            "yearly_regional_load": {"ac": "resources/data/load/Provincial_Load_2020_2060_MWh.csv"}
        },
        "atlite": {
            "cutout_name": "cutout_test",
            "cutouts": {"cutout_test": {"param": 1}}
        },
        "enable": {"build_cutout": True},
        "renewable": {"wind": {"foo": "bar"},
        "heat_coupling": True
        },
    }


def test_config_manager_init_and_handle_scenarios(sample_config):
    cm = ConfigManager(sample_config)
    result = cm.handle_scenarios()
    assert isinstance(result, dict)
    assert "scenario" in result


def test_config_manager_fetch_co2_restriction(sample_config):
    cm = ConfigManager(sample_config)
    cm.handle_scenarios()
    res = cm.fetch_co2_restriction("test_co2_scen", 2020)
    assert "co2_pr_or_limit" in res
    assert "control" in res


def test_config_manager_make_wildcards(sample_config):
    cm = ConfigManager(sample_config)
    with pytest.raises(NotImplementedError):
        cm.make_wildcards()


def test_ghg_handler_valid(sample_config):
    handler = GHGConfigHandler(sample_config)
    out = handler.handle_ghg_scenarios()
    assert isinstance(out, dict)
    assert "co2_scenarios" in out


def test_ghg_handler_invalid_control(sample_config):
    bad_config = dict(sample_config)
    bad_config["co2_scenarios"] = {
        "test_co2_scen": {"control": "invalid", "pathway": {"2020": 0.1, "2030": 0.2}}
    }
    print(bad_config)
    with pytest.raises(ValueError):
        GHGConfigHandler(bad_config)


def test_ghg_handler_missing_keys(sample_config):
    bad_config = dict(sample_config)
    bad_config["co2_scenarios"] = {
        "test_co2_scen": {"control": "reduction"}
    }
    with pytest.raises(ValueError):
        GHGConfigHandler(bad_config)


def test_path_manager_costs_dir_default(sample_config):
    pm = PathManager(sample_config)
    with mock.patch("os.path.exists", return_value=True):
        assert pm.costs_dir() == "resources/data/costs"
        assert not pm.costs_dir().endswith("/")


def test_path_manager_costs_dir_absolute(sample_config):
    pm = PathManager(sample_config)
    sample_config["paths"]["costs_dir"] = "some/rel/path"
    with mock.patch("os.path.exists", return_value=False), \
         mock.patch("os.path.abspath", return_value="/abs/path"):
        assert pm.costs_dir() == "/abs/path"


def test_path_manager_derived_data_dir(sample_config):
    pm = PathManager(sample_config)
    ddir = pm.derived_data_dir()
    assert "tests/derived_data" in ddir


def test_path_manager_logs_dir(sample_config):
    pm = PathManager(sample_config)
    with mock.patch.object(PathManager, "_get_version", return_value="1.0.0"):
        ldir = pm.logs_dir()
        assert ldir.startswith("logs")


def test_path_manager_cutouts_dir(sample_config):
    pm = PathManager(sample_config)
    assert pm.cutouts_dir() == "tests/testdata"


def test_path_manager_landuse_raster_data(sample_config):
    pm = PathManager(sample_config)
    assert pm.landuse_raster_data() == "tests/testdata/landuse_availability"


def test_path_manager_profile_base_p(sample_config):
    pm = PathManager(sample_config)
    base = pm.profile_base_p("wind")
    assert "cutout_cutout_test" in base
    assert "foo" in base