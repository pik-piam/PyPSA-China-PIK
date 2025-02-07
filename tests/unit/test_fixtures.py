import yaml


def test_write_config(config_file):
    with open(config_file, "r") as f:
        config_data = yaml.load(f, Loader=yaml.SafeLoader)
    # Perform your tests with config_data here
    assert config_data["scenario"]["planning_horizons"] == 2040
