import os
import glob
import pytest
import yaml

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_DIR = os.path.join(root_dir, "config")
config_files =  glob.glob(os.path.join(CONFIG_DIR, "*.yml")) + glob.glob(os.path.join(CONFIG_DIR, "*.yaml"))

@pytest.mark.parametrize("yaml_file", config_files)
def test_yaml_file_valid_and_importable(yaml_file):
    # Test that the YAML file is valid and can be loaded
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict) or isinstance(data, list)
