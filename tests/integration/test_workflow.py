import logging
import os
import shutil
import subprocess
from hashlib import sha256

import pytest

# Test the workflow for different foresights, years and time resolutions
# serial needed as snakemake locks directory


def copy_failed_config(cfg_path: os.PathLike) -> str:
    """Copy a failed config for local debugging

    Args:
        cfg_path (os.PathLike): the config path

    Returns:
        str: the hash id of the config
    """
    hash_id = sha256(cfg_path.encode()).hexdigest()
    failed_test_config_path = f"tests/failed_test_config_{hash_id}.yaml"
    shutil.copy(cfg_path, failed_test_config_path)
    return hash_id


def launch_subprocess(cmd: str, env=None) -> subprocess.CompletedProcess:
    """Launch a subprocess

    Args:
        cmd (str): a command to run
        env (os.environment.copy, optional): an environment. Defaults to None.

    Returns:
        CompletedProcess: process result
    """
    try:
        logging.debug(f"Running command: {cmd}")
        res = subprocess.run(cmd, check=True, shell=True, capture_output=True, text=True, env=env)
        logging.info("\n\t".join(res.stdout.split("\n")))
        logging.info(f"return code: {res.returncode}")
        logging.info(f"====== stderr ====== :\n {'\n\t'.join(res.stderr.split('\n'))}")
    except subprocess.CalledProcessError as e:
        logging.error(e.stderr)
        logging.error(e)
        assert False, "Workflow integration test failed"
    return res


# TODO: add existing baseyear, add remind_coupled, add_plotting
@pytest.mark.parametrize(
    "make_test_config_file",
    [
        (
            {
                "time_res": 1752,
                "plan_year": 2040,
                "heat_coupling": True,
                "foresight": "overnight",
                "existing_capacities": {"add": False},
            }
        ),
        # currently broken (fix coming)
        # (
        #     {
        #         "time_res": 24,
        #         "plan_year": 2060,
        #         "heat_coupling": True,
        #         "foresight": "myopic",
        #         "existing_capacities": {"add": False},
        #     }
        # ),
        (
            {
                "time_res": 5,
                "start_d": "02-02 00:00",
                "end_d": "02-04 18:00",
                "plan_year": 2060,
                "heat_coupling": False,
                "foresight": "overnight",
                "existing_capacities": {"add": False},
            }
        ),
    ],
    indirect=True,
)
def test_dry_run(make_test_config_file):
    """Simple workflow test to check the snakemake inputs and outputs are valid"""
    cfg = make_test_config_file
    cmd = f"snakemake --configfile {cfg} -n -f"
    cmd += " --rerun-incomplete"
    res = launch_subprocess(cmd)
    if res.returncode != 0:
        hash_id = copy_failed_config(cfg)
    assert res.returncode == 0, f"Snakemake dry run failed, config id {hash_id}"


@pytest.mark.parametrize(
    "make_test_config_file",
    [
        {
            "time_res": 1752,
            "plan_year": 2040,
            "heat_coupling": True,
            "foresight": "overnight",
            "existing_capacities": {"add": False},
        }
    ],
    indirect=True,
)
def test_dry_run_build_cutouts(make_test_config_file):
    """Simple workflow test to check the snakemake inputs and outputs are valid"""
    cfg = make_test_config_file
    cmd = f"snakemake --configfile {cfg} --rerun-incomplete"
    cmd += ' -n --config \'enable={"build_cutout: 1","retrieve_cutout: 1","retrieve_raster: 1"}\''

    res = launch_subprocess(cmd)
    if res.returncode != 0:
        hash_id = copy_failed_config(cfg)
    assert res.returncode == 0, f"Snakemake dry run w build cutouts failed, config id {hash_id}"


# TODO use case cases pluggin
@pytest.mark.parametrize(
    "make_test_config_file",
    [
        {
            "time_res": 8,
            "plan_year": 2040,
            "heat_coupling": True,
            "foresight": "overnight",
            "existing_capacities": {"add": False},
        }
    ],
    indirect=True,
)
def test_workflow(make_test_config_file):
    logging.info("Starting workflow test")
    # reduce network size

    env = os.environ.copy()
    # make smaller network by limiting province sizes
    env["PROV_NAMES"] = '["Anhui", "Jiangsu", "Shanghai"]'  # Override CONST1
    env["IS_TEST"] = "1"
    cfg = make_test_config_file
    # snakemake command to test up to prepare network
    cmd = f"snakemake --configfile {cfg}"
    cmd += " --rerun-incomplete --cores 2"
    res = launch_subprocess(cmd, env)
    if res.returncode != 0:
        hash_id = copy_failed_config(cfg)
    assert res.returncode == 0, f"Snakemake run failed, config id {hash_id}"
