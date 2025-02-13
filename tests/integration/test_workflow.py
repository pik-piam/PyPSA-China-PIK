import pytest
import subprocess
import logging
import shutil


# Test the workflow for different foresights, years and time resolutions
# serial needed as snakemake locks directory


# TODO ideally this would be a poll of the subprocess
def test_subprocess(cmd):
    try:
        logging.debug(f"Running command: {cmd}")
        res = subprocess.run(cmd, check=True, shell=True, capture_output=True, text=True)
        logging.info("\n\t".join(res.stdout.split("\n")))
        logging.info(f"return code: {res.returncode}")
        logging.info(f"====== stderr ====== :\n {"\n\t".join(res.stderr.split("\n"))}")
    except subprocess.CalledProcessError as e:
        logging.error(e.stderr)
        logging.error(e)
        assert False, "Workflow integration test failed"
    return res


# @pytest.mark.serial
@pytest.mark.parametrize(
    "make_test_config_file",
    [
        ({"time_res": 1752, "plan_year": [2040], "heat_coupling": True, "foresight": "overnight"}),
        ({"time_res": 24, "plan_year": [2060], "heat_coupling": True, "foresight": "myopic"}),
        (
            {
                "time_res": 5,
                "start_d": "02-02 00:00",
                "end_d": "02-04 18:00",
                "plan_year": 2060,
                "heat_coupling": False,
                "foresight": "overnight",
            }
        ),
    ],
    indirect=True,
)
def test_dry_run(make_test_config_file):
    """Simple workflow test to check the snakemake inputs and outputs are valid"""
    cfg = make_test_config_file
    cmd = f"snakemake --configfile {cfg} -n -f"
    res = test_subprocess(cmd)
    if res.returncode != 0:
        shutil.copy(cfg, "tests/failed_test_config.yaml")
    assert res.returncode == 0, "Snakemake dry run failed"


@pytest.mark.parametrize(
    "make_test_config_file",
    [({"time_res": 1752, "plan_year": [2040], "heat_coupling": True, "foresight": "overnight"})],
    indirect=True,
)
def test_dry_run_build_cutouts(make_test_config_file):
    """Simple workflow test to check the snakemake inputs and outputs are valid"""
    cfg = make_test_config_file
    cmd = f'snakemake --configfile {cfg} -n --config \'enable={{"build_cutout: 1","retrieve_cutout: 1","retrieve_raster: 1"}}\''
    res = test_subprocess(cmd)
    if res.returncode != 0:
        shutil.copy(cfg, "tests/failed_test_config.yaml")
    assert res.returncode == 0, "Snakemake dry run w build cutouts failed"


# TODO use case cases pluggin
@pytest.mark.serial
@pytest.mark.parametrize(
    "make_test_config_file",
    [({"time_res": 1752, "plan_year": [2040], "heat_coupling": True, "foresight": "overnight"})],
    indirect=True,
)
def test_workflow(make_test_config_file):
    logging.info("Starting workflow test")
    # snakemake command to test up to prepare network
    cfg = make_test_config_file
    cmd = f"snakemake --configfile {cfg}"
    res = test_subprocess(cmd)
    if res.returncode != 0:
        shutil.copy(cfg, "tests/failed_test_config.yaml")
    assert res.returncode == 0, "Snakemake run failed"
