import pytest
import subprocess
import logging
import shutil


# Test the workflow for different foresights, years and time resolutions
# serial needed as snakemake locks directory
@pytest.mark.serial
@pytest.mark.parametrize(
    "make_test_config_file",
    [
        ({"time_res": 1752, "plan_year": [2040], "heat_coupling": True, "foresight": "overnight"}),
        # ({"time_res": 1460, "plan_year": 2060, "heat_coupling": True, "foresight": "myopic"}),
        # (
        #     {
        #         "time_res": 5,
        #         "start_d": "02-02 00:00",
        #         "end_d": "02-04 18:00",
        #         "plan_year": 2060,
        #         "heat_coupling": False,
        #         "foresight": "overnight",
        #     }
        # ),
    ],
    indirect=True,
)
def test_dry_run(make_test_config_file):
    """Simple workflow test to check the snakemake inputs and outputs are valid"""
    cfg = make_test_config_file
    cmd = f"snakemake --configfile {cfg} -n"
    res = subprocess.run(cmd, check=True, shell=True, capture_output=True, text=True)
    logging.error(res.stderr)
    if res.returncode != 0:
        shutil.copy(cfg, "tests/failed_test_config.yaml")
    assert res.returncode == 0, "Workflow dry-run not working"


@pytest.mark.parametrize(
    "make_test_config_file",
    [({"time_res": 1752, "plan_year": [2040], "heat_coupling": True, "foresight": "overnight"})],
    indirect=True,
)
def test_dry_run_build_cutouts(make_test_config_file):
    """Simple workflow test to check the snakemake inputs and outputs are valid"""
    cfg = make_test_config_file
    cmd = f'snakemake --configfile {cfg} -n --config \'enable={{"build_cutout: 1","retrieve_cutout: 1","retrieve_raster: 1"}}\''
    res = subprocess.run(cmd, check=True, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        shutil.copy(cfg, "tests/failed_test_config.yaml")
    assert res.returncode == 0, "Workflow dry-run not working"


@pytest.mark.parametrize(
    "make_test_config_file",
    [({"time_res": 1752, "plan_year": [2040], "heat_coupling": True, "foresight": "overnight"})],
    indirect=True,
)
def test_workflow(make_test_config_file):

    # snakemake command to test up to prepare network
    cmd = f"snakemake --configfile {make_test_config_file}"
    res = subprocess.run(
        cmd, check=True, shell=True, capture_output=True, text=True, universal_newlines=True
    )
    logging.error(res.stderr.decode().split("\n"))
    if res.returncode != 0:
        shutil.copy(make_test_config_file, "tests/failed_test_config.yaml")
    assert res.returncode == 0, "Workflow is broken "


# # Test the workflow for different foresights, years and time resolutions
# # serial needed as snakemake locks directory
# @pytest.mark.serial
# @pytest.mark.make_test_config_args(
#     "make_test_config_file",
#     [
#         ({"time_res": 1752, "plan_year": [2040], "heat_coupling": True, "foresight": "overnight"}),
#         ({"time_res": 1460, "plan_year": 2060, "heat_coupling": True, "foresight": "myopic"}),
#         (
#             {
#                 "time_res": 5,
#                 "start_d": "04-01 00:00",
#                 "end_d": "04-01 18:00",
#                 "plan_year": 2060,
#                 "heat_coupling": False,
#                 "foresight": "overnight",
#             }
#         ),
#     ],
#     indirect=True,
# )
# def test_workflow(make_test_config_args):
#     # snakemake command
#     cmd = f"snakemake --config {make_test_config_args} --use-conda"
#     res = subprocess.run(cmd, check=True, shell=True, capture_output=True, text=True)

#     assert res.returncode == 0, "Workflow not working "
