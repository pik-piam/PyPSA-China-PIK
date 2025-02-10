import pytest
import subprocess


# Test the workflow for different foresights, years and time resolutions
# serial needed as snakemake locks directory
@pytest.mark.serial
@pytest.mark.parametrize(
    "make_test_config_file",
    [
        ({"time_res": 1752, "plan_year": [2040], "heat_coupling": True, "foresight": "overnight"}),
        ({"time_res": 1460, "plan_year": 2060, "heat_coupling": True, "foresight": "myopic"}),
        (
            {
                "time_res": 5,
                "start_d": "04-01 00:00",
                "end_d": "04-01 18:00",
                "plan_year": 2060,
                "heat_coupling": False,
                "foresight": "overnight",
            }
        ),
    ],
    indirect=True,
)
def test_workflow(make_test_config_file):
    # snakemake command
    cmd = f"snakemake --configfile {make_test_config_file}"
    res = subprocess.run(cmd, check=True, shell=True, capture_output=True)

    assert res.returncode == 0, "Workflow not working "


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
#     res = subprocess.run(cmd, check=True, shell=True, capture_output=True)

#     assert res.returncode == 0, "Workflow not working "
