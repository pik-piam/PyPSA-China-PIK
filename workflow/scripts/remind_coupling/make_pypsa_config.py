"""Script to create a PyPSA config file based on the REMIND output/config.
NB: Needs to be run before the coupled PyPSA run.

Example:
    # !! config file name needs to match the output of the snakemake rule
    `snakemake config_file_name -f --cores=1 # makes config_file_name`
    `snakemake --configfile=config_file_name` # the run
"""

# -*- coding: utf-8 -*-
import os
import yaml
import sys
import logging

import setup  # sets up paths for standalone

import rpycpl.utils as coupl_utils

logger = logging.getLogger(__name__)


def read_remind_data(remind_outp_dir: os.PathLike, region: str) -> dict:
    """
    Read relevant REMIND data from the output directory.
    Args:
        remind_outp_dir (os.PathLike): Path to the REMIND output directory.
        region (str): Remind region to filter the data by.
    Returns:
        dict: Dictionary containing REMIND data.
    """

    co2_p = (
        coupl_utils.read_remind_csv(os.path.join(remind_outp_dir, "p_priceCO2.csv"))
        .query("region == @region")
        .drop(columns=["region"])
        .set_index("year")
    )
    # get remind version
    with open(os.path.join(remind_outp_dir, "c_model_version.csv"), "r") as f:
        remind_v = f.read().split("\n")[1].replace(",", "").replace(" ", "")
    # get remind run name
    with open(os.path.join(remind_outp_dir, "c_expname.csv"), "r") as f:
        remind_exp_name = f.read().split("\n")[1].replace(",", "").replace(" ", "")

    return {"co2_prices": co2_p, "version": remind_v, "expname": remind_exp_name}


# TODO read  remind regions and write to config
# TODO centralise joint settings = overwrite hours
# TODO add disagg config


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        snakemake = setup._mock_snakemake("build_run_config")

    name_len = snakemake.params.expname_max_len
    region = snakemake.params.remind_region

    remind_data = read_remind_data(snakemake.input.remind_output, region)

    # read template config
    with open(snakemake.input.config_template) as f:
        template_cfg = yaml.safe_load(f)

    cfg = template_cfg.copy()
    cfg["scenario"]["planning_horizons"] = remind_data["co2_prices"].index.tolist()
    sc_name = remind_data["expname"][:name_len]
    if "co2_scenarios" not in cfg.keys():
        cfg["co2_scenarios"] = {sc_name: {}}
    cfg["co2_scenarios"][sc_name]["pathway"] = remind_data["co2_prices"]["value"].to_dict()
    cfg["scenario"]["co2_pathway"] = [sc_name]

    remind_cfg = {
        "remind": {
            "coupling": "1way",
            "version": remind_data["version"],
            "run_name": remind_data["expname"],
        }
    }
    cfg["run"].update({"is_remind_coupled": True})
    cfg["paths"]["costs_dir"] = None
    cfg.update(remind_cfg)

    with open(snakemake.output.coupled_config, "w") as output_file:
        yaml.dump(cfg, output_file, default_flow_style=False, sort_keys=False)
