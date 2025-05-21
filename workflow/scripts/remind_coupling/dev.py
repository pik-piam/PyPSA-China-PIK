import importlib  
import os
import pandas as pd
import logging

remindcoupling = importlib.import_module("REMIND-PyPSA-coupling")
remind_etl = importlib.import_module("REMIND-PyPSA-coupling.remind2pypsa_etl")

logger = logging.getLogger(__name__)

MW_C = 12  # g/mol
MW_CO2 = 2 * 16 + MW_C  # g/mol
UNIT_CONVERSION = {
    "capex": 1e6,  # TUSD/TW(h) to USD/MW(h)
    "VOM": 1e6 / 8760,  # TUSD/TWa to USD/MWh
    "FOM": 100,  # p.u to percent
    "co2_intensity": 1e9 * (MW_CO2 / MW_C) / 8760 / 1e6,  # Gt_C/TWa to t_CO2/MWh
}

STOR_TECHS = ["h2stor", "btstor", "phs"]
REMIND_PARAM_MAP = {
    "tech_data": "pm_data",
    "capex": "p32_capCost",
    "eta": "pm_dataeta",
    "eta_part2": "pm_eta_conv",
    # TODO export converged too
    "fuel_costs": "p32_PEPriceAvg",
    "discount_r": "p32_discountRate",
    "co2_intensity": "pm_emifac",
    "weights_gen": "p32_weightGen",
}



if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    region = "CHA"  # China w Maccau, Taiwan

    # make paths
    # the remind export uses the name of the symbol as the file name
    # base_path = os.path.join(os.path.abspath(root_dir + "/.."), "gams_learning/pypsa_export/")
    base_path = os.path.expanduser(
        "~/downloads/output_REMIND/SSP2-Budg1000-PyPSAxprt_2025-05-09/pypsa_export"
    )
    paths = {
        key: os.path.join(base_path, value + ".csv") for key, value in REMIND_PARAM_MAP.items()
    }

    # load the data
    frames = {k: remindcoupling.read_remind_csv(v) for k, v in paths.items()}
    frames = {
        k: df.query("region == @region").drop(columns="region") if "region" in df.columns else df
        for k, df in frames.items()
    }
    # special case, eff split across two tables
    frames["eta"] = pd.concat([frames["eta"], frames["eta_part2"]]).drop_duplicates().reset_index()

    # get remind version
    with open(os.path.join(base_path, "c_model_version.csv"), "r") as f:
        remind_v = f.read().split("\n")[1].replace(",", "").replace(" ", "")

    # make the stitched weight frames
    weight_frames = [frames[k].assign(weight_type=k) for k in frames if k.startswith("weights")]
    weights = pd.concat(
        [df.rename(columns={"carrier": "technology", "value": "weight"}) for df in weight_frames]
    )

    # TODO switch with settings
    # years
    years = frames["capex"].year.unique()

    # make a pypsa like cost table, with remind values
    costs_remind = remindcoupling.make_pypsa_like_costs(frames)
    # add weights by techs
    costs_remind = costs_remind.merge(weights, on=["technology", "year"], how="left")

    # load the mapping
    mappings = pd.read_csv(root_dir + "/data/techmapping_remind2py.csv")
    mappings.loc[:, "reference"] = mappings["reference"].apply(to_list)

    # check the data & mappings
    remindcoupling.validate_mappings(mappings)
    remindcoupling.validate_remind_data(costs_remind, mappings)

    # load pypsa costs
    pypsa_costs_dir = os.path.join(
        os.path.abspath(root_dir + "/.."), "PyPSA-China-PIK/resources/data/costs"
    )
    pypsa_cost_files = [os.path.join(pypsa_costs_dir, f) for f in os.listdir(pypsa_costs_dir)]
    pypsa_costs = pd.read_csv(pypsa_cost_files.pop())
    for f in pypsa_cost_files:
        pypsa_costs = pd.concat([pypsa_costs, pd.read_csv(f)])

    # apply the mappings to pypsa tech
    mapped_costs = remindcoupling.map_to_pypsa_tech(
        remind_costs_formatted=costs_remind,
        pypsa_costs=pypsa_costs,
        mappings=mappings,
        weights=weights,
        years=years,
    )
    mapped_costs["value"].fillna(0, inplace=True)
    mapped_costs.fillna(" ", inplace=True)
    logger.info(f"Writing mapped costs data to {os.path.join(root_dir, 'output')}")
    descript = f"test_remind_{remind_v}_pk1000"
    if not os.path.exists(os.path.join(root_dir, "output")):
        os.mkdir(os.path.join(root_dir, "output"))
    if not os.path.exists(os.path.join(root_dir, "output", descript)):
        os.mkdir(os.path.join(root_dir, "output", descript))
    write_cost_data(mapped_costs, root_dir + "/output/", descript=descript)

    logger.info("Finished")
