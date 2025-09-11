"""
Prepare remind outputs for pypsa-coupled runs using the Remind-PyPSA-coupling package
"""

REMIND_REGION = config["run"].get("remind", {}).get("region")


# Only generate EV references if sector coupling is enabled
if config.get("sector_coupling", {}).get("enable", False):
    rule generate_regional_references:
        """
        Generate reference data files for different departments
        """
        params:
            gompertz_config=config.get("gompertz", {}),
            years=config["scenario"]["planning_horizons"],
        input:
            historical_gdp="resources/data/load/History_GDP.csv",
            historical_pop="resources/data/load/History_POP.csv",
            historical_cars="resources/data/load/History_private_car.csv",
            ssp2_pop="resources/data/load/SSPs_POP_Prov_v2.xlsx",
            ssp2_gdp="resources/data/load/SSPs_GDP_Prov_v2.xlsx",
        output:
            ev_passenger_reference=DERIVED_DATA + "/remind/references/ev_passenger_shares.csv",
            ev_freight_reference=DERIVED_DATA + "/remind/references/ev_freight_shares.csv",
        log:
            LOG_DIR + "/remind_coupling/generate_references.log",
        conda:
            "remind-coupling"
        script:
            "../scripts/remind_coupling/generate_regional_references.py"


rule build_run_config:
    """
    Build the run config for the pypsa-coupled run. This extracts the REMIND Co2 price and makes a corresponding
       pypsa CO2 scenario named after the remind c_expname
    # TODO: output per remind run & snakemake run?
    Example:
        snakemake resources/tmp/remind_coupled.yaml --cores 1 --configfile=config/templates/remind_cpled.yml
    """
    params:
        remind_region=REMIND_REGION,
        expname_max_len=20,
        currency_conv=0.912,
    input:
        remind_output=config["paths"].get("remind_outpt_dir", ""),
        config_template="config/templates/remind_cpled.yml",
    output:
        coupled_config="resources/tmp/remind_coupled.yaml",
    log:
        LOGS_COMMON + "/remind_coupling/build_run_cfg.log",
    conda:
        "remind-coupling"
    script:
        "../scripts/remind_coupling/make_pypsa_config.py"


rule transform_remind_data:
    """
    Import the remind data from the remind output & transform it to the pypsa-china format using
    """
    params:
        etl_cfg=config.get("remind_etl"),
        region=REMIND_REGION,  # overlaps with cofnig
        use_gdx=False,
    input:
        pypsa_costs=path_manager.costs_dir(ignore_remind=True),
        remind_output_dir=config["paths"].get("remind_outpt_dir", ""),
    output:
        **{
            f"costs_{yr}": DERIVED_DATA + f"/remind/costs/costs_{yr}.csv"
            for yr in config["scenario"]["planning_horizons"]
        },
        loads=DERIVED_DATA + "/remind/yrly_loads.csv",
        remind_caps=DERIVED_DATA + "/remind/preinv_capacities.csv",
        remind_tech_groups=DERIVED_DATA + "/remind/tech_groups.csv",
    log:
        LOG_DIR + "/remind_coupling/transform_remind_data.log",
    conda:
        "remind-coupling"
    script:
        "../scripts/remind_coupling/generic_etl.py"

# For the sector coupling, we can define it in the python file to enable/disable the sector coupling
rule disaggregate_remind_data:
    """
    Disaggregate the data from the remind output to the network time and spatial resolutions
    """
    params:
        etl_cfg=config.get("remind_etl"),
        region=REMIND_REGION,  # overlaps with cofnig
        reference_load_year=lambda wildcards: config["scenario"]["planning_horizons"][0],
        expand_dirs=config["scenario"]["planning_horizons"],
        separate_loads=config["run"].get("separate_loads", False),
    input:
        pypsa_powerplants=DERIVED_DATA + f"/existing_infrastructure/capacities.csv",
        remind_caps=DERIVED_DATA + "/remind/preinv_capacities.csv",
        remind_tech_groups=DERIVED_DATA + "/remind/tech_groups.csv",
        loads=DERIVED_DATA + "/remind/yrly_loads.csv",
        # todo switch to default?
        reference_load="resources/data/load/Provincial_Load_2020_2060_MWh.csv",
        ev_pass_reference=DERIVED_DATA + "/remind/references/ev_passenger_shares.csv",
        ev_freight_reference=DERIVED_DATA + "/remind/references/ev_freight_shares.csv",
    output:
        capacities=DERIVED_DATA + "/remind/harmonized_capacities/capacities.csv",
        paid_off=DERIVED_DATA + "/remind/harmonized_capacities/paid_off_capacities.csv",
        disagg_load=DERIVED_DATA + "/remind/ac_load_disagg.csv",
    log:
        LOG_DIR + "/remind_coupling/disaggregate_data.log",
    conda:
        "remind-coupling"
    script:
        "../scripts/remind_coupling/disaggregate_data.py"
