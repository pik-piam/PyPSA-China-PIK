"""
Prepare remind outputs for pypsa-coupled runs using the Remind-PyPSA-coupling package
"""

REMIND_REGION = config["run"].get("remind", {}).get("region")


rule build_run_config:
    """
    Build the run config for the pypsa-coupled run
    # TODO: output per remind run & snakemake run?
    Example:
        snakemake resources/derived_data/tmp/remind_coupled.yaml --cores 1
    """
    params:
        remind_region=REMIND_REGION,
        expname_max_len=20,
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
    conda:
        "remind-coupling"
    script:
        "../scripts/remind_coupling/generic_etl.py"


rule disaggregate_remind_data:
    """
    Disaggregate the data from the remind output to the network time and spatial resolutions
    """
    params:
        etl_cfg=config.get("remind_etl"),
        region=REMIND_REGION,  # overlaps with cofnig
        reference_load_year=2025,
        expand_dirs=config["scenario"]["planning_horizons"],
    input:
        **{
            f"pypsa_powerplants_{yr}": DERIVED_DATA
            + f"/existing_infrastructure/capacities_{yr}.csv"
            for yr in config["scenario"]["planning_horizons"]
        },
        remind_caps=DERIVED_DATA + "/remind/preinv_capacities.csv",
        remind_tech_groups=DERIVED_DATA + "/remind/tech_groups.csv",
        loads=DERIVED_DATA + "/remind/yrly_loads.csv",
        # todo switch to default?
        reference_load="resources/data/load/Provincial_Load_2020_2060_MWh.csv",
    output:
        **{
            f"caps_{yr}": DERIVED_DATA
            + f"/remind/harmonized_capacities/capacities_{yr}.csv"
            for yr in config["scenario"]["planning_horizons"]
        },
        paid_off=DERIVED_DATA + "/remind/harmonized_capacities/paid_off_capacities.csv",
        disagg_load=DERIVED_DATA + "/remind/ac_load_disagg.csv",
    conda:
        "remind-coupling"
    script:
        "../scripts/remind_coupling/disaggregate_data.py"
