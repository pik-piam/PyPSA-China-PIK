"""
Prepare remind outputs for pypsa-coupled runs using the Remind-PyPSA-coupling package
"""

REMIND_REGION = config["run"]["remind"]["region"]


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
        remind_output=config["paths"]["remind_outpt_dir"],
        config_template="config/templates/remind_cpled.yml",
    output:
        coupled_config=DERIVED_COMMON + "/tmp/remind_coupled.yaml",
    log:
        LOGS_COMMON + "/remind_coupling/build_run_cfg.log",
    conda:
        "../envs/remind.yaml"
    script:
        "../scripts/remind_coupling/make_pypsa_config.py"


# TODO how to pass config?
rule transform_remind_data:
    """
    Import the remind data from the remind output & transform it to the pypsa-china format using 
    """
    params:
        etl_cfg=config.get("remind_etl"),
        region=REMIND_REGION,  # overlaps with cofnig
        use_gdx=False,
    input:
        pypsa_costs="resources/data/costs",
        remind_output_dir=os.path.expanduser(
            "~/downloads/output_REMIND/SSP2-Budg1000-PyPSAxprt_2025-05-09/pypsa_export"
        ),
    output:
        loads=DERIVED_COMMON + "/remind/yrly_loads.csv",
        technoeconomic_data=DERIVED_COMMON + "/remind/costs/",
        remind_caps=DERIVED_COMMON + "/remind/remind_capacities.csv",
    conda:
        "../envs/remind.yaml"
    script:
        "scripts/remind_coupling/etl_remind.py"


rule disaggregate_data:
    """
    Disaggregate the data from the remind output to the network time and spatial resolutions
    """
    params:
        etl_cfg=config.get("remind_etl"),
        region=REMIND_REGION,  # overlaps with cofnig
        reference_load_year=2025,
    input:
        pypsa_powerplants="",
        remind_outputs="",
        loads=DERIVED_COMMON + "/remind/yrly_loads.csv",
        reference_load="resources/data/load/Provincial_Load_2020_2060_MWh.csv",
    output:
        disagg_load=DERIVED_COMMON + "/remind/ac_load_disagg.csv",
    conda:
        "../envs/remind.yaml"
    script:
        "scripts/remind_coupling/transform_loads.py"
