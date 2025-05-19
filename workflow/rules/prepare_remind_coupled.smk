"""
Prepare remind outputs for pypsa-coupled runs using the Remind-PyPSA-coupling package
"""


rule build_run_config:
    """
    Build the run config for the pypsa-coupled run
    """
    params:
        remind_region="CHA",
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


rule transform_load:
    """
    Transform the load data from the remind output to the pypsa-china format
    """
    input:
        remind_output_dir="",
    output:
        ac_load="",
    conda:
        "../envs/remind.yaml"
    script:
        "scripts/remind_coupling/transform_loads.py"


rule transform_technoeconomic:
    """
    Transform the technoeconomic data from the remind output to make the costs data
    """
    input:
        remind_output_dir="",
    output:
        ac_load="",
    conda:
        "../envs/remind.yaml"
    script:
        "scripts/remind_coupling/transform_loads.py"


rule transform_capacities:
    """
    Transform the pre-investment capacities data for pypsa
    """
    input:
        pypsa_powerplants="",
        remind_capacities="",
    output:
        existing_baseyear_caps="",
        paid_off_caps="",
    conda:
        "../envs/remind.yaml"
    script:
        "scripts/remind_coupling/transform_loads.py"
