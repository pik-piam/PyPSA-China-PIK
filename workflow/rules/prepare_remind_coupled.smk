"""
Prepare remind outputs for pypsa-coupled runs using the Remind-PyPSA-coupling package
"""


rule build_run_config:
    """
    Build the run config for the pypsa-coupled run
    """
    params:
        remind_region="CHA",
    input:
        remind_output="",
        config_template="",
    output:
        coupled_config="",
    conda:
        "../envs/remind.yaml"
    script:
        "scripts/remind_coupling/make_pypsa_config.py"


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
