# snakemake rules for datainput check

from os.path import join

scenario_ph = {"planning_horizons": config["scenario"]["planning_horizons"]}
scenario_wildcards = {k: v for k, v in config["scenario"].items() if k != "planning_horizons"}


if config["foresight"] in ["None", "overnight", "non-pathway", "myopic"]:
    rule plot_parameters:
        input:
            # 使用pre_networks中的成本数据
            costs = expand(RESULTS_DIR + "/prenetworks/ntwk_{planning_horizons}.nc", **scenario_ph),
            config = "config/plot_config.yaml",
            reference_costs = lambda wildcards: "resources/data/costs/reference_values/tech_costs_subset_litreview.csv" if config["plotting"].get("plot_reference", True) else None
        output:
            cost_map = expand(RESULTS_DIR + "/plots/costs/parameters_comparison.pdf", **scenario_wildcards)
        params:
            plot_reference = lambda wildcards: config["plotting"].get("plot_reference", True)
        log:
            expand(LOG_DIR + "/plot_costs/parameters_comparison.log", **scenario_wildcards)
        resources:
            mem_mb = 2000
        script:
            "../scripts/plot_parameters.py"
else:
    raise NotImplementedError(
        f"Plotting fororesight {config['foresight']} not implemented"
    )
