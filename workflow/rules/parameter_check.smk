# snakemake rules for datainput check

from os.path import join

scenario_ph = {"planning_horizons": config["scenario"]["planning_horizons"]}
scenario_wildcards = {k: v for k, v in config["scenario"].items() if k != "planning_horizons"}


if config["foresight"] in ["None", "overnight", "non-pathway", "myopic"]:
    rule plot_parameters:
        input:
            costs = expand("resources/data/costs/costs_{planning_horizons}.csv", **scenario_ph)
        output:
            cost_map = expand(RESULTS_DIR + "/plots/costs/parameters_comparison.pdf", **scenario_wildcards)
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
