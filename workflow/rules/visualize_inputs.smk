# snakemake rules for plot_cost_data

from os.path import join

scenario_ph = {"planning_horizons": config["scenario"]["planning_horizons"]}
scenario_wildcards = {k: v for k, v in config["scenario"].items() if k != "planning_horizons"}

COSTS_DATA = path_manager.costs_dir()

if config["foresight"] in ["None", "overnight", "non-pathway", "myopic"]:
    rule plot_parameters:
        input:
            costs = lambda wildcards: join(COSTS_DATA, f"costs_{wildcards.planning_horizons}.csv"),
            config = "config/plot_config.yaml",
            reference_costs = lambda wildcards: "resources/data/costs/reference_values/tech_costs_subset_litreview.csv" if config["plotting"]["visualize_inputs"]["reference_costs"] else None
        output:
            cost_map = expand(RESULTS_DIR + "/plots/costs/parameters_comparison.pdf", **scenario_wildcards)
        params:
            plot_reference = lambda wildcards: config["plotting"]["visualize_inputs"]["reference_costs"]
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
rule plot_renewable_classes:
    input:
        province_shape=DERIVED_COMMON + "/province_shapes/CHN_adm1.shp",
        renewable_classes=DERIVED_CUTOUT
        + "/"
        + "{technology}_regions_by_class_{rc_params}.geojson",
        average_distance=DERIVED_CUTOUT
        + "/"
        + "average_distance_{technology}-{rc_params}.h5",
    output:
        renewable_grades_bins=DERIVED_CUTOUT + "/" + "{technology}_{rc_params}_bins.png",
        renewable_grades_cf=DERIVED_CUTOUT + "/" + "{technology}_{rc_params}_cfs.png",
        distances_hist=DERIVED_CUTOUT
        + "/"
        + "{technology}_{rc_params}_avg_distances.png",
    script:
        "../scripts/plot_inputs_visualisation.py"
