# Snakefile rules for postprocessing (plotting etc)

from os.path import join

STATISTICS_BARPLOTS = [
    "capacity_factor",
    "installed_capacity",
    "optimal_capacity",
    "capital_expenditure",
    "operational_expenditure",
    "curtailment",
    "supply",
    "withdrawal",
    "market_value",
    "lcoe",
    "province_peakload_capacity",
    "mv_minus_lcoe",
]


if config["foresight"] in ["None", "overnight", "non-pathway", "myopic"]:

    rule plot_network:
        input:
            network=join(
                RESULTS_DIR,
                "postnetworks/ntwk_{planning_horizons}.nc",
            ),
            tech_costs=COSTS_DATA + "/costs_{planning_horizons}.csv",
            province_shape=DERIVED_COMMON + "/province_shapes/CHN_adm1.shp",
        output:
            cost_map=RESULTS_DIR + "/plots/networks/ntwk_{planning_horizons}-cost.png",
            el_supply_map=RESULTS_DIR
            + "/plots/networks/ntwk_{planning_horizons}-el_supply.png",
        log:
            LOG_DIR + "/plot_network/ntwk_{planning_horizons}.log",
        script:
            "../scripts/plot_network.py"

    rule make_summary:
        input:
            network=join(
                RESULTS_DIR,
                "postnetworks/ntwk_{planning_horizons}.nc",
            ),
            tech_costs=COSTS_DATA + "/costs_{planning_horizons}.csv",
        output:
            directory(RESULTS_DIR + "/summary/ntwk_{planning_horizons}"),
        log:
            LOG_DIR + "/make_summary_ntwk_{planning_horizons}.log",
        resources:
            mem_mb=5000,
        script:
            "../scripts/make_summary.py"

    rule plot_summary:
        input:
            expand(
                RESULTS_DIR + "/summary/ntwk_{planning_horizons}",
                **{
                    k: v
                    for k, v in config["scenario"].items()
                    if k != "planing_horizons"
                },
            ),
        output:
            energy=RESULTS_DIR + "/plots/summary/elec_balance.png",
            costs=RESULTS_DIR + "/plots/summary/pathway_costs.png",
        log:
            LOG_DIR + "/plot/summary_plot_ntwk_summary.log",
        script:
            "../scripts/plot_summary_all.py"

    rule plot_statistics:
        input:
            network=join(
                RESULTS_DIR,
                "postnetworks/ntwk_{planning_horizons}.nc",
            ),
        params:
            stat_types=STATISTICS_BARPLOTS,
            carrier="AC",
        output:
            stats_dir=directory(RESULTS_DIR + "/plots/statistics_{planning_horizons}"),
        log:
            LOG_DIR + "/plot_statistics_ntwk_{planning_horizons}.log",
        script:
            "../scripts/plot_statistics.py"

    # TODO auto search expesnive weeks like in notebooks
    rule plot_snapshots:
        input:
            network=join(
                RESULTS_DIR,
                "postnetworks/ntwk_{planning_horizons}.nc",
            ),
        params:
            winter_day1="12-10 21:00",  # mm-dd HH:MM 
            winter_day2="12-17 12:00",  # mm-dd HH:MM
            spring_day1="04-01 21:00",  # mm-dd HH:MM
            spring_day2="04-07 12:00",  # mm-dd HH:MM
            summer_day1="07-15 21:00",  # mm-dd HH:MM
            summer_day2="07-22 12:00",  # mm-dd HH:MM
        output:
            outp_dir=directory(RESULTS_DIR + "/plots/snapshots_{planning_horizons}"),
        log:
            LOG_DIR + "/plot_snapshots_ntwk_{planning_horizons}.log",
        script:
            "../scripts/plot_time_series.py"

else:
    raise NotImplementedError(
        f"Plotting fororesight {config["foresight"]} not implemented"
    )


# TODO fix
rule plot_heatmap:
    input:
        network=RESULTS_DIR + "/postnetworks/ntwk_{planning_horizons}.nc",
    output:
        water=RESULTS_DIR
        + "/plots/heatmap/water_tank/water_tank-{planning_horizons}.png",
        water_store=RESULTS_DIR
        + "/plots/heatmap/water_tank/water_store-{planning_horizons}.png",
        battery=RESULTS_DIR + "/plots/heatmap/battery/battery-{planning_horizons}.png",
        H2=RESULTS_DIR + "/plots/heatmap/H2/H2-{planning_horizons}.png",
    script:
        "../scripts/plot_heatmap.py"
