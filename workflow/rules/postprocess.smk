# TAKE CARE OF PLOTTING
from os.path import join

if config["foresight"] == ["None", "overnight", "non-pathway"]:

    rule plot_overnight:
        input:
            expand(
                RESULTS_DIR + "/plots/ntwk_{planning_horizons}_costs.png",
                **config["scenario"],
            ),

    rule plot_network:
        input:
            network=join(
                RESULTS_DIR,
                "postnetworks/ntwk_{planning_horizons}.nc",
            ),
            tech_costs="resources/data/costs/costs_{planning_horizons}.csv",
            province_shape="resources/data/province_shapes/CHN_adm1.shp",
        output:
            cost_map=RESULTS_DIR + "/plots/networks/ntwk_{planning_horizons}-cost.pdf",
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
            tech_costs="resources/data/costs/costs_{planning_horizons}.csv",
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
            energy=RESULTS_DIR + "/plots/summary/ntwk_pathway_energy.png",
            costs=RESULTS_DIR + "/plots/summary/ntwk_pathway_costs.png",
        log:
            LOG_DIR + "/plot/summary_plot_ntwk_summary.log",
        script:
            "../scripts/plot_summary_all.py"

elif config["foresight"] == "myopic":

    rule plot_myopic:
        input:
            expand(
                RESULTS_DIR + "/plots/summary/ntwk_{planning_horizons}_costs.png",
                **config["scenario"],
            ),
            expand(
                RESULTS_DIR + "/plots/networks/ntwk_{planning_horizons}-cost.pdf",
                **config["scenario"],
            ),
            expand(
                RESULTS_DIR + "/plots/networks/ntwk_{planning_horizons}_ext_heat.pdf",
                **config["scenario"],
            ),

    rule plot_network:
        input:
            network=RESULTS_DIR + "/postnetworks/ntwk_{planning_horizons}.nc",
            tech_costs="resources/data/costs/costs_{planning_horizons}.csv",
            province_shape="resources/data/province_shapes/CHN_adm1.shp",
        output:
            cost_map=RESULTS_DIR + "/plots/networks/ntwk_{planning_horizons}-cost.pdf",
            el_suppy_map=RESULTS_DIR
            + "/plots/networks/ntwk_{planning_horizons}_el_supply.pdf",
        log:
            LOG_DIR + "/plot_network/network/ntwk_{planning_horizons}.log",
        script:
            "../scripts/plot_network.py"

    rule make_summary:
        input:
            network=RESULTS_DIR + "/postnetworks/ntwk_{planning_horizons}.nc",
            tech_costs="resources/data/costs/costs_{planning_horizons}.csv",
        output:
            directory(RESULTS_DIR + "/summary/postnetworks/ntwk_{planning_horizons}"),
        log:
            LOG_DIR + "/make_summary_postnetworks/ntwk_{planning_horizons}.log",
        resources:
            mem_mb=5000,
        script:
            "../scripts/make_summary.py"

    rule plot_summary:
        input:
            expand(
                RESULTS_DIR + "/summary/postnetworks/ntwk_{planning_horizons}",
                **{
                    k: v
                    for k, v in config["scenario"].items()
                    if k != "planing_horizons"
                },
            ),
        output:
            energy=RESULTS_DIR + "/plots/summary/pathway_energy.png",
            costs=RESULTS_DIR + "/plots/summary/pathway_costs.png",
        log:
            LOG_DIR + "/plot_summary_plot_ntwk_summary.log",
        script:
            "../scripts/plot_summary_all.py"


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
