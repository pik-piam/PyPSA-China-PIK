# TAKE CARE OF PLOTTING


base_results_dir = config["base_results_dir"]

if config["foresight"] == "steady-state":

    rule plot_steady_state:
        input:
            expand(
                base_results_dir
                + "/plots/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}_ext.pdf",
                **config["scenario"],
            ),
            expand(
                base_results_dir
                + "/plots/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}_costs.png",
                **config["scenario"],
            ),


# TODO fix comments: there shouldnt be commented out code on a live version
elif config["foresight"] == "myopic":

    rule plot_myopic:
        input:
            expand(
                base_results_dir
                + "/plots/summary/{heating_demand}/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}_costs.png",
                **config["scenario"],
            ),
            # expand(
            #     base_results_dir + '/plots/heatmap/water_tank/water_tank-{opts}-{topology}-{pathway}-{planning_horizons}.png',
            #     ** config["scenario"]
            # ),
            # expand(
            #     base_results_dir + '/plots/heatmap/water_tank/water_store-{opts}-{topology}-{pathway}-{planning_horizons}.png',
            #     ** config["scenario"]
            # ),
            # expand(
            #     base_results_dir + '/plots/network/{heating_demand}/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}-cost.pdf',
            #     **config["scenario"]
            # ),
            # expand(
            #     base_results_dir + '/plots/network/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}_ext_heat.pdf',
            #     **config["scenario"]
            # )


# TODO fix paths
rule plot_network:
    input:
        network=base_results_dir
        + "/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.nc",
        tech_costs="resources/data/costs_{planning_horizons}.csv",
    output:
        only_map=base_results_dir
        + "/plots/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.pdf",
        cost_map=base_results_dir
        + "/plots/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}-cost.pdf",
        ext=base_results_dir
        + "/plots/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}_ext.pdf",
    log:
        "logs/plot_network/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.log",
    script:
        "../scripts/plot_network.py"


rule make_summary:
    input:
        network=base_results_dir
        + "/postnetworks/{heating_demand}/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.nc",
        tech_costs="resources/data/costs/costs_{planning_horizons}.csv",
    output:
        directory(
            base_results_dir
            + "/summary/postnetworks/{heating_demand}/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}"
        ),
    log:
        "logs/make_summary_postnetworks_{heating_demand}/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.log",
    resources:
        mem_mb=5000,
    script:
        "../scripts/make_summary.py"


rule plot_summary:
    input:
        base_results_dir
        + "/summary/postnetworks/{heating_demand}/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}",
    output:
        energy=base_results_dir
        + "/plots/summary/{heating_demand}/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}_energy.png",
        costs=base_results_dir
        + "/plots/summary/{heating_demand}/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}_costs.png",
    log:
        "logs/plot/{heating_demand}_summary_plot_postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.log",
    script:
        "../scripts/plot_summary.py"


# TODO add to config options
# rule plot_heatmap:
#     input:
#         network = base_results_dir + '/postnetworks/{heating_demand}/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.nc',
#     output:
#         water = base_results_dir + '/plots/heatmap/{heating_demand}/water_tank/water_tank-{opts}-{topology}-{pathway}-{planning_horizons}.png',
#         water_store = base_results_dir + '/plots/heatmap/{heating_demand}/water_tank/water_store-{opts}-{topology}-{pathway}-{planning_horizons}.png',
#         battery = base_results_dir + '/plots/heatmap/{heating_demand}/battery/battery-{opts}-{topology}-{pathway}-{planning_horizons}.png',
#         H2 = base_results_dir + '/plots/heatmap/{heating_demand}/H2/H2-{opts}-{topology}-{pathway}-{planning_horizons}.png',
#     script:  "../scripts/plot_heatmap.py"
