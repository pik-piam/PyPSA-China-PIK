rule plot_network:
    input:
        network=config['results_dir'] + 'version-' + str(
            config['version']) + '/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.nc',
    output:
        cost_map=config['results_dir'] + 'version-' + str(
            config['version']) + '/plots/network/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}-cost.pdf',
    log: "logs/plot_network/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.log"
    script: "../scripts/plot_network.py"

rule make_summary:
    input:
        network=config['results_dir'] + 'version-' + str(
            config['version']) + '/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.nc',
    output:
        directory(config['results_dir'] + 'version-' + str(
            config['version']) + '/summary/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}'),
    log: "logs/make_summary/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.log"
    resources: mem_mb=5000
    script: "../scripts/make_summary.py"

rule plot_summary:
    input:
        config['results_dir'] + 'version-' + str(
            config['version']) + '/summary/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}'
    output:
        energy=config['results_dir'] + 'version-' + str(
            config['version']) + '/plots/summary/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.png',
        cost=config['results_dir'] + 'version-' + str(
            config['version']) + '/plots/summary/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}_costs.png'
    log: "logs/plot/summary/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.log"
    script: "../scripts/plot_summary.py"

rule plot_heatmap:
    input:
        network = config['results_dir'] + 'version-' + str(config['version']) + '/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.nc',
    output:
        water = config['results_dir'] + 'version-' + str(config['version']) + '/plots/heatmap/water_tank/water_tank-{opts}-{topology}-{pathway}-{planning_horizons}.png',
        water_store = config['results_dir'] + 'version-' + str(config['version']) + '/plots/heatmap/water_tank/water_store-{opts}-{topology}-{pathway}-{planning_horizons}.png',
        battery = config['results_dir'] + 'version-' + str(config['version']) + '/plots/heatmap/battery/battery-{opts}-{topology}-{pathway}-{planning_horizons}.png',
        H2 = config['results_dir'] + 'version-' + str(config['version']) + '/plots/heatmap/H2/H2-{opts}-{topology}-{pathway}-{planning_horizons}.png',
    script:  "../scripts/plot_heatmap.py"