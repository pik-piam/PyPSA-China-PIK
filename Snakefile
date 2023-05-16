# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

from os.path import normpath, exists
from shutil import copyfile, move

from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
HTTP = HTTPRemoteProvider()

configfile: "config_exponential.yaml" # select config file; grid expansion: "config.yaml", pathway: "config_linear.yaml", "config_exponential.yaml", "config_beta.yaml"

ATLITE_NPROCESSES = config['atlite'].get('nprocesses', 4)

if config["foresight"] == "non-pathway":
    rule prepare_all_networks:
        input:
            expand(
                config['results_dir'] + 'version-' + str(config['version']) + '/prenetworks/prenetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}.nc',
                **config["scenario"]
            )

    rule solve_all_networks:
        input:
            expand(
                config['results_dir'] + 'version-' + str(config['version']) + '/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}.nc',
                **config["scenario"]
            ),

    rule plot_all:
        input:
        #     expand(
        #         config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}_ext.pdf',
        #         **config["scenario"]
        #     ),
            expand(
                config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}_costs.png',
                **config["scenario"]
            ),

if config["foresight"] == "myopic":

    rule solve_all_networks:
        input:
            expand(
                config['results_dir'] + 'version-' + str(config['version']) + '/prenetworks-brownfield/prenetwork-{opts}-{topology}-{pathway}-{planning_horizons}.nc',
                **config["scenario"]
            ),

            expand(
                config['results_dir'] + 'version-' + str(config['version']) + '/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.nc',
                **config["scenario"]
            ),

    rule plot_summaries:
        input:
            expand(
                config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}_costs.png',
                **config["scenario"]
            )

    rule plot_all:
        input:
            expand(
                config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}_ext.pdf',
                **config["scenario"]
            ),
        #     expand(
        #         config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}_costs.png',
        #         **config["scenario"]
        #     ),


rule build_p_nom:
    output:
        coal_capacity="data/p_nom/coal_p_nom.h5",
        CHP_capacity="data/p_nom/CHP_p_nom.h5",
        OCGT_capacity="data/p_nom/OCGT_p_nom.h5",
        offwind_capacity="data/p_nom/offwind_p_nom.h5",
        onwind_capacity="data/p_nom/onwind_p_nom.h5",
        solar_capacity="data/p_nom/solar_p_nom.h5",
        nuclear_capacity="data/p_nom/nuclear_p_nom.h5"
    threads:1
    resources: mem_mb=500
    script: "scripts/build_p_nom.py"

rule build_population:
    output:
        population="data/population/population.h5"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/build_population.py"

if config['enable'].get('retrieve_cutout', True):
    rule retrieve_cutout:
        input: HTTP.remote("zenodo.org/record/6510859/files/China-2020.nc", keep_local=True, static=True)
        output: "cutouts/{cutout}.nc"
        run: move(input[0], output[0])


if config['enable'].get('build_cutout', False):
    rule build_cutout:
        input:
            regions_onshore="data/resources/regions_onshore.geojson",
            regions_offshore="data/resources/regions_offshore.geojson"
        output: "cutouts/{cutout}.nc"
        log: "logs/build_cutout/{cutout}.log"
        benchmark: "benchmarks/build_cutout_{cutout}"
        threads: ATLITE_NPROCESSES
        resources: mem_mb=ATLITE_NPROCESSES * 1000
        script: "scripts/build_cutout.py"

rule build_population_gridcell_map:
    input:
        infile="data/population/population.h5"
    output:
        population_map="data/population/population_gridcell_map.h5"
    threads: 1
    resources: mem_mb=35000
    script: "scripts/build_population_gridcell_map.py"

rule build_solar_thermal_profiles:
    input:
        infile="data/population/population_gridcell_map.h5"
    output:
        profile_solar_thermal = f"data/heating/solar_thermal-{config['solar_thermal_angle']}.h5"
    threads: 8
    resources: mem_mb=30000
    script: "scripts/build_solar_thermal_profiles.py"

rule build_heat_demand_profiles:
    input:
        infile="data/population/population_gridcell_map.h5",
        cutout="cutouts/China-2020.nc"
    output:
        daily_heat_demand="data/heating/daily_heat_demand.h5"
    threads: 8
    resources: mem_mb=30000
    script: "scripts/build_heat_demand_profiles.py"

rule build_cop_profiles:
    input:
        infile="data/population/population_gridcell_map.h5"
    output:
        cop="data/heating/cop.h5"
    threads: 8
    resources: mem_mb=30000
    script: "scripts/build_cop_profiles.py"

rule build_temp_profiles:
    input:
        infile="data/population/population_gridcell_map.h5"
    output:
        temp="data/heating/temp.h5"
    threads: 8
    resources: mem_mb=30000
    script: "scripts/build_temp_profiles.py"

rule build_energy_totals:
    input:
        infile1="data/population/population.h5",
        infile2="data/population/population_gridcell_map.h5"
    output:
        outfile1="data/energy_totals_{planning_horizons}.h5"
    threads: 1
    resources: mem_mb=10000
    script: "scripts/build_energy_totals.py"

if config['enable'].get('retrieve_raster', True):
    rule retrieve_build_up_raster:
        input: HTTP.remote("zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_BuiltUp-CoverFraction-layer_EPSG-4326.tif", keep_local=True, static=True)
        output: "data/resources/Build_up.tif"
        run: move(input[0], output[0])
    rule retrieve_Grass_raster:
        input: HTTP.remote("zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Grass-CoverFraction-layer_EPSG-4326.tif", keep_local=True, static=True)
        output: "data/resources/Grass.tif"
        run: move(input[0], output[0])
    rule retrieve_Bare_raster:
        input: HTTP.remote("zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Bare-CoverFraction-layer_EPSG-4326.tif", keep_local=True, static=True)
        output: "data/resources/Bare.tif"
        run: move(input[0], output[0])
    rule retrieve_Shrubland_raster:
        input: HTTP.remote("zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Shrub-CoverFraction-layer_EPSG-4326.tif", keep_local=True, static=True)
        output: "data/resources/Shrubland.tif"
        run: move(input[0], output[0])

rule build_renewable_potential:
    input:
        Build_up_raster="data/landuse_availability/Build_up.tif",
        Grass_raster="data/landuse_availability/Grass.tif",
        Bare_raster="data/landuse_availability/Bare.tif",
        Shrubland_raster="data/landuse_availability/Shrubland.tif",
        natura1='data/landuse_availability/WDPA_WDOECM_Mar2022_Public_CHN_shp/WDPA_WDOECM_Mar2022_Public_CHN_shp_0/WDPA_WDOECM_Mar2022_Public_CHN_shp-polygons.shp',
        natura2='data/landuse_availability/WDPA_WDOECM_Mar2022_Public_CHN_shp/WDPA_WDOECM_Mar2022_Public_CHN_shp_1/WDPA_WDOECM_Mar2022_Public_CHN_shp-polygons.shp',
        natura3='data/landuse_availability/WDPA_WDOECM_Mar2022_Public_CHN_shp/WDPA_WDOECM_Mar2022_Public_CHN_shp_2/WDPA_WDOECM_Mar2022_Public_CHN_shp-polygons.shp',
        gebco="data/landuse_availability/GEBCO_tiff/gebco_2021.tif",
        provinces_shp="data/province_shapes/CHN_adm1.shp",
        offshore_province_shapes="data/resources/regions_offshore_province.geojson",
        offshore_shapes="data/resources/regions_offshore.geojson",
        cutout= "cutouts/China-2020.nc"
    output:
        solar_profile="resources/profile_solar.nc",
        onwind_profile="resources/profile_onwind.nc",
        offwind_profile="resources/profile_offwind.nc"
    log: "logs/build_renewable_potential.log"
    threads: ATLITE_NPROCESSES
    resources: mem_mb=ATLITE_NPROCESSES * 5000
    script: "scripts/build_renewable_potential.py"

if config["foresight"] == "non-pathway":
    rule prepare_networks:
        input:
            population_name="data/population/population.h5",
            solar_thermal_name="data/heating/solar_thermal-{angle}.h5".format(angle=config['solar_thermal_angle']),
            heat_demand_name="data/heating/daily_heat_demand.h5",
            cop_name="data/heating/cop.h5",
            energy_totals_name="data/energy_totals_{planning_horizons}.h5",
            co2_totals_name="data/co2_totals.h5",
            temp="data/heating/temp.h5",
            tech_costs = "data/costs_{planning_horizons}.csv",
            **{f"profile_{tech}": f"resources/profile_{tech}.nc"
               for tech in config['renewable']}
        output:
            network_name=config['results_dir'] + 'version-' + str(config['version']) + '/prenetworks/prenetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}.nc',
        threads: 1
        resources: mem_mb=10000
        script: "scripts/prepare_network.py"

    rule solve_networks:
        input:
            network_name=config['results_dir'] + 'version-' + str(config['version']) + '/prenetworks/prenetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}.nc',
        output:
            network_name=config['results_dir'] + 'version-' + str(config['version']) + '/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}.nc'
        log:
            solver=normpath("logs/solve_operations_network/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}.log")
        threads: 4
        resources: mem_mb=35000
        script: "scripts/solve_network.py"
    #
    # rule plot_network:
    #     input:
    #         network=config['results_dir'] + 'version-' + str(config['version']) + '/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}.nc',
    #         tech_costs="data/costs_{planning_horizons}.csv"
    #     output:
    #         only_map=config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}.pdf',
    #         cost_map=config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}-cost.pdf',
    #         ext=config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}_ext.pdf'
    #     log: "logs/plot_network/postnetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}.log"
    #     script: "scripts/plot_network.py"

    # rule make_summary:
    #     input:
    #         network=config['results_dir'] + 'version-' + str(config['version']) + '/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}.nc',
    #         tech_costs="data/costs_{planning_horizons}.csv",
    #     output:
    #         directory(config['results_dir'] + 'version-' + str(config['version']) + '/summary/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}'),
    #     log: "logs/make_summary/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}.log"
    #     resources: mem_mb=5000
    #     script: "scripts/make_summary.py"
    #
    # rule plot_summary:
    #     input:
    #         config['results_dir'] + 'version-' + str(config['version']) + '/summary/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}'
    #     output:
    #         energy = config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}_energy.png',
    #         cost = config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}_costs.png'
    #     log: "logs/plot/summary/postnetwork-{opts}-{topology}-{pathway}-{co2_reduction}-{planning_horizons}.log"
    #     script: "scripts/plot_summary.py"

if config["foresight"] == "myopic":
    rule prepare_base_networks:
        input:
            population_name="data/population/population.h5",
            solar_thermal_name="data/heating/solar_thermal-{angle}.h5".format(angle=config['solar_thermal_angle']),
            heat_demand_name="data/heating/daily_heat_demand.h5",
            cop_name="data/heating/cop.h5",
            energy_totals_name="data/energy_totals_{planning_horizons}.h5",
            co2_totals_name="data/co2_totals.h5",
            temp="data/heating/temp.h5",
            tech_costs= "data/costs_{planning_horizons}.csv",
            **{f"profile_{tech}": f"resources/profile_{tech}.nc"
               for tech in config['renewable']}
        output:
            network_name=config['results_dir'] + 'version-' + str(config['version']) + '/prenetworks/prenetwork-{opts}-{topology}-{pathway}-{planning_horizons}.nc',
        threads: 1
        resources: mem_mb=10000
        script: "scripts/prepare_base_network.py"

    rule add_existing_baseyear:
        input:
            overrides="data/override_component_attrs",
            network=config['results_dir'] + 'version-' + str(config['version']) + '/prenetworks/prenetwork-{opts}-{topology}-{pathway}-{planning_horizons}.nc',
            tech_costs="data/costs_{planning_horizons}.csv",
            **{f"existing_{tech}": f"data/existing_infrastructure/{tech} capacity.csv"
               for tech in config['existing_infrastructure']},
        output: config['results_dir'] + 'version-' + str(config['version']) + '/prenetworks-brownfield/prenetwork-{opts}-{topology}-{pathway}-{planning_horizons}.nc'
        wildcard_constraints:
            planning_horizons=config['scenario']['planning_horizons'][0] #only applies to baseyear
        threads: 1
        resources: mem_mb=2000
        script: "scripts/add_existing_baseyear.py"

    def solved_previous_horizon(wildcards):
        planning_horizons = config["scenario"]["planning_horizons"]
        i = planning_horizons.index(int(wildcards.planning_horizons))
        planning_horizon_p = str(planning_horizons[i-1])
        return config['results_dir'] + 'version-' + str(config['version']) + "/postnetworks/postnetwork-{opts}-{topology}-{pathway}-" + planning_horizon_p + ".nc"

    rule add_brownfield:
        input:
            overrides="data/override_component_attrs",
            network=config['results_dir'] + 'version-' + str(config['version']) + '/prenetworks/prenetwork-{opts}-{topology}-{pathway}-{planning_horizons}.nc',
            network_p=solved_previous_horizon,#solved network at previous time step
            costs="data/costs_{planning_horizons}.csv",
            **{f"profile_{tech}": f"resources/profile_{tech}.nc"
                for tech in config['renewable']}
        output:
            network_name = config['results_dir'] + 'version-' + str(config['version']) + '/prenetworks-brownfield/prenetwork-{opts}-{topology}-{pathway}-{planning_horizons}.nc',
        threads: 4
        resources: mem_mb=10000
        script: "scripts/add_brownfield.py"

    ruleorder: add_existing_baseyear > add_brownfield

    rule solve_network_myopic:
        input:
            overrides = "data/override_component_attrs",
            network=config['results_dir'] + 'version-' + str(config['version']) + '/prenetworks-brownfield/prenetwork-{opts}-{topology}-{pathway}-{planning_horizons}.nc',
            costs="data/costs_{planning_horizons}.csv",
        output:
            network_name = config['results_dir'] + 'version-' + str(config['version']) + '/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.nc'
        log:
            solver = normpath("logs/solve_operations_network/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.log")
        threads: 4
        resources: mem_mb = 35000
        script: "scripts/solve_network_myopic.py"

    rule plot_network:
        input:
            network=config['results_dir'] + 'version-' + str(config['version']) + '/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.nc',
            tech_costs="data/costs_{planning_horizons}.csv"
        output:
            only_map=config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.pdf',
            cost_map=config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}-cost.pdf',
            ext=config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}_ext.pdf',
        log: "logs/plot_network/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.log"
        script: "scripts/plot_network.py"

    rule plot_network_heat:
        input:
             network=config['results_dir'] + 'version-' + str(config['version']) + '/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.nc',
             tech_costs="data/costs_{planning_horizons}.csv"
        output:
            only_map=config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}_heat.pdf',
            ext=config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}_ext_heat.pdf',
        log: "logs/plot_network/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}_heat.log"
        script: "scripts/plot_network_heat.py"

    rule make_summary:
        input:
            network=config['results_dir'] + 'version-' + str(config['version']) + '/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.nc',
            tech_costs="data/costs_{planning_horizons}.csv",
        output:
            directory(config['results_dir'] + 'version-' + str(config['version']) + '/summary/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}'),
        log: "logs/make_summary/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.log"
        resources: mem_mb=5000
        script: "scripts/make_summary.py"

    rule plot_summary:
        input:
            config['results_dir'] + 'version-' + str(config['version']) + '/summary/postnetworks/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}'
        output:
            energy = config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.png',
            cost = config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}_costs.png'
        log: "logs/plot/summary/postnetwork-{opts}-{topology}-{pathway}-{planning_horizons}.log"
        script: "scripts/plot_summary.py"




