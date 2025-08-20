# snakemake rules for data fetch operations

# TODO rework this, save shapes with all admin levels
# build nodes with another script and save that to DERIVED_DATA
# nodes could be read by snakefile and passed as a param to the relevant rules
rule fetch_region_shapes:
    output: 
        country_shape=DERIVED_COMMON + "/regions/country.geojson",
        province_shapes=DERIVED_COMMON + "/regions/provinces_onshore.geojson",
        offshore_shapes=DERIVED_COMMON + "/regions/provinces_offshore.geojson",
        prov_shpfile=DERIVED_COMMON + "/province_shapes/CHN_adm1.shp"
    log: LOGS_COMMON+"/fetch_regions_shapes.log"
    script: "../scripts/fetch_shapes.py"


# TODO build actual fetch rules with the sentinel/copernicus APIs. 
# TODO See if there are datasets succeeding the S2 LC100 cover to get newer data
if config['enable'].get('retrieve_raster', True):
    rule retrieve_build_up_raster:
        input: storage.http("https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_BuiltUp-CoverFraction-layer_EPSG-4326.tif")
        output: "resources/data/regions/Build_up.tif"
        run: shutil.move(input[0], output[0])
    rule retrieve_Grass_raster:
        input: storage.http("https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Grass-CoverFraction-layer_EPSG-4326.tif")
        output: "resources/data/regions/Grass.tif"
        run: shutil.move(input[0], output[0])
    rule retrieve_Bare_raster:
        input: storage.http("https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Bare-CoverFraction-layer_EPSG-4326.tif")
        output: "resources/data/regions/Bare.tif"
        run: shutil.move(input[0], output[0])
    rule retrieve_Shrubland_raster:
        input: storage.http("https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Shrub-CoverFraction-layer_EPSG-4326.tif")
        output: "resources/data/regions/Shrubland.tif"
        run: shutil.move(input[0], output[0])

if config["enable"].get("retrieve_vector", False) and config['enable'].get('build_cutout', False):
    raise ValueError("Settings error: you must choose between retrieving a pre-built cutout or building one from scratch")
elif config['enable'].get('retrieve_cutout', False):
    rule retrieve_cutout:
        input: storage.http("https://zenodo.org/record/8343761/files/China-2020.nc")
        output: "resources/cutouts/China2020-Xiaowei.nc"
        run: shutil.move(input[0], output[0])