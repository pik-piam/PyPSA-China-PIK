# snakemake rules for data fetch operations

rule fetch_region_shapes:
    output: 
        country_shape=DERIVED_COMMON + "/regions/country.geojson",
        province_shapes=DERIVED_COMMON + "/regions/provinces_onshore.geojson",
        offshore_shapes=DERIVED_COMMON + "/regions/provinves_offshore.geojson",
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

if config['enable'].get('retrieve_cutout', True):
    rule retrieve_cutout:
        input: storage.http("https://zenodo.org/record/8343761/files/China-2020.nc")
        output: "resources/cutouts/{cutout}.nc"
        run: shutil.move(input[0], output[0])