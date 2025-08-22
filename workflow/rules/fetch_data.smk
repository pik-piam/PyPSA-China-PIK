# snakemake rules for data fetch operations
from zipfile import ZipFile
import shutil
import os


# TODO rework this, save shapes with all admin levels
# build nodes with another script and save that to DERIVED_DATA
# nodes could be read by snakefile and passed as a param to the relevant rules
rule fetch_region_shapes:
    output:
        country_shape=DERIVED_COMMON + "/regions/country.geojson",
        province_shapes=DERIVED_COMMON + "/regions/provinces_onshore.geojson",
        offshore_shapes=DERIVED_COMMON + "/regions/provinces_offshore.geojson",
        prov_shpfile=DERIVED_COMMON + "/province_shapes/CHN_adm1.shp",
    log:
        LOGS_COMMON + "/fetch_regions_shapes.log",
    script:
        "../scripts/fetch_shapes.py"


# TODO build actual fetch rules with the sentinel/copernicus APIs.
# TODO See if there are datasets succeeding the S2 LC100 cover to get newer data
if config["enable"].get("retrieve_raster", True):

    # rule retrieve_rasters:
    #     input:
    #         grass=storage.http(
    #             "https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Grass-CoverFraction-layer_EPSG-4326.tif"
    #         ),
    #         shrubland=storage.http(
    #             "https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Shrub-CoverFraction-layer_EPSG-4326.tif"
    #         ),
    #         bare=storage.http(
    #             "https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Bare-CoverFraction-layer_EPSG-4326.tif"
    #         ),
    #         builtup=storage.http(
    #             "https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_BuiltUp-CoverFraction-layer_EPSG-4326.tif"
    #         ),
    #     output:
    #         builtup="resources/data/landuse_availability/Build_up.tif",
    #         bare="resources/data/landuse_availability/Bare.tif",
    #         grass="resources/data/landuse_availability/Grass.tif",
    #         shrubland="resources/data/landuse_availability/Shrubland.tif",
    #     run:
    #         for name, p in output.items():
    #             shutil.move(input[name], p)

    rule retrieve_Grass_raster:
        input: storage.http("https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Grass-CoverFraction-layer_EPSG-4326.tif")
        output: "resources/data/landuse_availability/Grass.tif"
        run: shutil.move(input[0], output[0])
    rule retrieve_Bare_raster:
        input: storage.http("https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Bare-CoverFraction-layer_EPSG-4326.tif")
        output: "resources/data/landuse_availability/Bare.tif"
        run: shutil.move(input[0], output[0])
    rule retrieve_Shrubland_raster:
        input: storage.http("https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Shrub-CoverFraction-layer_EPSG-4326.tif")
        output: "resources/data/landuse_availability/Shrubland.tif"
        run: shutil.move(input[0], output[0])
    rule retrieve_Build_up_raster:
        input: storage.http("https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_BuiltUp-CoverFraction-layer_EPSG-4326.tif")
        output: "resources/data/landuse_availability/Build_up.tif"
        run: shutil.move(input[0], output[0])

    rule retrieve_bathymetry_raster:
        input:
            gebco=storage.http(
                "https://zenodo.org/record/16792792/files/GEBCO_tiff.zip"
            ),
        output:
            gebco="resources/data/landuse_availability/GEBCO_tiff/gebco_2024_CN.tif",
        params:
            zip_file="resources/data/landuse_availability/GEBCO_tiff.zip",
        run:
            os.rename(input.gebco, params.zip_file)
            with ZipFile(params.zip_file, "r") as zip_ref:
                zip_ref.extractall(os.path.dirname(params.zip_file))
            os.remove(params.zip_file)

rule retrieve_powerplants:
    input:
        powerplants=storage.http(
            "https://zenodo.org/records/16810831/files/Global-integrated-Plant-Tracker-July-2025_china.xlsx"
        ),
    output:
        powerplants="resources/data/existing_infrastructure/gem_data_raw/Global-integrated-Plant-Tracker-July-2025_china.xlsx",
    run:
        os.makedirs(os.path.dirname(output.powerplants), exist_ok=True)
        shutil.move(input.powerplants, output.powerplants)


if config["enable"].get("retrieve_cutout", False) and config["enable"].get(
    "build_cutout", False
):
    raise ValueError(
        "Settings error: you must choose between retrieving a pre-built cutout or building one from scratch"
    )
elif config["enable"].get("retrieve_cutout", False):

    rule retrieve_cutout:
        input:
            zenodo_cutout = storage.http("https://zenodo.org/record/16792792/files/China-2020c.nc"),
        output:
            cutout = "resources/cutouts/China-2020c.nc",
        run:
            os.makedirs(os.path.dirname(output.cutout), exist_ok=True)
            shutil.move(input.zenodo_cutout, output.cutout)

            # import logging
            # print("attempting")
            # logging.info("hellp")
            # logging.info(f"Unzipping {input[0]} to {output[0]}")

            # import shutil
            # import zipfile
            # import os

            # dirname = os.path.dirname(output[0])
            # print(dirname, os.path.exists(dirname))

            # # Move, unzip the file
            # shutil.move(input[0], dirname + ".zip")
            # logging.info(f"Unzipping {dirname}.zip to {dirname}")
            # print()
            # with zipfile.ZipFile(dirname + ".zip", 'r') as zip_ref:
            #     zip_ref.extractall(os.path.dirname(dirname))
            # os.remove(dirname + ".zip")
