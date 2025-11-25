# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# Adapted by the PyPSA-China team
# SPDX-License-Identifier: MIT
"""
The script performs a land eligibility analysis of what share of land is
availability for developing the selected technology at each cutout grid cell.
The script uses the `atlite <https://github.com/pypsa/atlite>`_ library and
several GIS datasets like the Copernicus land use data,
Natura2000 nature reserves, GEBCO bathymetry data

The copernicus land monitoring data is/can be fetched by the pipeline.
The GEBCO data must currently be manually downloaded
  <https://en.wikipedia.org/wiki/Bathymetry>`_ data set with a global terrain
  model for ocean and land at 15 arc-second intervals by the `General
  Bathymetric Chart of the Oceans (GEBCO)
  <https://www.gebco.net/data_and_products/gridded_bathymetry_data/>`_.
<https://www.gebco.net/data_and_products/images/gebco_2019_grid_image.jpg>`_
The natura data must be manually downloaded for now or retrieved with the data bundle

"""

import functools
import logging
import os.path
import time
from os import mkdir

import atlite
import geopandas as gpd
import numpy as np
from _helpers import configure_logging, mock_snakemake
from constants import OFFSHORE_WIND_NODES, PROV_NAMES
from pandas import concat
from readers_geospatial import read_offshore_province_shapes, read_province_shapes

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake("build_availability_matrix", technology="solar")

    configure_logging(snakemake)

    nprocesses = int(snakemake.threads)
    noprogress = snakemake.config["run"].get("disable_progressbar", True)
    noprogress = bool(noprogress or not snakemake.config["atlite"]["show_progress"])

    technology = snakemake.wildcards.technology
    params = snakemake.config["renewable"][technology]

    if technology != "offwind":
        regions = read_province_shapes(snakemake.input.province_shape)
        regions = regions.reindex(PROV_NAMES).rename_axis("bus")
        buses = regions.index
    else:
        regions = read_offshore_province_shapes(snakemake.input.offshore_province_shapes)
        regions = regions.reindex(OFFSHORE_WIND_NODES).rename_axis("bus")
        buses = regions.index

    cutout = atlite.Cutout(snakemake.input.cutout)

    res = params.get("excluder_resolution", 100)
    excluder = atlite.ExclusionContainer(crs=3035, res=res)

    # TODO improve pipeline and combine the files into a single one as part of the fetch
    if not params["natura"]:
        nat1 = gpd.read_file(snakemake.input["natura1"])
        nat2 = gpd.read_file(snakemake.input["natura2"])
        nat3 = gpd.read_file(snakemake.input["natura3"])
        protected_shp = gpd.GeoDataFrame(concat([nat1, nat2, nat3], ignore_index=True))
        protected_shp = gpd.GeoDataFrame(protected_shp.geometry)

        if technology == "offwind":
            protected_Marine_shp = gpd.tools.overlay(
                protected_shp, regions.union_all("unary"), how="intersection"
            )
            # TO
            # this is to avoid atlite complaining about parallelisation (still relevant?)
            logger.info("Creating tmp directory for protected marine shapefile")
            TMP = "resources/derived_data/tmp/atlite_protected_marine.shp"
            logger.info(f"parent exists: {os.path.isdir(os.path.dirname(os.path.dirname(TMP)))}")
            if not os.path.isdir(os.path.dirname(os.path.dirname(TMP))):
                mkdir(os.path.dirname(os.path.dirname(TMP)))
            if not os.path.isdir(os.path.dirname(TMP)):
                mkdir(os.path.dirname(TMP))
            protected_Marine_shp.to_file(TMP)
            excluder.add_geometry(TMP)

    # Use Copernicus LC100 discrete classification map instead of percentage cover fractions
    # This replaces the old approach of using separate Grass/Bare/Shrubland rasters
    if technology != "offwind":
        # Land cover codes from config for allowed areas
        # Copernicus LC100 codes: 20=Shrubland, 30=Herbaceous, 40=Agriculture,
        # 50=Urban, 60=Bare, 90=Wetland, 100=Moss
        codes = snakemake.params.land_cover_codes

        if codes is None:
            logger.warning(
                f"No land_cover_codes defined for {technology}, " "skipping land cover filtering"
            )
        else:
            logger.info(f"Using Copernicus LC100 land cover codes for {technology}: {codes}")
            # invert=True means these codes represent ALLOWED areas (not excluded)
            excluder.add_raster(
                snakemake.input.copernicus_land_cover,
                codes=codes,
                invert=True,
                crs=4326,
            )

    if params.get("max_slope"):
        func = functools.partial(np.less, params["max_slope"])
        excluder.add_raster(
            snakemake.input.gebco_slope,
            codes=func,
            crs=4326,
        )

    if params.get("max_altitude"):
        func = functools.partial(np.less, params["max_altitude"])
        excluder.add_raster(
            snakemake.input.gebco,
            codes=func,
            nodata=-32767,
        )
    # Note: Built-up area handling is integrated into land_cover_codes in config
    # To exclude built-up areas, remove code 50 from the land_cover_codes list

    if params.get("max_depth"):
        func = functools.partial(np.greater, -params["max_depth"])
        # lambda not supported for atlite + multiprocessing
        # use named function np.greater with partially frozen argument instead
        # and exclude areas where: -max_depth > grid cell depth
        excluder.add_raster(snakemake.input.gebco, codes=func, nodata=-32767)

    if params.get("min_depth"):
        func = functools.partial(np.greater, -params["min_depth"])
        excluder.add_raster(snakemake.input.gebco, codes=func, nodata=-32767, invert=True)

    if "min_shore_distance" in params:
        buffer = params["min_shore_distance"]
        excluder.add_geometry(regions.union_all("unary"), buffer=buffer)

    if "max_shore_distance" in params:
        buffer = params["max_shore_distance"]
        excluder.add_geometry(
            excluder.add_geometry(regions.union_all("unary")),
            buffer=buffer,
            invert=True,
        )

    logger.info(f"Calculate landuse availability for {technology}...")
    start = time.time()

    kwargs = dict(nprocesses=nprocesses, disable_progressbar=noprogress)
    availability = cutout.availabilitymatrix(regions, excluder, **kwargs)

    duration = time.time() - start
    logger.info(f"Completed landuse availability calculation for {technology} ({duration:2.2f}s)")

    availability.to_netcdf(snakemake.output[0])
