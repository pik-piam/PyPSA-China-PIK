# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT
""" 
Functions associated with the build_renewable_potential rule.
- Temporal Profiles are built based on the atlite cutout
- Potentials are built based on the atlite cutout and raster data (land availability)
"""
import progressbar as pgb

import logging
import functools
import atlite
import xarray as xr
import geopandas as gpd
import numpy as np
import time
import pandas as pd
from atlite.gis import ExclusionContainer
from os import PathLike

from _helpers import mock_snakemake, configure_logging, calc_utc_timeshift, get_cutout_params
from readers import read_province_shapes
from constants import PROV_NAMES, CRS, OFFSHORE_WIND_NODES, DEFAULT_OFFSHORE_WIND_CORR_FACTOR

logger = logging.getLogger(__name__)


def make_solar_profile(
    solar_config: dict,
    cutout: atlite.Cutout,
    outp_path: PathLike,
    delta_t: pd.Timedelta,
):
    """Make the solar geographical potentials and per unit availability time series for each
    raster cell
    ! Somewhat compute intensive !

    Args:
        solar_config (dict): the solar configuration (from the yaml config read by snakemake)
        cutout (atlite.Cutout): the atlite cutout
        outp_path (PathLike): the output path for the raster data
        delta_t (pd.Timedelta): the time delta to convert to ntwk time
    """

    logger.info("Making solar profile ")
    solar_config = snakemake.config["renewable"]["solar"]
    solar_resource = solar_config["resource"]
    solar_correction_factor = solar_config.get("correction_factor", 1.0)
    solar_capacity_per_sqkm = solar_config["capacity_per_sqkm"]
    if solar_correction_factor != 1.0:
        logger.info(f"solar_correction_factor is set as {solar_correction_factor}")

    # TODO not hardcoded res
    excluder_solar = ExclusionContainer(crs=3035, res=500)
    excluder_build_up = ExclusionContainer(crs=3035, res=500)

    build_up = snakemake.input["Build_up_raster"]

    excluder_build_up.add_raster(build_up, invert=True, crs=CRS)
    excluder_solar.add_raster(grass, invert=True, crs=CRS)
    excluder_solar.add_raster(bare, invert=True, crs=CRS)
    excluder_solar.add_raster(shrubland, invert=True, crs=CRS)

    kwargs = dict(nprocesses=nprocesses, disable_progressbar=noprogress)
    # TODO remove if else?
    if noprogress:
        logger.info("Calculate solar landuse availabilities...")
        start = time.time()
        solar_matrix = cutout.availabilitymatrix(provinces_shp, excluder_solar, **kwargs)
        buildup_matrix = cutout.availabilitymatrix(provinces_shp, excluder_build_up, **kwargs)
        duration = time.time() - start
        logger.info(f"Completed solar availability calculation ({duration:2.2f}s)")
    else:
        solar_matrix = cutout.availabilitymatrix(
            shapes=provinces_shp, excluder=excluder_solar, **kwargs
        )
        buildup_matrix = cutout.availabilitymatrix(provinces_shp, excluder_build_up, **kwargs)

    solar_potential = (
        solar_capacity_per_sqkm * solar_matrix.sum("bus") * area
        + solar_capacity_per_sqkm * buildup_matrix.sum("bus") * area
    )

    solar_func = getattr(cutout, solar_resource.pop("method"))
    solar_resource["dask_kwargs"] = {"num_workers": nprocesses}  # ?
    solar_capacity_factor = solar_correction_factor * solar_func(
        capacity_factor=True, **solar_resource
    )
    solar_layout = solar_capacity_factor * area * solar_capacity_per_sqkm
    solar_profile, solar_capacities = solar_func(
        matrix=solar_matrix.stack(spatial=["y", "x"]),
        layout=solar_layout,
        index=buses,
        per_unit=True,
        return_capacity=True,
        **solar_resource,
    )

    logger.info("Calculating solar maximal capacity per bus (method 'simple')")

    solar_p_nom_max = solar_capacity_per_sqkm * solar_matrix @ area

    solar_ds = xr.merge(
        [
            (solar_correction_factor * solar_profile).rename("profile"),
            solar_capacities.rename("weight"),
            solar_p_nom_max.rename("p_nom_max"),
            solar_potential.rename("potential"),
        ]
    )

    solar_ds = solar_ds.sel(
        bus=(
            (solar_ds["profile"].mean("time") > solar_config.get("min_p_max_pu", 0.0))
            & (solar_ds["p_nom_max"] > solar_config.get("min_p_nom_max", 0.0))
        )
    )

    if "clip_p_max_pu" in solar_config:
        min_p_max_pu = solar_config["clip_p_max_pu"]
        solar_ds["profile"] = solar_ds["profile"].where(solar_ds["profile"] >= min_p_max_pu, 0)

    # shift back from UTC to network time
    solar_ds["time"] = solar_ds["time"].values + delta_t

    solar_ds.to_netcdf(outp_path)


def make_onshore_wind_profile(
    onwind_config: dict, cutout: atlite.Cutout, outp_path: PathLike, delta_t: pd.Timedelta
):
    """Make the onwind geographical potentials and per unit availability time series for
    each raster cell
    ! Somewhat compute intensive !

    Args:
        onwind_config (dict): the onshore wind config (from the yaml config read by snakemake)
        cutout (atlite.Cutout): the atlite cutout
        outp_path (PathLike): the output path for the raster data
        delta_t (pd.Timedelta): the time delta to convert to ntwk time
    """

    logger.info("Making onshore wind profile ")

    onwind_resource = onwind_config["resource"]
    onwind_correction_factor = onwind_config.get("correction_factor", 1.0)
    onwind_capacity_per_sqkm = onwind_config["capacity_per_sqkm"]
    if onwind_correction_factor != 1.0:
        logger.info(f"onwind_correction_factor is set as {onwind_correction_factor}")

    excluder_onwind = ExclusionContainer(crs=3035, res=500)

    excluder_onwind.add_raster(grass, invert=True, crs=4326)
    excluder_onwind.add_raster(bare, invert=True, crs=4326)
    excluder_onwind.add_raster(shrubland, invert=True, crs=4326)

    kwargs = dict(nprocesses=nprocesses, disable_progressbar=noprogress)
    if noprogress:
        logger.info("Calculate onwind landuse availabilities...")
        start = time.time()
        onwind_matrix = cutout.availabilitymatrix(provinces_shp, excluder_onwind, **kwargs)
        duration = time.time() - start
        logger.info(f"Completed onwind availability calculation ({duration:2.2f}s)")
    else:
        onwind_matrix = cutout.availabilitymatrix(provinces_shp, excluder_onwind, **kwargs)

    onwind_potential = onwind_capacity_per_sqkm * onwind_matrix.sum("bus") * area

    onwind_func = getattr(cutout, onwind_resource.pop("method"))
    onwind_resource["dask_kwargs"] = {"num_workers": nprocesses}  # ?
    onwind_capacity_factor = onwind_correction_factor * onwind_func(
        capacity_factor=True, **onwind_resource
    )
    onwind_layout = onwind_capacity_factor * area * onwind_capacity_per_sqkm
    onwind_profile, onwind_capacities = onwind_func(
        matrix=onwind_matrix.stack(spatial=["y", "x"]),
        layout=onwind_layout,
        index=buses,
        per_unit=True,
        return_capacity=True,
        **onwind_resource,
    )

    logger.info("Calculating onwind maximal capacity per bus (method 'simple')")

    onwind_p_nom_max = onwind_capacity_per_sqkm * onwind_matrix @ area

    onwind_ds = xr.merge(
        [
            (onwind_correction_factor * onwind_profile).rename("profile"),
            onwind_capacities.rename("weight"),
            onwind_p_nom_max.rename("p_nom_max"),
            onwind_potential.rename("potential"),
        ]
    )

    onwind_ds = onwind_ds.sel(
        bus=(
            (onwind_ds["profile"].mean("time") > onwind_config.get("min_p_max_pu", 0.0))
            & (onwind_ds["p_nom_max"] > onwind_config.get("min_p_nom_max", 0.0))
        )
    )

    if "clip_p_max_pu" in onwind_config:
        min_p_max_pu = onwind_config["clip_p_max_pu"]
        onwind_ds["profile"] = onwind_ds["profile"].where(onwind_ds["profile"] >= min_p_max_pu, 0)

    # shift back from UTC to network time
    onwind_ds["time"] = onwind_ds["time"].values + delta_t
    onwind_ds.to_netcdf(outp_path)


def make_offshore_wind_profile(
    offwind_config: dict, cutout: atlite.Cutout, outp_path: PathLike, delta_t: pd.Timedelta
):
    """Make the offwind geographical potentials and per unit availability time series for
      each raster cell
    ! Somewhat compute intensive !


    Args:
        offwind_config (dict): the configuration for the offshore wind
        cutout (atlite.Cutout): the atlite cutout
        outp_path (PathLike): the output path for the raster date
        delta_t (pd.Timedelta): the time delta to convert to ntwk time
    """
    offwind_resource = offwind_config["resource"]
    offwind_correction_factor = offwind_config.get(
        "correction_factor", DEFAULT_OFFSHORE_WIND_CORR_FACTOR
    )
    offwind_capacity_per_sqkm = offwind_config["capacity_per_sqkm"]
    if offwind_correction_factor != 1.0:
        logger.info(f"offwind_correction_factor is set as {offwind_correction_factor}")

    offwind_provinces = OFFSHORE_WIND_NODES

    EEZ_shp = gpd.read_file(snakemake.input["offshore_shapes"])
    EEZ_province_shp = gpd.read_file(snakemake.input["offshore_province_shapes"]).set_index("index")
    EEZ_province_shp = EEZ_province_shp.reindex(offwind_provinces).rename_axis("bus")
    excluder_offwind = ExclusionContainer(crs=3035, res=500)

    if "max_depth" in offwind_config:
        func = functools.partial(np.greater, -offwind_config["max_depth"])
        excluder_offwind.add_raster(snakemake.input.gebco, codes=func, crs=CRS, nodata=-1000)

    if offwind_config["natura"]:
        Protected_shp = gpd.read_file(snakemake.input["natura1"])
        Protected_shp1 = gpd.read_file(snakemake.input["natura2"])
        Protected_shp2 = gpd.read_file(snakemake.input["natura3"])
        Protected_shp = pd.concat([Protected_shp, Protected_shp1], ignore_index=True)
        Protected_shp = pd.concat([Protected_shp, Protected_shp2], ignore_index=True)
        Protected_shp = Protected_shp.geometry
        Protected_shp = gpd.GeoDataFrame(Protected_shp)
        Protected_Marine_shp = gpd.tools.overlay(Protected_shp, EEZ_shp, how="intersection")
        # this is to avoid atlite complaining about parallelisation
        Protected_Marine_shp.to_file(snakemake.output.protected_areas_offshore)
        # excluder_offwind.add_geometry(Protected_Marine_shp.geometry)
        excluder_offwind.add_geometry("Protected_Marine.shp")

    kwargs = dict(nprocesses=nprocesses, disable_progressbar=noprogress)
    if noprogress:
        logger.info("Calculate offwind landuse availabilities...")
        start = time.time()
        offwind_matrix = cutout.availabilitymatrix(EEZ_province_shp, excluder_offwind, **kwargs)
        duration = time.time() - start
        logger.info(f"Completed offwind availability calculation ({duration:2.2f}s)")
    else:
        offwind_matrix = cutout.availabilitymatrix(EEZ_province_shp, excluder_offwind, **kwargs)

    offwind_potential = offwind_capacity_per_sqkm * offwind_matrix.sum("bus") * area

    offwind_func = getattr(cutout, offwind_resource.pop("method"))
    offwind_resource["dask_kwargs"] = {"num_workers": nprocesses}  # ?
    offwind_capacity_factor = offwind_correction_factor * offwind_func(
        capacity_factor=True, **offwind_resource
    )
    offwind_layout = offwind_capacity_factor * area * offwind_capacity_per_sqkm
    offwind_profile, offwind_capacities = offwind_func(
        matrix=offwind_matrix.stack(spatial=["y", "x"]),
        layout=offwind_layout,
        index=EEZ_province_shp.index,
        per_unit=True,
        return_capacity=True,
        **offwind_resource,
    )

    logger.info("Calculating offwind maximal capacity per bus (method 'simple')")

    offwind_p_nom_max = offwind_capacity_per_sqkm * offwind_matrix @ area

    offwind_ds = xr.merge(
        [
            (offwind_correction_factor * offwind_profile).rename("profile"),
            offwind_capacities.rename("weight"),
            offwind_p_nom_max.rename("p_nom_max"),
            offwind_potential.rename("potential"),
        ]
    )

    offwind_ds = offwind_ds.sel(
        bus=(
            (offwind_ds["profile"].mean("time") > offwind_config.get("min_p_max_pu", 0.0))
            & (offwind_ds["p_nom_max"] > offwind_config.get("min_p_nom_max", 0.0))
        )
    )

    if "clip_p_max_pu" in offwind_config:
        min_p_max_pu = offwind_config["clip_p_max_pu"]
        offwind_ds["profile"] = offwind_ds["profile"].where(
            offwind_ds["profile"] >= min_p_max_pu, 0
        )
    # shift back from UTC to network time
    offwind_ds["time"] = offwind_ds["time"].values + delta_t
    offwind_ds.to_netcdf(outp_path)


if __name__ == "__main__":

    if "snakemake" not in globals():
        snakemake = mock_snakemake("build_renewable_potential")

    configure_logging(snakemake, logger=logger)
    pgb.streams.wrap_stderr()  # ?

    nprocesses = int(snakemake.threads)  # ?
    noprogress = not snakemake.config["atlite"].get("show_progress", True)

    cutout = atlite.Cutout(snakemake.input.cutout)
    cutout.prepare()
    provinces_shp = read_province_shapes(snakemake.input.provinces_shp)
    provinces_shp = provinces_shp.reindex(PROV_NAMES).rename_axis("bus")
    buses = provinces_shp.index

    grass = snakemake.input.Grass_raster
    bare = snakemake.input.Bare_raster
    shrubland = snakemake.input.Shrubland_raster

    # TODO explain
    area = cutout.grid.to_crs(3035).area / 1e6
    area = xr.DataArray(area.values.reshape(cutout.shape), [cutout.coords["y"], cutout.coords["x"]])

    # atlite to network timedelta
    weather_year = get_cutout_params(snakemake.config)["weather_year"]
    delta_t = calc_utc_timeshift(snakemake.config["snapshots"], weather_year)

    if snakemake.config["Technique"]["solar"]:
        make_solar_profile(
            solar_config=snakemake.config["renewable"]["solar"],
            cutout=cutout,
            outp_path=snakemake.output.solar_profile,
            delta_t=delta_t,
        )

    if snakemake.config["Technique"]["onwind"]:
        make_onshore_wind_profile(
            onwind_config=snakemake.config["renewable"]["onwind"],
            cutout=cutout,
            outp_path=snakemake.output.onwind_profile,
            delta_t=delta_t,
        )

    if snakemake.config["Technique"]["offwind"]:
        offwind_config = snakemake.config["renewable"]["offwind"]
        make_offshore_wind_profile(
            offwind_config=offwind_config,
            cutout=cutout,
            outp_path=snakemake.output.offwind_profile,
            delta_t=delta_t,
        )

    logger.info("Renewable potential profiles successfully built.")
