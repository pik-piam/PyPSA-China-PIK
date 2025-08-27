# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Adapted from pypsa-EUR by the pypsa China-PIK authors

Calculates for each clustered region the
(i) installable capacity (based on land-use from :mod:`determine_availability_matrix`)
(ii) the available generation time series (based on weather data)
(iii) the average distanc from the node for onshore wind,
AC-connected offshore wind, DC-connected offshore wind and solar PV generators.

Outputs
-------

- ``resources/profile_{technology}.nc`` with the following structure

    ===================  ====================  =====================================================
    Field                Dimensions            Description
    ===================  ====================  =====================================================
    profile              year, bus, bin, time  the per unit hourly availability factors for each bus
    -------------------  --------------------  -----------------------------------------------------
    p_nom_max            bus, bin              maximal installable capacity at the bus (in MW)
    -------------------  --------------------  -----------------------------------------------------
    average_distance     bus, bin              average distance of units in the region to the
                                               grid bus for onshore techs and to the shoreline
                                               for offshore technologies (in km)
    ===================  ====================  =====================================================

Description
-----------

This script functions at two main spatial resolutions: the resolution of the
clustered network regions, and the resolution of the cutout grid cells for the
weather data. Typically the weather data grid is finer than the network regions,
so we have to work out the distribution of generators across the grid cells
within each region. This is done by taking account of a combination of the
available land at each grid cell (computed in
:mod:`determine_availability_matrix`) and the capacity factor there.

Based on the availability matrix, the script first computes how much of the
technology can be installed at each cutout grid cell. To compute the layout of
generators in each clustered region, the installable potential in each grid cell
is multiplied with the capacity factor at each grid cell. This is done since we
assume more generators are installed at cells with a higher capacity factor.

Based on the average capacity factor, the potentials are further divided into a
configurable number of resource classes (bins).

This layout is then used to compute the generation availability time series from
the weather data cutout from ``atlite``.

The maximal installable potential for the node (`p_nom_max`) is computed by
adding up the installable potentials of the individual grid cells.
"""

import logging
import time
from itertools import product

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from _helpers import configure_logging, mock_snakemake
from atlite import Cutout
from atlite.gis import ExclusionContainer
from constants import OFFSHORE_WIND_NODES, PROV_NAMES, TIMEZONE
from dask.distributed import Client
from readers_geospatial import read_offshore_province_shapes, read_province_shapes

logger = logging.getLogger(__name__)


def prepare_resource_config(params: dict, nprocesses: int, noprogress=True) -> tuple[dict]:
    """Parse the resource config (atlite calc config)

    Args:
        params (dict): the renewable options
        nprocesses (int): the number or processes
        noprogress (bool): whether to show progress bars

    Returns:
        (dict, dict): the resource config for the atlite calcs, the turbine/panel models
    """

    resource = params["resource"]  # pv panel params / wind turbine params
    resource["show_progress"] = not noprogress
    tech = "panel" if "panel" in resource else "turbine"

    # in case of multiple years
    models = resource[tech]
    if not isinstance(models, dict):
        models = {0: models}

    if nprocesses > 1:
        client = Client(n_workers=nprocesses, threads_per_worker=1)
        resource["dask_kwargs"] = {"scheduler": client}

    return resource, models


def build_resource_classes(
    cutout: Cutout,
    nbins: int,
    regions: gpd.GeoSeries,
    capacity_factor: xr.DataArray,
    params: dict,
) -> tuple[xr.DataArray, gpd.GeoSeries]:
    """Bin resources based on their capacity factor
    The number of bins can be dynamically reduced based on a min delta cf

    Args:
        cutout (Cutout): the atlite cutout
        nbins (int): the number of bins
        regions (gpd.GeoSeries): the regions
        capacity_factor (xr.DataArray,): the capacity factor
        params (dict): the config for VREs

    Returns:
        xr.DataArray: the mask for the resource classes
        gpd.GeoSeries: multi-indexed series [bus, bin]: geometry
    """
    resource_classes = params.get("resource_classes", {})
    nbins = resource_classes.get("n", 1)
    min_cf_delta = resource_classes.get("min_cf_delta", 0.0)
    buses = regions.index

    # indicator matrix for which cells touch which regions
    IndMat = np.ceil(cutout.availabilitymatrix(regions, ExclusionContainer()))
    cf_by_bus = capacity_factor * IndMat.where(IndMat > 0)

    epsilon = 1e-3
    cf_min, cf_max = (
        cf_by_bus.min(dim=["x", "y"]) - epsilon,
        cf_by_bus.max(dim=["x", "y"]) + epsilon,
    )

    # avoid binning resources that are very similar
    nbins_per_bus = [int(min(nbins, x)) for x in (cf_max - cf_min) // min_cf_delta]
    normed_bins = xr.DataArray(
        np.vstack(
            [np.hstack([[0] * (nbins - n), np.linspace(0, 1, n + 1)]) for n in nbins_per_bus]
        ),
        dims=["bus", "bin"],
        coords={"bus": regions.index},
    )
    bins = cf_min + (cf_max - cf_min) * normed_bins

    cf_by_bus_bin = cf_by_bus.expand_dims(bin=range(nbins))
    lower_edges = bins[:, :-1]
    upper_edges = bins[:, 1:]
    class_masks = (cf_by_bus_bin >= lower_edges) & (cf_by_bus_bin < upper_edges)

    if nbins == 1:
        bus_bin_mi = pd.MultiIndex.from_product([regions.index, [0]], names=["bus", "bin"])
        class_regions = regions.set_axis(bus_bin_mi)
        class_regions["cf"] = bins.to_series()
    else:
        grid = cutout.grid.set_index(["y", "x"])
        class_regions = {}
        for bus, bin_id in product(buses, range(nbins)):
            bus_bin_mask = (
                class_masks.sel(bus=bus, bin=bin_id).stack(spatial=["y", "x"]).to_pandas()
            )
            grid_cells = grid.loc[bus_bin_mask]
            geometry = grid_cells.intersection(regions.loc[bus, "geometry"]).union_all().buffer(0)
            class_regions[(bus, bin_id)] = geometry

        class_regions = gpd.GeoDataFrame(
            {"geometry": class_regions.values()},
            index=pd.MultiIndex.from_tuples(class_regions.keys(), names=["bus", "bin"]),
        )
        class_regions["cf"] = bins.to_series()

    return class_masks, class_regions


def localize_cutout_time(cutout: Cutout, drop_leap=True) -> Cutout:
    """Localize the time to the local timezone

    Args:
        cutout (Cutout): the atlite cutout object
        drop_leap (bool, optional): drop 29th Feb. Defaults to True.

    Returns:
        Cutout: the updated cutout
    """

    data = cutout.data

    timestamps = pd.DatetimeIndex(data.time)
    # go from ECMWF/atlite UTC to local time
    ts_naive = timestamps.tz_localize("UTC").tz_convert(TIMEZONE).tz_localize(None)
    cutout.data = cutout.data.assign_coords(time=ts_naive)

    if drop_leap:
        data = cutout.data
        cutout.data = data.sel(time=~((data.time.dt.month == 2) & (data.time.dt.day == 29)))

    return cutout


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "build_renewable_profiles", technology="solar", rc_params="n3_min_cf_0.05"
        )

    configure_logging(snakemake)

    nprocesses = int(snakemake.threads)
    noprogress = snakemake.config["run"].get("disable_progressbar", True)
    noprogress = noprogress or not snakemake.config["atlite"]["show_progress"]

    technology = snakemake.wildcards.technology
    params = snakemake.config["renewable"][technology]

    resource, models = prepare_resource_config(params, nprocesses, noprogress)
    logger.info(f"Resource config: {resource}")

    if technology != "offwind":
        regions = read_province_shapes(snakemake.input.province_shape)
        regions = regions.reindex(PROV_NAMES).rename_axis("bus")
        buses = regions.index
    else:
        regions = read_offshore_province_shapes(snakemake.input.offshore_province_shapes)
        regions = regions.reindex(OFFSHORE_WIND_NODES).rename_axis("bus")
        buses = regions.index

    cutout = Cutout(snakemake.input.cutout)
    cutout = localize_cutout_time(cutout, drop_leap=True)

    func = getattr(cutout, resource.pop("method"))
    availability = xr.open_dataarray(snakemake.input.availability_matrix)

    correction_factor = params.get("correction_factor", 1.0)
    capacity_per_sqkm = params["capacity_per_sqkm"]
    capacity_factor = correction_factor * func(capacity_factor=True, **resource)

    area = cutout.grid.to_crs(3035).area / 1e6
    area = xr.DataArray(area.values.reshape(cutout.shape), [cutout.coords["y"], cutout.coords["x"]])

    if correction_factor != 1.0:
        logger.info(f"correction_factor is set as {correction_factor}")

    logger.info(f"Calculate average capacity factor per grid cell for technology {technology}...")
    start = time.time()

    duration = time.time() - start
    logger.info(
        "Completed average capacity factor calculation per grid cell for technology"
        + f" {technology} ({duration:2.2f}s)"
    )

    nbins = params.get("resource_classes", 1)
    logger.info(f"Create masks for {nbins} resource classes for technology {technology}...")
    start = time.time()

    class_masks, class_regions = build_resource_classes(
        cutout=cutout,
        nbins=nbins,
        regions=regions,
        capacity_factor=capacity_factor,
        params=params,
    )

    duration = time.time() - start
    logger.info(
        f"Completed resource class calculation for technology {technology} ({duration:2.2f}s)"
    )

    class_regions.to_file(snakemake.output.class_regions, index=True)

    layout = capacity_factor * area * capacity_per_sqkm

    profiles = []
    for year, panel_or_turbine in models.items():
        logger.info(
            "Calculate weighted capacity factor time series for model"
            + f" {panel_or_turbine} for technology {technology}..."
        )
        start = time.time()

        tech = "panel" if "panel" in resource else "turbine"
        resource[tech] = panel_or_turbine

        matrix = (availability * class_masks).stack(bus_bin=["bus", "bin"], spatial=["y", "x"])

        profile = func(
            matrix=matrix,
            layout=layout,
            index=matrix.indexes["bus_bin"],
            per_unit=True,
            return_capacity=False,
            **resource,
        )
        profile = profile.unstack("bus_bin")

        dim = {"year": [year]}
        profile = profile.expand_dims(dim)

        profiles.append(profile.rename("profile"))

        duration = time.time() - start
        logger.info(
            "Completed weighted capacity factor time series calculation for model"
            + f" {panel_or_turbine} for technology {technology} ({duration:2.2f}s)"
        )

    logger.info(f"calculated n={len(profiles)} profiles")
    logger.info(f"Calculating maximal capacity per bus for technology {technology}")
    profiles = xr.merge(profiles)
    p_nom_max = capacity_per_sqkm * availability * class_masks @ area

    logger.info(f"Calculate average distances for technology {technology}.")
    layoutmatrix = (layout * availability * class_masks).stack(
        bus_bin=["bus", "bin"], spatial=["y", "x"]
    )

    coords = cutout.grid.representative_point().to_crs(3035)

    average_distance = []
    bus_bins = layoutmatrix.indexes["bus_bin"]
    logger.info(f"Calculating average distances for {len(bus_bins)} bus bins")

    regional_centers = regions.geometry.representative_point().to_crs(3035)
    for bus, bin in bus_bins:
        row = layoutmatrix.sel(bus=bus, bin=bin).data
        nz_b = row != 0
        row = row[nz_b]
        co = coords[nz_b]
        distances = co.distance(regional_centers[bus]).div(1e3)  # km
        average_distance.append((distances * (row / row.sum())).sum())

    average_distance = xr.DataArray(average_distance, [bus_bins]).unstack("bus_bin")
    average_distance.to_netcdf(snakemake.output.average_distance)

    ds = xr.merge(
        [
            correction_factor * profiles,
            p_nom_max.rename("p_nom_max"),
            average_distance.rename("average_distance"),
        ]
    )
    # select only buses with some capacity and minimal capacity factor
    mean_profile = ds["profile"].mean("time").max(["year", "bin"])
    sum_potential = ds["p_nom_max"].sum("bin")

    ds = ds.sel(
        bus=(
            (mean_profile > params.get("min_p_max_pu", 0.0))
            & (sum_potential > params.get("min_p_nom_max", 0.0))
        )
    )

    if "clip_p_max_pu" in params:
        min_p_max_pu = params["clip_p_max_pu"]
        ds["profile"] = ds["profile"].where(ds["profile"] >= min_p_max_pu, 0)

    ds.to_netcdf(snakemake.output.profile)

    if nprocesses > 1:
        client = resource["dask_kwargs"]["scheduler"]
        client.shutdown()
