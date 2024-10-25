import logging
from _helpers import configure_logging

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import atlite
import xarray as xr
from constants import PROV_NAMES, CRS

logger = logging.getLogger(__name__)


def xarr_to_gdf(
    xarr: xr.DataArray, var_name="__xarray_dataarray_variable__", x_var="x", y_var="y", crs=CRS
) -> gpd.GeoDataFrame:
    """convert an xarray to GDF

    Args:
        xarr (xr.DataArray): the input array
        var_name (str, optional): the array variable to be converted . Defaults to "__xarray_dataarray_variable__".
        x_var (str, optional): the x dimension. Defaults to "x".
        y_var (str, optional): the y dimension. Defaults to "y".
        crs (_type_, optional): the crs. Defaults to CRS.

    Returns:
        gpd.GeoDataFrame: geodata frame in chosen CRS
    """
    df = xarr.to_dataframe()
    df.reset_index(inplace=True)
    return gpd.GeoDataFrame(
        df[var_name], geometry=gpd.points_from_xy(df[x_var], df[y_var]), crs=crs
    )


def build_population_map():

    # =============== load data ===================
    with pd.HDFStore(snakemake.input.population, mode="r") as store:
        pop_province_count = store["population"]

    da = xr.open_dataset(snakemake.input.population_density_grid)
    pop_ww = xarr_to_gdf(da)  # TODO is the CRS correct?

    # CFSR points and Provinces
    prov_poly = gpd.read_file(snakemake.input.province_shape)[["province", "geometry"]]
    prov_poly.set_index("province", inplace=True)
    prov_poly = prov_poly.reindex(PROV_NAMES)
    prov_poly.reset_index(inplace=True)

    # load renewable profiles & grid & extract gridpoints
    cutout = atlite.Cutout(snakemake.input.cutout)
    cutout.grid
    # TODO check this is exact replica of previous code
    c_grid_points = cutout.coords
    grid_points = xarr_to_gdf(c_grid_points.to_dataset())
    # df = pd.DataFrame({"Coordinates": tuple(map(tuple, c_grid_points))})
    # df["Coordinates"] = df["Coordinates"].apply(Point)
    # grid_points = gpd.GeoDataFrame(df, geometry="Coordinates", crs=CRS)

    # match cutout grid to province
    cutout_pts_in_prov = gpd.tools.sjoin(grid_points, prov_poly, how="left", predicate="intersects")
    cutout_pts_in_prov.rename(
        columns={"index_right": "province_index", "province": "province_name"}, inplace=True
    )

    #### Province masks merged with population density
    # TODO: THIS REQUIRES EXPLANATION - can't just use random crs :||
    cutout_pts_in_prov = cutout_pts_in_prov.to_crs(3857)
    pop_ww = pop_ww.to_crs(3857)

    merged = gpd.tools.sjoin_nearest(cutout_pts_in_prov, pop_ww, how="inner")
    merged = merged.to_crs(CRS)

    #### save in the right format

    points_in_provinces = pd.DataFrame(index=cutout_pts_in_prov.index)

    # TODO switch to native pandas with groupbyapply
    # normalise pop per province
    for province in PROV_NAMES:
        pop_pro = merged[merged.province_name == province].Population_density
        points_in_provinces[province] = pop_pro / pop_pro.sum()

    points_in_provinces.index.name = ""
    points_in_provinces.fillna(0.0, inplace=True)
    # re multiply by province
    points_in_provinces *= pop_province_count

    with pd.HDFStore(snakemake.output.population_map, mode="w", complevel=4) as store:
        store["population_gridcell_map"] = points_in_provinces


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_population_gridcell_map")

    configure_logging(snakemake)

    build_population_map()
