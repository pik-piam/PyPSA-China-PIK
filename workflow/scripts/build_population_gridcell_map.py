import logging
import atlite
import xarray as xr
import pandas as pd
import geopandas as gpd

from constants import PROV_NAMES, CRS
from os import PathLike

from _helpers import configure_logging, mock_snakemake
from readers_geospatial import read_pop_density, read_province_shapes

logger = logging.getLogger(__name__)


def xarr_to_gdf(
    xarr: xr.DataArray, var_name: str, x_var="x", y_var="y", crs=CRS
) -> gpd.GeoDataFrame:
    """convert an xarray to GDF

    Args:
        xarr (xr.DataArray): the input array
        var_name (str): the array variable to be converted.
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


def load_cfrs_data(target: PathLike) -> gpd.GeoDataFrame:
    """load  CFRS_grid.nc type files into a geodatafram

    Args:
        target (PathLike): the abs path

    Returns:
        gpd.GeoDataFrame: the data in gdf
    """
    pop_density = xr.open_dataarray(target).to_dataset(name="pop_density")
    pop_ww = xarr_to_gdf(pop_density, var_name="pop_density")  # TODO is the CRS correct?

    return pop_ww

# TODO see whether still needed
def build_gridded_population(
    prov_pop_path: PathLike,
    pop_density_raster_path: PathLike,
    cutout_path: PathLike,
    province_shape_path: PathLike,
    gridded_pop_out: PathLike,
):
    """Build a gridded population DataFrame by matching population density to the cutout grid cells.
    This DataFrame is a sparse matrix of the population and shape BusesxCutout_gridcells
      where buses are the provinces

    Args:
        prov_pop_path (PathLike): Path to the province population count file (hdf5).
        pop_density_raster_path (PathLike): Path to the population density raster file.
        cutout_path (PathLike): Path to the cutout file containing the grid.
        province_shape_path (PathLike): Path to the province shape file.
        gridded_pop_out (PathLike): output file path.
    """

    with pd.HDFStore(prov_pop_path, mode="r") as store:
        pop_province = store["population"]

    prov_poly = read_province_shapes(province_shape_path)
    pop_density = read_pop_density(pop_density_raster_path, prov_poly, crs=CRS)

    cutout = atlite.Cutout(cutout_path)
    grid_points = cutout.grid
    # this is in polygons but need points for sjoin with pop dnesity to work
    grid_points.to_crs(3857, inplace=True)
    grid_points["geometry"] = grid_points.centroid
    grid_points.to_crs(CRS, inplace=True)

    # match cutout grid to province
    # cutout_pts_in_prov = gpd.tools.sjoin(grid_points, prov_poly,
    # how="left", predicate="intersects")
    # TODO: do you want to dropna here?
    cutout_pts_in_prov = gpd.tools.sjoin(
        grid_points, prov_poly, how="left", predicate="intersects"
    )  # .dropna()
    cutout_pts_in_prov.rename(
        columns={"index_right": "province_index", "province": "province_name"},
        inplace=True,
    )

    # match cutout grid to province
    cutout_pts_in_prov = gpd.tools.sjoin(grid_points, prov_poly, how="left", predicate="intersects")
    cutout_pts_in_prov.rename(
        columns={"index_right": "province_index", "province": "province_name"},
        inplace=True,
    )
    # cutout_pts_in_prov.dropna(inplace=True)

    # TODO CRS, think about whether this makes sense or need grid interp
    merged = gpd.tools.sjoin_nearest(
        cutout_pts_in_prov.to_crs(3857), pop_density.to_crs(3857), how="inner"
    )
    merged = merged.to_crs(CRS)
    # points outside china are NaN, need to rename to keep the index cutout after agg
    # otherwise the spare matrix will not match the cutoutpoints
    #  (smarter would be to change the cutout)
    merged.fillna({"province_name": "OutsideChina"}, inplace=True)

    points_in_provinces = pd.DataFrame(index=cutout_pts_in_prov.index)
    # normalise pop per province and make a loc_id/province table
    points_in_provinces = (
        merged.groupby("province_name")["pop_density"]
        .apply(lambda x: x / x.sum())
        .unstack(fill_value=0.0)
        .T
    )
    # now get rid of the outside china "province"
    points_in_provinces.drop(columns="OutsideChina", inplace=True)
    points_in_provinces.index.name = ""
    points_in_provinces.fillna(0.0, inplace=True)

    points_in_provinces *= pop_province

    with pd.HDFStore(gridded_pop_out, mode="w", complevel=4) as store:
        store["population_gridcell_map"] = points_in_provinces


def build_population_map(
    prov_pop_path: PathLike,
    pop_density_raster_path: PathLike,
    cutout_path: PathLike,
    province_shape_path: PathLike,
    gridded_pop_out: PathLike,
):
    """Build a gridded population DataFrame by matching population density to the cutout grid cells.
    This DataFrame is a sparse matrix of the population and shape BusesxCutout_gridcells
      where buses are the provinces

    Args:
        prov_pop_path (PathLike): Path to the province population count file (hdf5).
        pop_density_raster_path (PathLike): Path to the population density raster file.
        cutout_path (PathLike): Path to the cutout file containing the grid.
        province_shape_path (PathLike): Path to the province shape file.
        gridded_pop_out (PathLike): output file path.
    """

    # =============== load data ===================
    with pd.HDFStore(prov_pop_path, mode="r") as store:
        pop_province_count = store["population"]

    # CFSR points and Provinces
    pop_ww = load_cfrs_data(pop_density_raster_path)

    prov_poly = gpd.read_file(province_shape_path)[["province", "geometry"]]
    prov_poly.set_index("province", inplace=True)
    prov_poly = prov_poly.reindex(PROV_NAMES)
    prov_poly.reset_index(inplace=True)

    # load renewable profiles & grid & extract gridpoints
    cutout = atlite.Cutout(cutout_path)
    grid_points = cutout.grid
    grid_points.to_crs(3857, inplace=True)
    grid_points["geometry"] = grid_points.centroid
    grid_points.to_crs(CRS, inplace=True)

    # match cutout grid to province
    cutout_pts_in_prov = gpd.tools.sjoin(grid_points, prov_poly, how="left", predicate="intersects")
    cutout_pts_in_prov.rename(
        columns={"index_right": "province_index", "province": "province_name"},
        inplace=True,
    )

    # Province masks merged with population density
    # TODO: THIS REQUIRES EXPLANATION - can't just use random crs :||
    cutout_pts_in_prov = cutout_pts_in_prov.to_crs(3857)
    pop_ww = pop_ww.to_crs(3857)

    merged = gpd.tools.sjoin_nearest(cutout_pts_in_prov, pop_ww, how="inner")
    merged = merged.to_crs(CRS)

    # normalised pop distribution per province
    # need an extra province for points not in the province, otherwise lose cutout grid index
    merged.fillna({"province_name": "OutsideChina"}, inplace=True)
    points_in_provinces = pd.DataFrame(index=cutout_pts_in_prov.index)
    points_in_provinces = (
        merged.groupby("province_name")["pop_density"]
        .apply(lambda x: x / x.sum())
        .unstack(fill_value=0.0)
        .T
    )
    # Cleanup the matrix: get rid of the outside china "province" et
    points_in_provinces.drop(columns="OutsideChina", inplace=True)
    points_in_provinces.index.name = ""
    points_in_provinces.fillna(0.0, inplace=True)

    # go from normalised distribution to head count
    points_in_provinces *= pop_province_count
    with pd.HDFStore(gridded_pop_out, mode="w", complevel=4) as store:
        store["population_gridcell_map"] = points_in_provinces


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake("build_population_gridcell_map")

    configure_logging(snakemake, logger=logger)

    build_population_map(
        prov_pop_path=snakemake.input.province_populations,
        pop_density_raster_path=snakemake.input.population_density_grid,
        cutout_path=snakemake.input.cutout,
        province_shape_path=snakemake.input.province_shape,
        gridded_pop_out=snakemake.output.population_map,
    )

    # build_gridded_population(
    #     snakemake.input.province_population,
    #     snakemake.input.population_density_nasa,
    #     snakemake.input.cutout,
    #     snakemake.input.province_shape,
    #     "some_path",
    # )

    logger.info("Population map successfully built")
