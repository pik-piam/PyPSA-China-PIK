
import logging
from _helpers import configure_logging

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import atlite
import xarray as xr
from constants import PROV_NAMES, CRS

logger = logging.getLogger(__name__)

def convert_to_gdf(df):
    df.reset_index(inplace=True)
    df['Coordinates'] = list(zip(df.x, df.y))
    df['Coordinates'] = df['Coordinates'].apply(Point)
    return gpd.GeoDataFrame(df, geometry='Coordinates', crs=4326)


def build_population_map():

    # =============== load data ===================
    with pd.HDFStore(snakemake.input.population, mode='r') as store:
        pop_province_count = store['population']

    da = xr.open_dataarray(snakemake.input.population_density_grid)
    pop_ww = da.to_dataframe(name='Population_density')
    pop_ww = convert_to_gdf(pop_ww)

    # CFSR points and Provinces
    prov_poly = gpd.read_file(snakemake.input.province_shape)[['province', 'geometry']]
    prov_poly.set_index('province', inplace=True)
    prov_poly = prov_poly.reindex(PROV_NAMES)
    prov_poly.reset_index(inplace=True)

    # load renewable profiles & grid & extract gridpoints
    cutout = atlite.Cutout(snakemake.input.cutout)
    c_grid_points = cutout.grid_coordinates()
    df = pd.DataFrame({"Coordinates":tuple(map(tuple, c_grid_points))})
    df['Coordinates'] = df['Coordinates'].apply(Point)
    grid_points = gpd.GeoDataFrame(df, geometry='Coordinates',crs=CRS)

    # match cutout grid to province
    cutout_pts_in_prov = gpd.tools.sjoin(grid_points, prov_poly, how='left', predicate='intersects')
    cutout_pts_in_prov.rename(columns={'index_right': 'province_index',
                                'province': 'province_name'},
                                inplace=True)

    #### Province masks merged with population density
    # TODO: THIS REQUIRES EXPLANATION - can't just use random crs :||
    cutout_pts_in_prov = cutout_pts_in_prov.to_crs(3857)
    pop_ww = pop_ww.to_crs(3857)

    merged = gpd.tools.sjoin_nearest(cutout_pts_in_prov, pop_ww, how='inner')
    merged = merged.to_crs(CRS)

    #### save in the right format

    points_in_provinces = pd.DataFrame(index=cutout_pts_in_prov.index)

    # TODO switch to native pandas with groupbyapply
    # normalise pop per province
    for province in PROV_NAMES:
        pop_pro = merged[merged.province_name == province].Population_density
        points_in_provinces[province] = pop_pro / pop_pro.sum()

    points_in_provinces.index.name = ''
    points_in_provinces.fillna(0., inplace=True)
    # re multiply by province
    points_in_provinces *= pop_province_count

    with pd.HDFStore(snakemake.output.population_map, mode='w', complevel=4) as store:
        store['population_gridcell_map'] = points_in_provinces


if __name__ == '__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_population_gridcell_map')

    configure_logging(snakemake)

    build_population_map()
