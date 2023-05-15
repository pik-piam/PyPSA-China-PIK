
import logging
from _helpers import configure_logging

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import atlite
import xarray as xr
from functions import pro_names

logger = logging.getLogger(__name__)

def convert_to_gdf(df):

    df.reset_index(inplace=True)

    df['Coordinates'] = list(zip(df.x, df.y))

    df['Coordinates'] = df['Coordinates'].apply(Point)

    return gpd.GeoDataFrame(df, geometry='Coordinates')


def build_population_map():

    with pd.HDFStore(snakemake.input.infile, mode='r') as store:
        pop_province_count = store['population']

    da = xr.open_dataarray('data/population/population_density_CFSR_grid.nc')

    pop_ww = da.to_dataframe(name='Population_density')

    pop_ww = convert_to_gdf(pop_ww)


    #### CFSR points and Provinces

    pro_poly = gpd.read_file('data/province_shapes/CHN_adm1.shp')[['NAME_1', 'geometry']]

    pro_poly.replace(to_replace={'Nei Mongol': 'InnerMongolia',
                                 'Xinjiang Uygur': 'Xinjiang',
                                 'Ningxia Hui': 'Ningxia',
                                 'Xizang':'Tibet'}, inplace=True)

    pro_poly.set_index('NAME_1', inplace=True)

    pro_poly = pro_poly.reindex(pro_names)

    pro_poly.reset_index(inplace=True)

    cutout = atlite.Cutout('cutouts/China-2020.nc')

    c_grid_points = cutout.grid_coordinates()


    df = pd.DataFrame()

    df['Coordinates'] = tuple(map(tuple, c_grid_points))

    df['Coordinates'] = df['Coordinates'].apply(Point)

    grid_points = gpd.GeoDataFrame(df, geometry='Coordinates')

    pointInPolys = gpd.tools.sjoin(grid_points, pro_poly, how='left', op='within')


    pointInPolys.rename(columns={'index_right': 'province_index',
                                'NAME_1': 'province_name'},
                                inplace=True)

    #### Province masks merged with population density

    merged = gpd.tools.sjoin_nearest(pointInPolys, pop_ww, how='inner')


    #### save in the right format

    points_in_provinces = pd.DataFrame(index=pointInPolys.index)

    for province in pro_names:

        pop_pro = merged[merged.province_name == province].Population_density

        points_in_provinces[province] = pop_pro / pop_pro.sum()


    points_in_provinces.index.name = ''

    points_in_provinces.fillna(0., inplace=True)

    points_in_provinces *= pop_province_count


    with pd.HDFStore(snakemake.output.population_map, mode='w', complevel=4) as store:
        store['population_gridcell_map'] = points_in_provinces



if __name__ == '__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_population_gridcell_map')
    configure_logging(snakemake)

    build_population_map()
