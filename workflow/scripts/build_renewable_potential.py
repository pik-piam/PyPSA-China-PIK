# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

import logging
from _helpers import configure_logging

import progressbar as pgb
import functools
import atlite
import xarray as xr
import geopandas as gpd
from atlite.gis import ExclusionContainer
import numpy as np
import time
import pandas as pd

from functions import pro_names

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_renewable_potential')
    configure_logging(snakemake)
    pgb.streams.wrap_stderr() #?

    nprocesses = int(snakemake.threads) #?
    noprogress = not snakemake.config['atlite'].get('show_progress', True)

    cutout = atlite.Cutout(snakemake.input.cutout)
    cutout.prepare()
    provinces_shp = gpd.read_file(snakemake.input.provinces_shp)[['province', 'geometry']]
    provinces_shp.replace(to_replace={'Nei Mongol': 'InnerMongolia',
                                      'Xinjiang Uygur': 'Xinjiang',
                                      'Ningxia Hui': 'Ningxia',
                                      'Xizang': 'Tibet'}, inplace=True)
    provinces_shp.set_index('province', inplace=True)
    provinces_shp = provinces_shp.reindex(pro_names).rename_axis('bus')

    buses = provinces_shp.index

    Grass = snakemake.input.Grass_raster
    Bare= snakemake.input.Bare_raster
    Shrubland = snakemake.input.Shrubland_raster

    area = cutout.grid.to_crs(3035).area / 1e6
    area = xr.DataArray(area.values.reshape(cutout.shape),
                        [cutout.coords['y'], cutout.coords['x']])

    if snakemake.config['Technique']['solar']:
        solar_config = snakemake.config['renewable']['solar']
        solar_resource = solar_config['resource']
        solar_correction_factor = solar_config.get('correction_factor', 1.)
        solar_capacity_per_sqkm = solar_config['capacity_per_sqkm']
        if solar_correction_factor != 1.:
            logger.info(f'solar_correction_factor is set as {solar_correction_factor}')

        excluder_solar = ExclusionContainer(crs=3035, res=500)
        excluder_build_up = ExclusionContainer(crs=3035, res=500)

        Build_up = snakemake.input['Build_up_raster']

        excluder_build_up.add_raster(Build_up, invert=True, crs=4326)
        excluder_solar.add_raster(Grass, invert=True, crs=4326)
        excluder_solar.add_raster(Bare, invert=True, crs=4326)
        excluder_solar.add_raster(Shrubland, invert=True, crs=4326)

        kwargs = dict(nprocesses=nprocesses, disable_progressbar=noprogress)
        if noprogress:
            logger.info('Calculate solar landuse availabilities...')
            start = time.time()
            solar_matrix = cutout.availabilitymatrix(provinces_shp, excluder_solar, **kwargs)
            buildup_matrix = cutout.availabilitymatrix(provinces_shp, excluder_build_up, **kwargs)
            duration = time.time() - start
            logger.info(f'Completed solar availability calculation ({duration:2.2f}s)')
        else:
            solar_matrix = cutout.availabilitymatrix(provinces_shp, excluder_solar, **kwargs)
            buildup_matrix = cutout.availabilitymatrix(provinces_shp, excluder_build_up, **kwargs)

        solar_potential = solar_capacity_per_sqkm * solar_matrix.sum('bus') * area + solar_capacity_per_sqkm * buildup_matrix.sum('bus') * area

        solar_func = getattr(cutout, solar_resource.pop('method'))
        solar_resource['dask_kwargs'] = {'num_workers': nprocesses} #?
        solar_capacity_factor = solar_correction_factor * solar_func(capacity_factor=True, **solar_resource)
        solar_layout = solar_capacity_factor * area * solar_capacity_per_sqkm
        solar_profile, solar_capacities = solar_func(matrix=solar_matrix.stack(spatial=['y', 'x']),
                                                     layout=solar_layout, index=buses,
                                                     per_unit=True, return_capacity=True, **solar_resource)

        logger.info(f"Calculating solar maximal capacity per bus (method 'simple')")

        solar_p_nom_max = solar_capacity_per_sqkm * solar_matrix @ area

        solar_ds = xr.merge([(solar_correction_factor * solar_profile).rename('profile'),
                       solar_capacities.rename('weight'),
                       solar_p_nom_max.rename('p_nom_max'),
                       solar_potential.rename('potential')])

        solar_ds = solar_ds.sel(bus=((solar_ds['profile'].mean('time') > solar_config.get('min_p_max_pu', 0.)) &
        (solar_ds['p_nom_max'] > solar_config.get('min_p_nom_max', 0.))))

        if 'clip_p_max_pu' in solar_config:
            min_p_max_pu = solar_config['clip_p_max_pu']
            solar_ds['profile'] = solar_ds['profile'].where(solar_ds['profile'] >= min_p_max_pu, 0)

        solar_ds['time'] = solar_ds['time'].values + pd.Timedelta(8, unit="h")  # UTC-8 instead of UTC

        solar_ds.to_netcdf(snakemake.output.solar_profile)

    if snakemake.config['Technique']['onwind']:
        onwind_config = snakemake.config['renewable']['onwind']
        onwind_resource = onwind_config['resource']
        onwind_correction_factor = onwind_config.get('correction_factor', 1.)
        onwind_capacity_per_sqkm = onwind_config['capacity_per_sqkm']
        if onwind_correction_factor != 1.:
            logger.info(f'onwind_correction_factor is set as {onwind_correction_factor}')

        excluder_onwind = ExclusionContainer(crs=3035, res=500)

        excluder_onwind.add_raster(Grass, invert=True, crs=4326)
        excluder_onwind.add_raster(Bare, invert=True, crs=4326)
        excluder_onwind.add_raster(Shrubland, invert=True, crs=4326)

        kwargs = dict(nprocesses=nprocesses, disable_progressbar=noprogress)
        if noprogress:
            logger.info('Calculate onwind landuse availabilities...')
            start = time.time()
            onwind_matrix = cutout.availabilitymatrix(provinces_shp, excluder_onwind, **kwargs)
            duration = time.time() - start
            logger.info(f'Completed onwind availability calculation ({duration:2.2f}s)')
        else:
            onwind_matrix = cutout.availabilitymatrix(provinces_shp, excluder_onwind, **kwargs)

        onwind_potential = onwind_capacity_per_sqkm * onwind_matrix.sum('bus') * area

        onwind_func = getattr(cutout, onwind_resource.pop('method'))
        onwind_resource['dask_kwargs'] = {'num_workers': nprocesses} #?
        onwind_capacity_factor = onwind_correction_factor * onwind_func(capacity_factor=True, **onwind_resource)
        onwind_layout = onwind_capacity_factor * area * onwind_capacity_per_sqkm
        onwind_profile, onwind_capacities = onwind_func(matrix=onwind_matrix.stack(spatial=['y', 'x']),
                                                        layout=onwind_layout, index=buses,
                                                        per_unit=True, return_capacity=True, **onwind_resource)

        logger.info(f"Calculating onwind maximal capacity per bus (method 'simple')")

        onwind_p_nom_max = onwind_capacity_per_sqkm * onwind_matrix @ area

        onwind_ds = xr.merge([(onwind_correction_factor * onwind_profile).rename('profile'),
                       onwind_capacities.rename('weight'),
                       onwind_p_nom_max.rename('p_nom_max'),
                       onwind_potential.rename('potential')])

        onwind_ds = onwind_ds.sel(bus=((onwind_ds['profile'].mean('time') > onwind_config.get('min_p_max_pu', 0.)) &
        (onwind_ds['p_nom_max'] > onwind_config.get('min_p_nom_max', 0.))))

        if 'clip_p_max_pu' in onwind_config:
            min_p_max_pu = onwind_config['clip_p_max_pu']
            onwind_ds['profile'] = onwind_ds['profile'].where(onwind_ds['profile'] >= min_p_max_pu, 0)

        onwind_ds['time'] = onwind_ds['time'].values + pd.Timedelta(8, unit="h")  # UTC-8 instead of UTC
        onwind_ds.to_netcdf(snakemake.output.onwind_profile)

    if snakemake.config['Technique']['offwind']:
        offwind_config = snakemake.config['renewable']['offwind']
        offwind_resource = offwind_config['resource']
        offwind_correction_factor = offwind_config.get('correction_factor', 1.)
        offwind_capacity_per_sqkm = offwind_config['capacity_per_sqkm']
        if offwind_correction_factor != 1.:
            logger.info(f'offwind_correction_factor is set as {offwind_correction_factor}')

        offwind_pro_names = np.array(['Fujian', 'Guangdong', 'Guangxi', 'Hainan', 'Hebei',
                                      'Jiangsu', 'Liaoning', 'Shandong', 'Shanghai', 'Tianjin', 'Zhejiang'],
                                     dtype=str)

        EEZ_shp = gpd.read_file(snakemake.input['offshore_shapes'])
        EEZ_province_shp = gpd.read_file(snakemake.input['offshore_province_shapes']).set_index('index')
        EEZ_province_shp = EEZ_province_shp.reindex(offwind_pro_names).rename_axis('bus')
        excluder_offwind = ExclusionContainer(crs=3035, res=500)

        if "max_depth" in offwind_config:
            func = functools.partial(np.greater, -offwind_config['max_depth'])
            excluder_offwind.add_raster(snakemake.input.gebco, codes=func, crs=4236, nodata=-1000)

        if offwind_config['natura']:
            Protected_shp = gpd.read_file(snakemake.input['natura1'])
            Protected_shp1 = gpd.read_file(snakemake.input['natura2'])
            Protected_shp2 = gpd.read_file(snakemake.input['natura3'])
            Protected_shp = pd.concat([Protected_shp,Protected_shp1], ignore_index=True)
            Protected_shp = pd.concat([Protected_shp,Protected_shp2], ignore_index=True)
            Protected_shp = Protected_shp.geometry
            Protected_shp = gpd.GeoDataFrame(Protected_shp)
            Protected_Marine_shp = gpd.tools.overlay(Protected_shp, EEZ_shp, how='intersection')
            excluder_offwind.add_geometry(Protected_Marine_shp.geometry)

        kwargs = dict(nprocesses=nprocesses, disable_progressbar=noprogress)
        if noprogress:
            logger.info('Calculate offwind landuse availabilities...')
            start = time.time()
            offwind_matrix = cutout.availabilitymatrix(EEZ_province_shp, excluder_offwind, **kwargs)
            duration = time.time() - start
            logger.info(f'Completed offwind availability calculation ({duration:2.2f}s)')
        else:
            offwind_matrix = cutout.availabilitymatrix(EEZ_province_shp, excluder_offwind, **kwargs)

        offwind_potential = offwind_capacity_per_sqkm * offwind_matrix.sum('bus') * area

        offwind_func = getattr(cutout, offwind_resource.pop('method'))
        offwind_resource['dask_kwargs'] = {'num_workers': nprocesses} #?
        offwind_capacity_factor = offwind_correction_factor * offwind_func(capacity_factor=True, **offwind_resource)
        offwind_layout = offwind_capacity_factor * area * offwind_capacity_per_sqkm
        offwind_profile, offwind_capacities = offwind_func(matrix=offwind_matrix.stack(spatial=['y', 'x']),
                                                           layout=offwind_layout, index=EEZ_province_shp.index,
                                                           per_unit=True, return_capacity=True, **offwind_resource)

        logger.info(f"Calculating offwind maximal capacity per bus (method 'simple')")

        offwind_p_nom_max = offwind_capacity_per_sqkm * offwind_matrix @ area

        offwind_ds = xr.merge([(offwind_correction_factor * offwind_profile).rename('profile'),
                       offwind_capacities.rename('weight'),
                       offwind_p_nom_max.rename('p_nom_max'),
                       offwind_potential.rename('potential')])

        offwind_ds = offwind_ds.sel(bus=((offwind_ds['profile'].mean('time') > offwind_config.get('min_p_max_pu', 0.)) &
        (offwind_ds['p_nom_max'] > offwind_config.get('min_p_nom_max', 0.))))

        if 'clip_p_max_pu' in offwind_config:
            min_p_max_pu = offwind_config['clip_p_max_pu']
            offwind_ds['profile'] = offwind_ds['profile'].where(offwind_ds['profile'] >= min_p_max_pu, 0)

        offwind_ds['time'] = offwind_ds['time'].values + pd.Timedelta(8, unit="h")  # UTC-8 instead of UTC
        offwind_ds.to_netcdf(snakemake.output.offwind_profile)



















    
    
    
    
    
    
    
    
