# SPDX-FileCopyrightText: : 2024 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT
import logging
import atlite
import geopandas as gpd
import pandas as pd
from _helpers import configure_logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_cutout', cutout='China-2020')
    configure_logging(snakemake)

    cutout_params = snakemake.config['atlite']['cutouts'][snakemake.wildcards.cutout]
    snapshots = pd.date_range(freq='h', tz='Asia/shanghai', **snakemake.config['snapshots'])
    snapshots = snapshots.tz_convert('UTC')
    snapshots = snapshots.tz_localize(None)
    time = [snapshots[0], snapshots[-1]]
    cutout_params['time'] = slice(*cutout_params.get('time', time))
    
    if {'x', 'y', 'bounds'}.isdisjoint(cutout_params):
        # Determine the bounds from bus regions with a buffer of two grid cells
        onshore = gpd.read_file(snakemake.input.regions_onshore)
        offshore = gpd.read_file(snakemake.input.regions_offshore)
        regions =  onshore.append(offshore)
        d = max(cutout_params.get('dx', 0.25), cutout_params.get('dy', 0.25))*2
        cutout_params['bounds'] = regions.total_bounds + [-d, -d, d, d]
    elif {'x', 'y'}.issubset(cutout_params):
        cutout_params['x'] = slice(*cutout_params['x'])
        cutout_params['y'] = slice(*cutout_params['y'])
        
    logging.info(f"Preparing cutout with parameters {cutout_params}.")
    cutout = atlite.Cutout(snakemake.output[0], **cutout_params)
    cutout.prepare()
