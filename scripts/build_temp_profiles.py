import logging
from _helpers import configure_logging

import atlite

import pandas as pd
import scipy as sp

logger = logging.getLogger(__name__)

def build_temp_profiles():

    with pd.HDFStore(snakemake.input.infile, mode='r') as store:
        pop_map = store['population_gridcell_map']

    #this one includes soil temperature
    cutout = atlite.Cutout('cutouts/China-2020.nc')

    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "provinces"

    temp = cutout.temperature(matrix=pop_matrix,index=index)

    with pd.HDFStore(snakemake.output.temp, mode='w', complevel=4) as store:
        store['temperature'] = temp.to_pandas().divide(pop_map.sum())


if __name__ == '__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_temp_profiles')
    configure_logging(snakemake)

    build_temp_profiles()
