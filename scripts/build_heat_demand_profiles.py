import logging
from _helpers import configure_logging

import atlite

import pandas as pd
import scipy as sp

logger = logging.getLogger(__name__)

def build_heat_demand_profiles():

    with pd.HDFStore(snakemake.input.infile, mode='r') as store:
        pop_map = store['population_gridcell_map']


    cutout = atlite.Cutout(snakemake.input['cutout'])

    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "provinces"

    hd = cutout.heat_demand(matrix=pop_matrix,
                            index=index,
                            threshold=15.,
                            a=1.,
                            constant=0.,
                            hour_shift=8.)
    
    Hd_2020 = hd.to_pandas().divide(pop_map.sum())
    Hd_2020.loc['2020-04-01':'2020-09-30'] = 0
    

    with pd.HDFStore(snakemake.output.daily_heat_demand, mode='w', complevel=4) as store:
        store['heat_demand_profiles'] = Hd_2020


if __name__ == '__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_heat_demand_profiles')
    configure_logging(snakemake)

    df = build_heat_demand_profiles()
