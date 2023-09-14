import logging
from _helpers import configure_logging

import atlite
import pandas as pd
import scipy as sp

logger = logging.getLogger(__name__)

def build_solar_thermal_profiles():

    with pd.HDFStore(snakemake.input.population_map, mode='r') as store:
        pop_map = store['population_gridcell_map']

    cutout = atlite.Cutout(snakemake.input.cutout)

    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "provinces"

    st = cutout.solar_thermal(orientation={'slope': float(snakemake.config['solar_thermal_angle']), 'azimuth': 180.},matrix=pop_matrix,index=index)

    st['time'] = st['time'].values + pd.Timedelta(8, unit="h") # UTC-8 instead of UTC

    with pd.HDFStore(snakemake.output.profile_solar_thermal, mode='w', complevel=4) as store:
        store['solar_thermal_profiles'] = st.to_pandas().divide(pop_map.sum())


if __name__ == '__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_solar_thermal_profiles')
    configure_logging(snakemake)

    build_solar_thermal_profiles()
