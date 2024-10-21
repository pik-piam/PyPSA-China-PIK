import logging
from _helpers import configure_logging

import pandas as pd
from functions import pro_names

logger = logging.getLogger(__name__)

def csv_to_df(csv_name=None):
    
    df = pd.read_csv(csv_name, index_col=0, header=0)
    
    df = df.apply(pd.to_numeric)

    return df['2020'].reindex(pro_names)

def build_population():

    population = 1.e3 * csv_to_df(csv_name=snakemake.input.population)

    population.name = "population"

    population.to_hdf(snakemake.output.population, key=population.name)


if __name__ == '__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_population')
    configure_logging(snakemake)

    build_population()
