import logging
from _helpers import configure_logging

import os
import pandas as pd
from functions 
from constants import PROV_NAMES, YEARBOOK_DATA2POP, POP_YEAR

logger = logging.getLogger(__name__)

def load_pop_csv(csv_path:os.PathLike)->pd.DataFrame:
    """Load the national bureau of statistics of China population
    (Yearbook - Population, table 2.5 pop at year end by Region)

    Args:
        csv_path (os.Pathlike): the csv path

    Returns:
        pd.DataFrame: the population for constants.POP_YEAR by province
    """    

    df = pd.read_csv(csv_path, index_col=0, header=0)
    df = df.apply(pd.to_numeric)
    # TODO add matching for Province PROV_names
    # TODO below line looks buggy
    # BUG: fix this?
    return df[POP_YEAR].reindex(PROV_NAMES)

def build_population():

    population = 1.e3 * load_pop_csv(csv_path=snakemake.input.population)
    population.name = "population"
    population.to_hdf(snakemake.output.population, key=population.name)


if __name__ == '__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_population')
    configure_logging(snakemake)

    build_population()
