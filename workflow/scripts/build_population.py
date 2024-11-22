import logging
from _helpers import configure_logging, mock_snakemake

import os
import pandas as pd
from constants import PROV_NAMES, YEARBOOK_DATA2POP, POP_YEAR

logger = logging.getLogger(__name__)


def load_pop_csv(csv_path: os.PathLike) -> pd.DataFrame:
    """Load the national bureau of statistics of China population
    (Yearbook - Population, table 2.5 pop at year end by Region)

    Args:
        csv_path (os.Pathlike): the csv path

    Returns:
        pd.DataFrame: the population for constants.POP_YEAR by province
    Raises:
        ValueError: if the province names are not as expected
    """

    df = pd.read_csv(csv_path, index_col=0, header=0)
    df = df.apply(pd.to_numeric)
    df = df[POP_YEAR][df.index.isin(PROV_NAMES)]
    if not sorted(df.index.to_list()) == sorted(PROV_NAMES):
        raise ValueError(
            f"Province names do not match {sorted(df.index.to_list())} != {sorted(PROV_NAMES)}"
        )
    return df


def build_population(data_path: os.PathLike = None):
    """read the population csv, ocnvert to unit pop (head count) and make an hdf5 file

    Args:
        data_path (os.PathLike, optional): the path to the pop csv. Defaults to None.
    """

    if data_path is None:
        data_path = snakemake.input.population

    population = YEARBOOK_DATA2POP * load_pop_csv(csv_path=data_path)
    population.name = "population"
    population.to_hdf(snakemake.output.population, key=population.name)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake("build_population")

    configure_logging(snakemake, logger=logger)
    build_population()
    logger.info("Population successfully built")
