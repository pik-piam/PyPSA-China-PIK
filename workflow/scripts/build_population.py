"""
Rules for building the population data by region
"""

import logging
import os

import pandas as pd
from _helpers import configure_logging, mock_snakemake
from constants import POP_YEAR, PROV_NAMES, YEARBOOK_DATA2POP

logger = logging.getLogger(__name__)


def load_pop_csv(csv_path: os.PathLike) -> pd.DataFrame:
    """Load the national bureau of statistics of China population.

    Supports both formats:
    - Yearbook format (2.5 pop at year end by Region)
    - Historical data format with comment lines

    Args:
        csv_path (os.Pathlike): Path to the CSV file

    Returns:
        pd.DataFrame: The population for constants.POP_YEAR by province

    Raises:
        ValueError: If the province names do not match expected names
    """
    # Read CSV, skipping comment lines that start with #
    df = pd.read_csv(csv_path, index_col=0, header=0, comment='#')
    df = df.apply(pd.to_numeric)
    df = df[POP_YEAR][df.index.isin(PROV_NAMES)]
    if not sorted(df.index.to_list()) == sorted(PROV_NAMES):
        raise ValueError(
            f"Province names do not match {sorted(df.index.to_list())} != {sorted(PROV_NAMES)}"
        )
    return df


def build_population(data_path: os.PathLike = None):
    """Build the population data by region

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
