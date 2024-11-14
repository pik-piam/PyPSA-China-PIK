import logging
import scipy as sp
import pandas as pd
import atlite

from os import PathLike
from _helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


def build_temp_profiles(pop_map_path: PathLike, cutout_path: PathLike, temperature_out: PathLike):
    """build the temperature profiles in the cutout, this converts the atlite temperature and weights
    the node building process by the population map

    Args:
        pop_map_path (PathLike): _description_
        cutout_path (PathLike): _description_
        temperature_out (PathLike): _description_
    """
    with pd.HDFStore(pop_map_path, mode="r") as store:
        pop_map = store["population_gridcell_map"]

    # this one includes soil temperature
    cutout = atlite.Cutout(cutout_path)

    # build a sparse matrix of BUSxCUTOUT_gridcells used to weigh the cutout->bus aggregation process
    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "provinces"

    temperature = cutout.temperature(matrix=pop_matrix, index=index)
    temperature["time"] = temperature["time"].values + pd.Timedelta(
        8, unit="h"
    )  # UTC-8 instead of UTC

    with pd.HDFStore(temperature_out, mode="w", complevel=4) as store:
        store["temperature"] = temperature.to_pandas().divide(pop_map.sum())


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake("build_temp_profiles")

    configure_logging(snakemake, logger=logger)
    build_temp_profiles(
        snakemake.input.population_map, snakemake.input.cutout, snakemake.output.temp
    )

    logger.info("Temperature profiles successfully built")
