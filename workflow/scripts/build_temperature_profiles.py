""" 
Functions associated with the build_temperature_profiles rule.

"""

import logging
import scipy as sp
import pandas as pd
import atlite

from os import PathLike
from _helpers import configure_logging, mock_snakemake
from constants import TIMEZONE

logger = logging.getLogger(__name__)


def build_temp_profiles(pop_map: pd.DataFrame, cutout: atlite.Cutout, temperature_out: PathLike):
    """build the temperature profiles in the cutout, this converts the atlite temperature & weights
    the node building process by the population map

    Note that atlite only supports a single time zone shift

    Args:
        pop_map (pd.DataFrame): the map to the pop density grid cell data (hdf5)
        cutout (atlite.Cutout): the weather data cutout (atlite cutout)
        temperature_out (PathLike): the output path (hdf5)
    """
    # build a sparse matrix of BUSxCUTOUT_gridcells to weigh the cutout->bus aggregation process
    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "provinces"

    temperature = cutout.temperature(matrix=pop_matrix, index=index)
    # convert the cutout UTC time to local time
    temperature["time"] = (
        pd.DatetimeIndex(temperature["time"], tz="UTC")
        .tz_convert(TIMEZONE)
        .tz_localize(None)
        .values
    )

    with pd.HDFStore(temperature_out, mode="w", complevel=4) as store:
        store["temperature"] = temperature.to_pandas().divide(pop_map.sum())


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake("build_temp_profiles")

    configure_logging(snakemake, logger=logger)
    
    with pd.HDFStore( snakemake.input.population_map, mode="r") as store:
        pop_map = store["population_gridcell_map"]

    # this one includes soil temperature
    cutout = atlite.Cutout(snakemake.input.cutout)
    build_temp_profiles(
        pop_map, cutout, snakemake.output.temp
    )

    logger.info("Temperature profiles successfully built")
