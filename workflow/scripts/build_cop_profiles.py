""" 
Snakemake rule script to calculate the heat pump coefficient of performance 
with atlite
"""

import atlite
import atlite.cutout
import pandas as pd
import scipy as sp
import logging
import os

from _helpers import configure_logging, mock_snakemake
from constants import TIMEZONE


logger = logging.getLogger(__name__)


# TODO cleanup
def build_cop_profiles(
        pop_map: pd.DataFrame,
        cutout: atlite.Cutout,
        temperature: pd.DataFrame,
        output_path: os.PathLike):
    """Build COP time profiles with atlite and write outputs to output_path as hf5

    Args:
        pop_map (pd.DataFrame): the population map (node resolution)
        cutout (atlite.cutout): the atlite cutout (weather data)
        temperature (pd.DataFrame): the temperature data (node resolution)
        output_path (os.PathLike): the path to write the output to as hdf5
    """

    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "provinces"

    soil_temp = cutout.soil_temperature(matrix=pop_matrix, index=index)
    soil_temp["time"] = (
        pd.DatetimeIndex(soil_temp["time"].values, tz="UTC")
        .tz_convert(TIMEZONE)
        .tz_localize(None)
        .values
    )


    source_T = temperature
    source_soil_T = soil_temp.to_pandas().divide(pop_map.sum())

    # quadratic regression based on Staffell et al. (2012)
    # https://doi.org/10.1039/C2EE22653G

    sink_T = 55.0  # Based on DTU / large area radiators

    delta_T = sink_T - source_T

    # TODO make this user set and document
    # For ASHP
    def ashp_cop(d):
        return 6.81 - 0.121 * d + 0.000630 * d**2

    cop = ashp_cop(delta_T)

    delta_soil_T = sink_T - source_soil_T

    # TODO make this user set and document
    # For GSHP
    def gshp_cop(d):
        return 8.77 - 0.150 * d + 0.000734 * d**2

    cop_soil = gshp_cop(delta_soil_T)

    with pd.HDFStore(output_path, mode="w", complevel=4) as store:
        store["ashp_cop_profiles"] = cop
        store["gshp_cop_profiles"] = cop_soil


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        snakemake = mock_snakemake("build_cop_profiles")
    configure_logging(snakemake, logger=logger)

    with pd.HDFStore(snakemake.input.temperature, mode="r") as store:
        temperature = store["temperature"]

    with pd.HDFStore(snakemake.input.population_map, mode="r") as store:
        pop_map = store["population_gridcell_map"]

    cutout = atlite.Cutout(snakemake.input.cutout)

    build_cop_profiles(pop_map, cutout, temperature, snakemake.output.cop)

    logger.info("COP profiles successfully built")
