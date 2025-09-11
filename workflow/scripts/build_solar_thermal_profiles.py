"""Build solar thermal profiles for heating demand modeling.

This module generates solar thermal collector profiles and heat demand time series
for residential and commercial sectors in the PyPSA-China energy system model.
"""

import logging
import os

import atlite
import pandas as pd
import scipy as sp
from _helpers import configure_logging, mock_snakemake
from constants import TIMEZONE

logger = logging.getLogger(__name__)


def build_solar_thermal_profiles(
    pop_map: pd.DataFrame, cutout: atlite.Cutout, outp_path: os.PathLike
) -> None:
    """Build per unit solar thermal time availability profiles and save them to a file

    Args:
        pop_map (pd.DataFrame): DataFrame with the population map
        cutout (atlite.Cutout): atlite cutout object with the weather data
        outp_path (os.PathLike): Path to the output file
    """
    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "provinces"

    st = cutout.solar_thermal(
        orientation={
            "slope": float(snakemake.config["solar_thermal_angle"]),
            "azimuth": 180.0,
        },
        matrix=pop_matrix,
        index=index,
    )

    st["time"] = (
        pd.DatetimeIndex(st["time"], tz="UTC").tz_convert(TIMEZONE).tz_localize(None).values
    )

    with pd.HDFStore(outp_path, mode="w", complevel=4) as store:
        store["solar_thermal_profiles"] = st.to_pandas().divide(pop_map.sum())


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake("build_solar_thermal_profiles")

    configure_logging(snakemake, logger=logger)

    with pd.HDFStore(snakemake.input.population_map, mode="r") as store:
        pop_map = store["population_gridcell_map"]

    cutout = atlite.Cutout(snakemake.input.cutout)
    build_solar_thermal_profiles(pop_map, cutout, snakemake.output.profile_solar_thermal)

    logger.info("Solar thermal profiles successfully built")
