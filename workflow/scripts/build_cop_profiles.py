import logging
from _helpers import configure_logging

import atlite
import pandas as pd
import scipy as sp

logger = logging.getLogger(__name__)


def build_cop_profiles():

    with pd.HDFStore(snakemake.input.population_map, mode="r") as store:
        pop_map = store["population_gridcell_map"]

    # this one includes soil temperature
    cutout = atlite.Cutout(snakemake.input.cutout)

    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "provinces"

    soil_temp = cutout.soil_temperature(matrix=pop_matrix, index=index)
    soil_temp["time"] = soil_temp["time"].values + pd.Timedelta(8, unit="h")  # UTC-8 instead of UTC

    with pd.HDFStore(snakemake.input.temp, mode="r") as store:
        temp = store["temperature"]

    source_T = temp
    source_soil_T = soil_temp.to_pandas().divide(pop_map.sum())

    # quadratic regression based on Staffell et al. (2012)
    # https://doi.org/10.1039/C2EE22653G

    sink_T = 55.0  # Based on DTU / large area radiators

    delta_T = sink_T - source_T

    # For ASHP
    def ashp_cop(d):
        return 6.81 - 0.121 * d + 0.000630 * d**2

    cop = ashp_cop(delta_T)

    delta_soil_T = sink_T - source_soil_T

    # For GSHP
    def gshp_cop(d):
        return 8.77 - 0.150 * d + 0.000734 * d**2

    cop_soil = gshp_cop(delta_soil_T)

    with pd.HDFStore(snakemake.output.cop, mode="w", complevel=4) as store:
        store["ashp_cop_profiles"] = cop
        store["gshp_cop_profiles"] = cop_soil


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_cop_profiles")
    configure_logging(snakemake)

    build_cop_profiles()
