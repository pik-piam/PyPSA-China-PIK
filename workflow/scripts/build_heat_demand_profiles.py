# TODO: is this actually used?
import logging
from _helpers import configure_logging, mock_snakemake
from constants import REF_YEAR
import atlite

import pandas as pd
import scipy as sp

logger = logging.getLogger(__name__)


def build_heat_demand_profiles():

    with pd.HDFStore(snakemake.input.population_map, mode="r") as store:
        pop_map = store["population_gridcell_map"]

    cutout = atlite.Cutout(snakemake.input.cutout)

    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "provinces"

    hd = cutout.heat_demand(
        matrix=pop_matrix, index=index, threshold=15.0, a=1.0, constant=0.0, hour_shift=8.0
    )

    hd_baseyear = hd.to_pandas().divide(pop_map.sum())
    # TODO fix hardcoded range
    hd_baseyear.loc[f"{REF_YEAR}-04-01":f"{REF_YEAR}-09-30"] = 0

    with pd.HDFStore(snakemake.output.daily_heat_demand, mode="w", complevel=4) as store:
        store["heat_demand_profiles"] = hd_baseyear


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake("build_daily_heat_demand_profile")
    configure_logging(snakemake, logger=logger)

    build_heat_demand_profiles()

    logger.info("Heat demand profiles successfully built")
