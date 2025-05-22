""" generic disaggregation development
Split steps into:

- ETL
- disagg (also an ETL op)

to be rebalanced with the remind_coupling package"""

from typing import Any
import logging
import pandas as pd
import os.path
import sys
from os import PathLike

from rpycpl.disagg import SpatialDisaggregator
from rpycpl.etl import ETL_REGISTRY, Transformation, register_etl
from generic_etl import ETLRunner

import setup  # sets up paths
from readers import read_yearly_load_projections

logger = logging.getLogger(__name__)


# TODO move to disag
@register_etl("disagg_acload_ref")
def disagg_ac_using_ref(
    data: pd.DataFrame, reference_data: pd.DataFrame, reference_year: int | str
) -> pd.DataFrame:
    """Disaggregate the load using yearly
    from Hu2013 reference data"""

    regional_reference = reference_data[int(reference_year)]
    regional_reference /= regional_reference.sum()
    electricity_demand = data["loads"].query("load == 'ac'").value
    logger.info("Disaggregating load according to Hu et al. demand projections")
    disagg_load = SpatialDisaggregator().use_static_reference(
        electricity_demand, regional_reference
    )

    return disagg_load


if __name__ == "__main__":

    
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        snakemake = setup._mock_snakemake(
            "rule disaggregate_data:
")
        
    logger.info(f"Available ETL functions: {ETL_REGISTRY.keys()}")