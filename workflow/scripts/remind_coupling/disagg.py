""" generic disaggregation development
Split steps into:

- ETL
- disagg (also an ETL op)

to be rebalanced with the remind_coupling package"""

import pandas as pd
import logging

from rpycpl.disagg import SpatialDisaggregator
from rpycpl.etl import ETL_REGISTRY, Transformation, register_etl
from generic_etl import ETLRunner

import setup  # sets up paths
from readers import read_yearly_load_projections

logger = logging.getLogger(__name__)


@register_etl("disagg_acload_ref")
def disagg_ac_using_ref(
    data: pd.DataFrame,
    reference_data: pd.DataFrame,
    reference_year: int | str,
) -> pd.DataFrame:
    """Spatially Disaggregate the load using regional/nodal reference data
        (e.g. the projections from Hu2013 as in the Zhou et al PyPSA-China version)

    Args:
        data (pd.DataFrame): DataFrame containing the load data
        reference_data (pd.DataFrame): DataFrame containing the reference data
        reference_year (int | str): Year to use for disaggregation
    Returns:
        pd.DataFrame: Disaggregated load data (Region x Year)
    """

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
        snakemake = setup._mock_snakemake("disaggregate_data")

    logger.info(f"Available ETL functions: {ETL_REGISTRY.keys()}")

    params = snakemake.params
    region = params.region
    config = params.etl_cfg
    if not config:
        raise ValueError("Aborting: No REMIND data ETL config provided")

    # Load data
    data = {"reference_load": read_yearly_load_projections(snakemake.input.reference_load)}
    data["loads"] = pd.read_csv(snakemake.input.loads, index_col=0)
    # Can generalise with a "reader" field and data class if needed later
    for k, path in config["data"].items():
        data[k] = pd.read_csv(path)

    logger.info(f"Loaded data files {data.keys()}")

    # transform remind data
    steps = config.get("disagg", [])
    outputs = {}
    for step_dict in steps:
        step = Transformation(**step_dict)
        if step.method == "disagg_acload_ref":
            result = ETLRunner.run(
                step,
                data,
                reference_data=data["reference_load"],
                reference_year=params["reference_load_year"],
            )
        else:
            result = ETLRunner.run(step, data)
        outputs[step.name] = result

    if "disagg_load" in outputs:
        outputs["disagg_load"].to_csv(
            snakemake.output.disagg_load,
        )
